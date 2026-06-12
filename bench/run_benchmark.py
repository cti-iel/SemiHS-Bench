#!/usr/bin/env python3
"""Config-driven SemiHS-Bench benchmark runner.

Run from the repository root:

    uv run python3 -m bench.run_benchmark --config bench/configs/smoke.mock.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

from lib import (  # noqa: E402
    extract_hs6_codes,
    get_tier_input,
    hierarchical_distance,
    in_scope_hs6_codes,
    load_taxonomy_csv,
    mrr,
    parse_ranked_indices,
    render_candidates_block,
    top_k_accuracy,
)
from score_submission import (  # noqa: E402
    SubmissionError,
    _index_records,
    format_markdown,
    score,
    validate,
)

try:  # Support both package and direct-script execution.
    from .langsmith_integration import LangSmithRecorder, sanitize_record_for_langsmith
    from .providers import ProviderError, ProviderRequest, ProviderResponse, build_provider, to_jsonable
except ImportError:  # pragma: no cover - direct script fallback
    from bench.langsmith_integration import LangSmithRecorder, sanitize_record_for_langsmith
    from bench.providers import ProviderError, ProviderRequest, ProviderResponse, build_provider, to_jsonable


BENCHMARK_VERSION = "2.0.0"
DEFAULT_SCHEMA_VERSION = "2.0.0"
TRUTHY_WEB_MODES = {"on", "true", "web", "unrestricted", "restricted"}
MIN_MAX_OUTPUT_TOKENS = 10000

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?(.*?)\n?```$", re.DOTALL)

_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "resource exhausted",
    "quota exceeded",
    "too many requests",
    "ratelimitexceeded",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def _compute_retry_sleep(attempt: int, base_s: float, *, rate_limited: bool) -> float:
    """Exponential backoff with jitter. Rate-limit hits use a longer floor (30s)."""
    if rate_limited:
        wait = min(max(30.0, base_s) * (2 ** attempt), 120.0) + random.uniform(0, 5.0)
    else:
        wait = min(base_s * (2 ** attempt), 60.0) + random.uniform(0, 1.0)
    return wait


def strip_code_fence(text: str) -> str:
    """Remove markdown code fences (```json ... ```) that some models wrap around JSON."""
    stripped = text.strip()
    m = _CODE_FENCE_RE.match(stripped)
    return m.group(1).strip() if m else stripped


def try_parse_json(text: str) -> Any:
    """JSON parse with code-fence stripping as fallback."""
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    try:
        return json.loads(strip_code_fence(raw))
    except ValueError:
        return None


@dataclass
class PromptBundle:
    system_name: str
    user_name: str
    system_template: str
    user_template: str
    prompt_hash: str
    prompt_version: str
    prompt_dir: str


@dataclass
class Combo:
    provider_config: Mapping[str, Any]
    provider_name: str
    model_name: str
    model_id: str
    reasoning_level: Optional[str]
    web_mode: str
    web_enabled: bool
    slug: str


@dataclass
class RecordJob:
    idx: int
    record: Mapping[str, Any]
    frozen_id: str
    system: str
    user: str
    candidate_codes: List[str]
    tier_text: str
    sanitized: Mapping[str, Any]
    trace_inputs: Mapping[str, Any]
    trace_meta: Mapping[str, Any]
    started_iso: str
    trace_id: Optional[str] = None


@dataclass
class RecordResult:
    job: RecordJob
    provider_response: Optional[ProviderResponse]
    parsed: Dict[str, Any]
    error_text: Optional[str]
    failures: List[Dict[str, Any]]
    ended_iso: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_window(seconds: float = 1.0) -> Dict[str, str]:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=seconds)
    return {"start": start.isoformat(), "end": end.isoformat()}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _strip_env_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def load_dotenv_files(config: Mapping[str, Any], config_path: Optional[Path] = None) -> List[str]:
    """Load simple KEY=VALUE pairs from .env files without adding a dependency.

    Existing environment variables win by default. Set ``env.override`` to true
    in the benchmark config if a .env file should replace existing values.
    """
    env_cfg = dict(config.get("env") or {})
    if env_cfg.get("load_dotenv", True) is False:
        return []
    configured = env_cfg.get("files")
    candidates: List[Path] = []
    if configured:
        for item in configured:
            path = Path(str(item))
            if not path.is_absolute():
                base = config_path.parent if config_path else ROOT
                path = base / path
            candidates.append(path)
    else:
        if config_path:
            candidates.append(config_path.parent / ".env")
        candidates.extend([Path.cwd() / ".env", ROOT / ".env"])

    loaded: List[str] = []
    seen: set = set()
    override = bool(env_cfg.get("override", False))
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export "):].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                continue
            if not override and key in os.environ:
                continue
            os.environ[key] = _strip_env_quotes(value)
        loaded.append(str(path))
    return loaded


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def append_jsonl_gz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:180] or "run"


def normalize_hs(value: object, width: int = 6) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())[:width]


def merge_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    data = dict(config.get("data") or {})
    run = dict(config.get("run") or {})
    langsmith = dict(config.get("langsmith") or {})
    if args.split:
        data["split"] = args.split
    if args.mode:
        data["mode"] = args.mode
    if args.tier:
        data["tiers"] = [int(t) for t in args.tier]
    if args.limit is not None:
        data["limit"] = args.limit
    if args.out:
        run["out_dir"] = args.out
    if getattr(args, "max_parallel_requests", None) is not None:
        run["max_parallel_requests"] = args.max_parallel_requests
    if getattr(args, "max_parallel_combos", None) is not None:
        run["max_parallel_combos"] = args.max_parallel_combos
    if getattr(args, "parallel_models", None):
        run["parallel_models"] = True
    if getattr(args, "rerun_errors", False):
        run["rerun_errors"] = True
    if getattr(args, "tavily_fallback", False):
        for model_cfg in config.get("models", []):
            model_cfg["tavily_fallback"] = True
    config["data"] = data
    config["run"] = run
    if args.langsmith_mode:
        langsmith["mode"] = args.langsmith_mode
    if args.langsmith_project:
        langsmith["project"] = args.langsmith_project
    if args.langsmith_dataset_name:
        langsmith["dataset_name"] = args.langsmith_dataset_name
    if langsmith:
        config["langsmith"] = langsmith
    if args.model:
        wanted = set(args.model)
        config["models"] = [
            m for m in config.get("models", [])
            if str(m.get("name") or m.get("model") or m.get("model_id")) in wanted
        ]
    if args.reasoning_level is not None or args.web_mode is not None:
        updated_models = []
        for model_cfg in config.get("models", []) or []:
            model_cfg = dict(model_cfg)
            if args.reasoning_level is not None:
                model_cfg["reasoning_levels"] = [
                    None if value == "default" else value
                    for value in args.reasoning_level
                ]
            if args.web_mode is not None:
                model_cfg["web_modes"] = list(args.web_mode)
            updated_models.append(model_cfg)
        config["models"] = updated_models
    return config


def resolve_data_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    data = dict(config.get("data") or {})
    # The benchmark ships matched train + eval splits; default data file is the eval split (eval.json)
    default_data = ROOT / "data" / "eval.json"
    return {
        "dataset": Path(data.get("path") or default_data),
        "taxonomy": Path(data.get("taxonomy_path") or ROOT / "data" / "taxonomy.csv"),
        "manifest": Path(data.get("manifest_path") or ROOT / "data" / "MANIFEST.json"),
    }


def load_prompt_file(prompt_dir: Path, name: str) -> str:
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".txt")
    if not path.is_absolute():
        path = prompt_dir / path
    text = path.read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def load_prompt_bundle(config: Mapping[str, Any], mode: str, *, web_mode: str = "off") -> PromptBundle:
    prompt_cfg = dict(config.get("prompts") or {})
    prompt_dir = Path(prompt_cfg.get("dir") or ROOT / "bench" / "prompts")
    if not prompt_dir.is_absolute():
        prompt_dir = ROOT / prompt_dir
    web_enabled = web_mode.lower() in TRUTHY_WEB_MODES
    if mode in {"open", "free"} and web_enabled:
        default_system = f"{mode}_web_topk_system"
        default_user = f"{mode}_web_topk_user"
    else:
        default_system = f"{mode}_topk_system"
        default_user = f"{mode}_topk_user"
    system_name = str(prompt_cfg.get("system") or default_system)
    user_name = str(prompt_cfg.get("user") or default_user)
    system_template = load_prompt_file(prompt_dir, system_name)
    user_template = load_prompt_file(prompt_dir, user_name)
    digest = sha256_text(system_template + "\n---USER---\n" + user_template)
    prompt_version = str(prompt_cfg.get("version") or prompt_cfg.get("prompt_version") or f"{mode}:{digest[:8]}")
    return PromptBundle(
        system_name=system_name,
        user_name=user_name,
        system_template=system_template,
        user_template=user_template,
        prompt_hash=digest,
        prompt_version=prompt_version,
        prompt_dir=str(prompt_dir),
    )


def build_combos(config: Mapping[str, Any], mode: str, tier: int) -> List[Combo]:
    combos: List[Combo] = []
    for provider_cfg in config.get("models", []) or []:
        provider_name = str(provider_cfg.get("provider") or "")
        model_id = str(provider_cfg.get("model") or provider_cfg.get("model_id") or "")
        model_name = str(provider_cfg.get("name") or model_id or provider_name)
        reasoning_levels = provider_cfg.get("reasoning_levels")
        if reasoning_levels is None:
            reasoning_levels = [provider_cfg.get("reasoning_level")]
        web_modes = provider_cfg.get("web_modes")
        if web_modes is None:
            web_modes = [provider_cfg.get("web_mode") or "off"]
        for reasoning in reasoning_levels:
            reasoning_text = None if reasoning in (None, "", "default") else str(reasoning)
            for web_mode_obj in web_modes:
                web_mode = str(web_mode_obj or "off")
                web_enabled = web_mode.lower() in TRUTHY_WEB_MODES
                slug = sanitize_slug(
                    "__".join([
                        model_name,
                        f"{mode}-tier{tier}",
                        f"reasoning-{reasoning_text or 'default'}",
                        f"web-{web_mode}",
                    ])
                )
                combos.append(Combo(
                    provider_config=provider_cfg,
                    provider_name=provider_name,
                    model_name=model_name,
                    model_id=model_id,
                    reasoning_level=reasoning_text,
                    web_mode=web_mode,
                    web_enabled=web_enabled,
                    slug=slug,
                ))
    return combos


def build_prompt(
    record: Mapping[str, Any],
    *,
    mode: str,
    tier: int,
    taxonomy: Mapping[str, Any],
    open_codes: Sequence[str],
    top_k: int,
    prompts: PromptBundle,
) -> Tuple[str, str, List[str], str]:
    tier_text = get_tier_input(record, tier)
    if mode == "constrained":
        candidate_codes = list((record.get("candidate_set") or {}).get("codes") or [])
        if not candidate_codes:
            raise ValueError(f"record {record.get('frozen_id')} has no candidate_set")
        candidates_block = render_candidates_block(candidate_codes, taxonomy=taxonomy)
        user = prompts.user_template.format(
            tier_text=tier_text,
            candidates_block=candidates_block,
            num_candidates=len(candidate_codes),
            top_k=top_k,
        )
    elif mode in {"open", "free"}:
        candidate_codes = []
        user = prompts.user_template.format(
            tier_text=tier_text,
            top_k=top_k,
        )
    else:
        raise ValueError(f"Unsupported mode {mode!r}")
    return prompts.system_template, user, candidate_codes, tier_text


def parse_constrained_response(text: str, candidate_codes: Sequence[str]) -> Dict[str, Any]:
    issues: List[str] = []
    expected = len(candidate_codes)
    candidates_by_code = {normalize_hs(code): str(code) for code in candidate_codes}
    ranked_codes: List[str] = []
    strict = False
    payload = try_parse_json(text)
    if isinstance(payload, list):
        if all(isinstance(item, int) or (isinstance(item, str) and item.strip().lstrip("-").isdigit()) for item in payload):
            indices: List[int] = []
            seen = set()
            for item in payload:
                idx = int(item)
                if 0 <= idx < expected and idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
            if len(indices) == expected:
                ranked_codes = [str(candidate_codes[i]) for i in indices]
                strict = True
        if not ranked_codes and all(isinstance(item, str) for item in payload):
            seen_codes = set()
            for item in payload:
                code = normalize_hs(item)
                if code in candidates_by_code and code not in seen_codes:
                    ranked_codes.append(candidates_by_code[code])
                    seen_codes.add(code)
            if len(ranked_codes) == expected:
                strict = True
    if not ranked_codes:
        seen_codes = set()
        for code in extract_hs6_codes(text or ""):
            if code in candidates_by_code and code not in seen_codes:
                ranked_codes.append(candidates_by_code[code])
                seen_codes.add(code)
        if ranked_codes:
            issues.append("parsed_hs6_codes_instead_of_indices")
    if len(ranked_codes) < expected:
        indices = parse_ranked_indices(text or "", expected=expected)
        repaired = [str(candidate_codes[i]) for i in indices]
        merged = ranked_codes + [code for code in repaired if code not in ranked_codes]
        ranked_codes = merged[:expected]
        if not strict:
            issues.append("ranking_repaired")
    return {
        "ranked_codes": ranked_codes,
        "parse_valid": strict,
        "parse_issues": issues,
    }


def parse_open_response(text: str, *, allowed_codes: Sequence[str], top_k: int) -> Dict[str, Any]:
    allowed = set(allowed_codes)
    codes: List[str] = []
    strict = False
    issues: List[str] = []
    payload = try_parse_json(text)
    if isinstance(payload, list):
        seen = set()
        for item in payload:
            code = normalize_hs(item)
            if len(code) == 6 and code not in seen:
                seen.add(code)
                codes.append(code)
        strict = bool(codes)
    if not codes:
        codes = extract_hs6_codes(text or "")
        if codes:
            issues.append("parsed_hs6_codes_from_free_text")
    if not codes:
        codes = ["000000"]
        issues.append("no_hs6_codes_found")
    outside = [code for code in codes if code not in allowed]
    if outside:
        issues.append(f"{len(outside)}_codes_outside_taxonomy")
    return {
        "ranked_codes": codes[:top_k],
        "parse_valid": strict and not outside,
        "parse_issues": issues,
        "outside_taxonomy": outside,
    }


def parse_response(text: str, *, mode: str, candidate_codes: Sequence[str], open_codes: Sequence[str], top_k: int) -> Dict[str, Any]:
    if mode == "constrained":
        return parse_constrained_response(text, candidate_codes)
    return parse_open_response(text, allowed_codes=open_codes, top_k=top_k)


def per_record_scores(ranked_codes: Sequence[str], gold: str, parse_valid: bool, web_used: bool) -> List[Dict[str, Any]]:
    preds = [list(ranked_codes)]
    labels = [gold]
    first = ranked_codes[0] if ranked_codes else ""
    return [
        {"key": "hs6_top1", "score": top_k_accuracy(preds, labels, k=1, level="hs6"), "feedback_source": {"type": "model"}},
        {"key": "hs6_top3", "score": top_k_accuracy(preds, labels, k=3, level="hs6"), "feedback_source": {"type": "model"}},
        {"key": "hs4_top1", "score": top_k_accuracy(preds, labels, k=1, level="hs4"), "feedback_source": {"type": "model"}},
        {"key": "hs2_top1", "score": top_k_accuracy(preds, labels, k=1, level="hs2"), "feedback_source": {"type": "model"}},
        {"key": "mrr", "score": mrr(preds, labels), "feedback_source": {"type": "model"}},
        {"key": "hierarchical_distance", "score": hierarchical_distance(first, gold), "feedback_source": {"type": "model"}},
        {"key": "parse_valid", "score": 1.0 if parse_valid else 0.0, "feedback_source": {"type": "model"}},
        {"key": "web_used", "score": 1.0 if web_used else 0.0, "feedback_source": {"type": "model"}},
    ]


def effective_max_output_tokens(run_cfg: Mapping[str, Any], mode: str) -> int:
    configured = int(run_cfg.get("max_output_tokens") or (512 if mode == "constrained" else 1024))
    floor = int(run_cfg.get("minimum_max_output_tokens") or MIN_MAX_OUTPUT_TOKENS)
    return max(configured, floor)


def build_trace_outputs(
    *,
    provider_response: Any,
    parsed: Mapping[str, Any],
    ranked_codes: Sequence[str],
    include_raw_provider_response: bool = False,
) -> Dict[str, Any]:
    model_output = provider_response.text if provider_response else ""
    provider_blob: Dict[str, Any] = {
        "text": model_output,
        "usage": provider_response.usage if provider_response else {},
        "citations": provider_response.citations if provider_response else [],
        "tool_calls": provider_response.tool_calls if provider_response else [],
        "reasoning_summaries": provider_response.thought_summaries if provider_response else [],
        "warnings": provider_response.warnings if provider_response else [],
        "elapsed_s": provider_response.elapsed_s if provider_response else None,
    }
    if include_raw_provider_response and provider_response:
        provider_blob["raw"] = provider_response.raw
    parsed_result = {
        "ranked_codes": list(ranked_codes),
        "parse_valid": parsed.get("parse_valid"),
        "parse_issues": parsed.get("parse_issues"),
        "outside_taxonomy": parsed.get("outside_taxonomy"),
    }
    return {
        "model_output": model_output,
        "reasoning_summaries": provider_blob["reasoning_summaries"],
        "provider_response": provider_blob,
        "parsed_result": parsed_result,
        "ranked_codes": list(ranked_codes),
        "response_text": model_output,
        "parse_valid": parsed.get("parse_valid"),
        "parse_issues": parsed.get("parse_issues"),
    }


def summary_scores_from_report(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    overall = report.get("overall") or {}
    keys = ["hs6_top1", "hs6_top3", "hs4_top1", "hs2_top1", "mrr", "mean_hier_dist"]
    return [
        {"key": f"summary_{key}", "score": float(overall[key]), "feedback_source": {"type": "model"}}
        for key in keys
        if key in overall
    ]


def dataset_metadata_for_record(
    record: Mapping[str, Any],
    *,
    split: str,
    mode: str,
    tier: int,
    prompt_hash: str,
    prompt_version: Optional[str],
    dataset_hash: Optional[str],
    taxonomy_hash: Optional[str],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    tags = list(record.get("difficulty_tags") or [])
    hashes = manifest.get("hashes") or {}
    file_hashes = hashes.get("file_sha256") or {}
    return {
        "benchmark": "SemiHS-Bench",
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": BENCHMARK_VERSION,
        "split": split,
        "mode": mode,
        "tier": tier,
        "prompt_hash": prompt_hash,
        "prompt_version": prompt_version,
        "dataset_file_sha256": dataset_hash,
        "taxonomy_file_sha256": taxonomy_hash,
        "manifest_eval_file_sha256": file_hashes.get("eval.json"),
        "frozen_id": record.get("frozen_id"),
        "record_id": record.get("id"),
        "confidence_tier": record.get("confidence_tier"),
        "segment": record.get("segment"),
        "hs2": record.get("hs2_label"),
        "hs4": record.get("hs4_label"),
        "hs6": record.get("hs6_label"),
        "difficulty_tags": tags,
        "difficulty_tags_joined": ",".join(tags),
        "has_difficulty_tags": bool(tags),
        "candidate_gold_rank": (record.get("candidate_set") or {}).get("gold_rank_in_candidates"),
        "tier2_classifiable": record.get("tier2_classifiable"),
        "tier1_source": record.get("tier1_source"),
    }


def langsmith_dataset_name(
    langsmith_config: Mapping[str, Any],
    *,
    split: str,
    mode: str,
    tier: int,
    prompt_hash: str,
) -> str:
    prompt8 = prompt_hash[:8]
    fields = {
        "split": split,
        "mode": mode,
        "tier": tier,
        "prompt_hash": prompt_hash,
        "prompt8": prompt8,
        "benchmark_version": BENCHMARK_VERSION,
    }
    template = langsmith_config.get("dataset_name_template")
    if template:
        return str(template).format(**fields)
    dataset_name = langsmith_config.get("dataset_name")
    if dataset_name:
        text = str(dataset_name)
        return text.format(**fields) if "{" in text else text
    return f"semihs-bench-v{BENCHMARK_VERSION}-{split}-{mode}-tier{tier}"


def sanitize_langsmith_example(
    record: Mapping[str, Any],
    *,
    split: str,
    mode: str,
    tier: int,
    taxonomy_hash: Optional[str],
    taxonomy_count: int,
    prompt_hash: str,
    prompt_version: Optional[str],
    dataset_hash: Optional[str],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_codes = list((record.get("candidate_set") or {}).get("codes") or []) if mode == "constrained" else []
    sanitized = sanitize_record_for_langsmith(
        record,
        tier_input=get_tier_input(record, tier),
        mode=mode,
        candidate_codes=candidate_codes,
        taxonomy_hash=taxonomy_hash,
        taxonomy_count=taxonomy_count,
        prompt_hash=prompt_hash,
        prompt_version=prompt_version,
        benchmark_version=BENCHMARK_VERSION,
    )
    sanitized["inputs"]["split"] = split
    sanitized["inputs"]["tier"] = tier
    sanitized["metadata"].update(dataset_metadata_for_record(
        record,
        split=split,
        mode=mode,
        tier=tier,
        prompt_hash=prompt_hash,
        prompt_version=prompt_version,
        dataset_hash=dataset_hash,
        taxonomy_hash=taxonomy_hash,
        manifest=manifest,
    ))
    return sanitized


def upload_langsmith_reference_datasets(
    *,
    config: Mapping[str, Any],
    active_split: str,
    mode: str,
    tier: int,
    prompts: PromptBundle,
    taxonomy_count: int,
    taxonomy_hash: Optional[str],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    langsmith_config = dict(config.get("langsmith") or {"mode": "off"})
    if str(langsmith_config.get("mode") or "off") not in {"dataset_experiment", "both"}:
        return {"enabled": False, "reason": "langsmith mode is not dataset_experiment/both"}
    if langsmith_config.get("upload_reference_datasets", True) is False:
        return {"enabled": False, "reason": "upload_reference_datasets is false"}
    recorder = LangSmithRecorder(langsmith_config, run_name=str(config.get("name") or "semihs-reference-datasets"))
    out: Dict[str, Any] = {"enabled": True, "datasets": [], "warnings": recorder.warnings}
    dataset_path = paths["dataset"]
    records = read_json(dataset_path)
    dataset_hash = sha256_file(dataset_path)
    dataset_name = langsmith_dataset_name(
        langsmith_config,
        split=active_split,
        mode=mode,
        tier=tier,
        prompt_hash=prompts.prompt_hash,
    )
    examples = [
        sanitize_langsmith_example(
            record,
            split=active_split,
            mode=mode,
            tier=tier,
            taxonomy_hash=taxonomy_hash,
            taxonomy_count=taxonomy_count,
            prompt_hash=prompts.prompt_hash,
            prompt_version=prompts.prompt_version,
            dataset_hash=dataset_hash,
            manifest=manifest,
        )
        for record in records
    ]
    status = recorder.upload_reference_dataset(
        dataset_name=dataset_name,
        examples=examples,
        description=(
            f"SemiHS-Bench v{BENCHMARK_VERSION} {active_split} split, "
            f"{mode}, tier {tier}, prompt {prompts.prompt_version} ({prompts.prompt_hash[:8]}). "
            "Local benchmark files remain canonical."
        ),
        metadata={
            "benchmark": "SemiHS-Bench",
            "benchmark_version": BENCHMARK_VERSION,
            "split": active_split,
            "mode": mode,
            "tier": tier,
            "prompt_hash": prompts.prompt_hash,
            "prompt_version": prompts.prompt_version,
            "dataset_file_sha256": dataset_hash,
            "taxonomy_file_sha256": taxonomy_hash,
            "n_examples": len(examples),
        },
    )
    out["datasets"].append(status)
    out["warnings"] = recorder.warnings
    return out


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for pred in payload.get("predictions", []) or []:
        if pred.get("frozen_id"):
            out[str(pred["frozen_id"])] = dict(pred)
    return out


def _load_errored_frozen_ids(calls_path: Path) -> set:
    """Return frozen_ids of records whose last call entry has status=error."""
    seen: Dict[str, str] = {}  # frozen_id -> last status
    if not calls_path.exists():
        return set()
    try:
        with gzip.open(calls_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    fid = str(entry.get("frozen_id") or "")
                    if fid:
                        seen[fid] = str(entry.get("status") or "")
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        return set()
    return {fid for fid, status in seen.items() if status == "error"}


def _is_error_prediction(pred: Mapping[str, Any]) -> bool:
    """True for the sentinel prediction written when all retries fail."""
    codes = list(pred.get("ranked_codes") or [])
    return codes == ["000000"]


def write_submission(path: Path, meta: Mapping[str, Any], predictions_by_id: Mapping[str, Mapping[str, Any]], record_order: Sequence[str]) -> Dict[str, Any]:
    predictions = [dict(predictions_by_id[fid]) for fid in record_order if fid in predictions_by_id]
    submission = {"submission": dict(meta), "predictions": predictions}
    write_json(path, submission)
    return submission


def score_submission(submission: Mapping[str, Any], records: Sequence[Mapping[str, Any]], report_md: Path, report_json: Path) -> Dict[str, Any]:
    by_frozen = _index_records(records)
    parsed = validate(submission, by_frozen)
    report = score(parsed)
    report_md.write_text(format_markdown(report), encoding="utf-8")
    write_json(report_json, report)
    return report


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def report_to_summary_row(combo: Combo, run_dir: Path, report: Mapping[str, Any], prompts: PromptBundle) -> Dict[str, Any]:
    overall = report.get("overall") or {}
    return {
        "run": combo.slug,
        "provider": combo.provider_name,
        "model": combo.model_id,
        "reasoning_level": combo.reasoning_level or "",
        "web_mode": combo.web_mode,
        "prompt_hash": prompts.prompt_hash,
        "prompt_version": prompts.prompt_version,
        "n": overall.get("n"),
        "hs6_top1": overall.get("hs6_top1"),
        "hs6_top3": overall.get("hs6_top3"),
        "hs4_top1": overall.get("hs4_top1"),
        "hs2_top1": overall.get("hs2_top1"),
        "mrr": overall.get("mrr"),
        "mean_hier_dist": overall.get("mean_hier_dist"),
        "run_dir": str(run_dir),
    }


def execute_record_job(
    *,
    job: RecordJob,
    provider: Any,
    combo: Combo,
    mode: str,
    tier: int,
    open_codes: Sequence[str],
    top_k: int,
    max_retries: int,
    retry_sleep_s: float,
    temperature: Optional[float],
    max_output_tokens: int,
    timeout_s: Optional[float],
    run_cfg: Mapping[str, Any],
) -> RecordResult:
    error_text: Optional[str] = None
    provider_response: Optional[ProviderResponse] = None
    parsed: Dict[str, Any] = {}
    failures: List[Dict[str, Any]] = []
    for attempt in range(max_retries + 1):
        try:
            provider_request = ProviderRequest(
                provider=combo.provider_name,
                model_id=combo.model_id,
                system=job.system,
                user=job.user,
                mode=mode,
                tier=tier,
                frozen_id=job.frozen_id,
                reasoning_level=combo.reasoning_level,
                web_enabled=combo.web_enabled,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
                extra={
                    "search_context_size": run_cfg.get("search_context_size"),
                    "web_max_uses": run_cfg.get("web_max_uses"),
                },
                private={
                    "record": job.record,
                    "candidate_codes": job.candidate_codes,
                    "gold_code": job.record.get("hs6_label"),
                    "open_codes": open_codes,
                },
            )
            provider_response = provider.complete(provider_request)
            parsed = parse_response(
                provider_response.text,
                mode=mode,
                candidate_codes=job.candidate_codes,
                open_codes=open_codes,
                top_k=top_k,
            )
            error_text = None
            break
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            rate_limited = _is_rate_limit_error(exc)
            failures.append({
                "time": utc_now(),
                "frozen_id": job.frozen_id,
                "attempt": attempt,
                "max_retries": max_retries,
                "error": error_text,
                "rate_limited": rate_limited,
                "traceback": traceback.format_exc(),
            })
            if attempt >= max_retries:
                parsed = {"ranked_codes": ["000000"], "parse_valid": False, "parse_issues": ["provider_error"]}
            else:
                time.sleep(_compute_retry_sleep(attempt, retry_sleep_s, rate_limited=rate_limited))
    return RecordResult(
        job=job,
        provider_response=provider_response,
        parsed=parsed,
        error_text=error_text,
        failures=failures,
        ended_iso=utc_now(),
    )


def run_combo(
    *,
    config: Mapping[str, Any],
    combo: Combo,
    records: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    open_codes: Sequence[str],
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    prompts: PromptBundle,
    out_root: Path,
) -> Dict[str, Any]:
    data_cfg = dict(config.get("data") or {})
    run_cfg = dict(config.get("run") or {})
    mode = str(data_cfg.get("mode") or "constrained")
    split = str(data_cfg.get("split") or "eval")
    tier = int(data_cfg.get("tier") or 1)
    top_k = int(data_cfg.get("top_k") or (4 if mode == "constrained" else 5))
    max_retries = int(run_cfg.get("max_retries") or 2)
    resume = bool(run_cfg.get("resume", True))
    configured_max_output_tokens = int(run_cfg.get("max_output_tokens") or (512 if mode == "constrained" else 1024))
    max_output_tokens = effective_max_output_tokens(run_cfg, mode)
    run_cfg["configured_max_output_tokens"] = configured_max_output_tokens
    run_cfg["max_output_tokens"] = max_output_tokens
    run_cfg["minimum_max_output_tokens"] = int(run_cfg.get("minimum_max_output_tokens") or MIN_MAX_OUTPUT_TOKENS)
    max_parallel_requests = max(1, int(run_cfg.get("max_parallel_requests") or 1))
    run_cfg["max_parallel_requests"] = max_parallel_requests
    checkpoint_every = max(1, int(run_cfg.get("checkpoint_every") or 1))
    retry_sleep_s = float(run_cfg.get("retry_sleep_s") or 1.0)
    temperature = run_cfg.get("temperature", 0.0)
    timeout_s = run_cfg.get("request_timeout_s")
    run_dir = out_root / combo.slug
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = run_dir / "submission.json"
    calls_path = run_dir / "calls.jsonl.gz"
    failures_path = run_dir / "failures.jsonl"
    report_md = run_dir / "report.md"
    report_json = run_dir / "report.json"
    resolved_config = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "combo": asdict(combo),
        "data": data_cfg,
        "run": run_cfg,
        "prompts": asdict(prompts),
        "paths": {k: str(v) for k, v in paths.items()},
        "hashes": {
            "dataset_file_sha256": sha256_file(paths["dataset"]),
            "taxonomy_file_sha256": sha256_file(paths["taxonomy"]) if paths["taxonomy"].exists() else None,
            "manifest": manifest.get("hashes", {}),
        },
    }
    write_json(run_dir / "run_config.resolved.json", resolved_config)
    dataset_hash = resolved_config["hashes"]["dataset_file_sha256"]
    taxonomy_hash = resolved_config["hashes"]["taxonomy_file_sha256"]
    predictions_by_id = load_existing_predictions(submission_path) if resume else {}
    rerun_errors = bool(run_cfg.get("rerun_errors", False))
    if resume and rerun_errors and predictions_by_id:
        errored_ids = _load_errored_frozen_ids(calls_path)
        sentinel_ids = {fid for fid, pred in predictions_by_id.items() if _is_error_prediction(pred)}
        to_rerun = errored_ids | sentinel_ids
        if to_rerun:
            for fid in to_rerun:
                predictions_by_id.pop(fid, None)
            print(f"[{combo.slug}] rerun_errors: cleared {len(to_rerun)} failed predictions for retry")
    record_order = [str(r["frozen_id"]) for r in records]
    meta = {
        "name": combo.slug,
        "model_id": combo.model_id,
        "mode": mode,
        "tier": tier,
        "schema_version": BENCHMARK_VERSION,
        "notes": (
            f"provider={combo.provider_name}; reasoning={combo.reasoning_level or 'default'}; "
            f"web={combo.web_mode}; prompt_version={prompts.prompt_version}; prompt_hash={prompts.prompt_hash}"
        ),
    }
    langsmith_config = dict(config.get("langsmith") or {"mode": "off"})
    include_raw_provider_response = bool(langsmith_config.get("include_raw_provider_response", False))
    if str(langsmith_config.get("mode") or "off") in {"dataset_experiment", "both"}:
        langsmith_config["dataset_name"] = langsmith_dataset_name(
            langsmith_config,
            split=split,
            mode=mode,
            tier=tier,
            prompt_hash=prompts.prompt_hash,
        )
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    langsmith = LangSmithRecorder(langsmith_config, run_name=f"{combo.slug}__{run_ts}")
    try:
        provider = build_provider(combo.provider_config)
    except ProviderError as exc:
        failure = {"time": utc_now(), "run": combo.slug, "error": str(exc)}
        append_jsonl(failures_path, failure)
        raise

    completed = 0
    skipped = 0
    consecutive_errors = 0
    abort = False
    max_consecutive_errors = max(1, int(run_cfg.get("max_consecutive_errors") or 100))
    jobs: List[RecordJob] = []

    def write_result(result: RecordResult) -> None:
        nonlocal completed, consecutive_errors, abort
        job = result.job
        for failure in result.failures:
            append_jsonl(failures_path, failure)
        ranked_codes = list(result.parsed.get("ranked_codes") or ["000000"])
        # constrained mode: 000000 is not a valid permutation - use candidate order as error sentinel
        if ranked_codes == ["000000"] and mode == "constrained" and job.candidate_codes:
            ranked_codes = list(job.candidate_codes)
        predictions_by_id[job.frozen_id] = {"frozen_id": job.frozen_id, "ranked_codes": ranked_codes}
        completed += 1
        if result.error_text:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                abort = True
                print(
                    f"[{combo.slug}] circuit breaker: {consecutive_errors} consecutive errors - "
                    f"stopping. Resume with --rerun-errors after resolving credits/rate limits."
                )
        else:
            consecutive_errors = 0
        if completed % checkpoint_every == 0:
            write_submission(submission_path, meta, predictions_by_id, record_order)
        record_scores = per_record_scores(
            ranked_codes,
            str(job.record.get("hs6_label") or ""),
            bool(result.parsed.get("parse_valid")),
            combo.web_enabled and bool(result.provider_response and (result.provider_response.tool_calls or result.provider_response.citations)),
        )
        trace_outputs = build_trace_outputs(
            provider_response=result.provider_response,
            parsed=result.parsed,
            ranked_codes=ranked_codes,
            include_raw_provider_response=include_raw_provider_response,
        )
        langsmith.end_trace(
            job.trace_id,
            outputs=trace_outputs,
            error=result.error_text,
            metadata={
                **job.trace_meta,
                "parse_valid": result.parsed.get("parse_valid"),
                "parse_issues": result.parsed.get("parse_issues"),
            },
        )
        langsmith.add_external_row(
            sanitized=job.sanitized,
            actual_outputs=trace_outputs,
            evaluation_scores=record_scores,
            run_metadata={
                **job.trace_meta,
                "usage": result.provider_response.usage if result.provider_response else {},
                "latency_s": result.provider_response.elapsed_s if result.provider_response else None,
            },
            start_time=job.started_iso,
            end_time=result.ended_iso,
            error=result.error_text,
        )
        append_jsonl_gz(calls_path, {
            "time": result.ended_iso,
            "run": combo.slug,
            "frozen_id": job.frozen_id,
            "status": "error" if result.error_text else "ok",
            "error": result.error_text,
            "request": {
                "provider": combo.provider_name,
                "model": combo.model_id,
                "reasoning_level": combo.reasoning_level,
                "web_mode": combo.web_mode,
                "mode": mode,
                "tier": tier,
                "prompt_hash": prompts.prompt_hash,
                "prompt_version": prompts.prompt_version,
                "system": job.system,
                "user": job.user,
                "provider_payload": result.provider_response.request_payload if result.provider_response else None,
            },
            "response": {
                "text": result.provider_response.text if result.provider_response else "",
                "raw": result.provider_response.raw if result.provider_response else None,
                "usage": result.provider_response.usage if result.provider_response else {},
                "citations": result.provider_response.citations if result.provider_response else [],
                "tool_calls": result.provider_response.tool_calls if result.provider_response else [],
                "thought_summaries": result.provider_response.thought_summaries if result.provider_response else [],
                "warnings": result.provider_response.warnings if result.provider_response else [],
                "elapsed_s": result.provider_response.elapsed_s if result.provider_response else None,
            },
            "parsed": result.parsed,
            "gold": {
                "hs6": job.record.get("hs6_label"),
                "hs4": job.record.get("hs4_label"),
                "hs2": job.record.get("hs2_label"),
            },
            "scores": record_scores,
        })
        if (completed % int(run_cfg.get("progress_every") or 25)) == 0:
            print(f"[{combo.slug}] completed {completed} new records ({skipped} resumed)")

    for idx, record in enumerate(records):
        frozen_id = str(record["frozen_id"])
        system, user, candidate_codes, tier_text = build_prompt(
            record,
            mode=mode,
            tier=tier,
            taxonomy=taxonomy,
            open_codes=open_codes,
            top_k=top_k,
            prompts=prompts,
        )
        sanitized = sanitize_record_for_langsmith(
            record,
            tier_input=tier_text,
            mode=mode,
            candidate_codes=candidate_codes,
            taxonomy_hash=resolved_config["hashes"]["taxonomy_file_sha256"],
            taxonomy_count=len(open_codes),
            prompt_hash=prompts.prompt_hash,
            prompt_version=prompts.prompt_version,
            benchmark_version=BENCHMARK_VERSION,
        )
        trace_inputs = {
            "frozen_id": frozen_id,
            "system": system,
            "user": user,
            "mode": mode,
            "tier": tier,
            "candidate_codes": candidate_codes if mode == "constrained" else None,
        }
        trace_meta = {
            "provider": combo.provider_name,
            "model": combo.model_id,
            "reasoning_level": combo.reasoning_level,
            "web_mode": combo.web_mode,
            "record_index": idx,
            **dataset_metadata_for_record(
                record,
                split=split,
                mode=mode,
                tier=tier,
                prompt_hash=prompts.prompt_hash,
                prompt_version=prompts.prompt_version,
                dataset_hash=dataset_hash,
                taxonomy_hash=taxonomy_hash,
                manifest=manifest,
            ),
        }
        if frozen_id in predictions_by_id:
            skipped += 1
            ranked_codes = list(predictions_by_id[frozen_id].get("ranked_codes") or [])
            resumed_scores = per_record_scores(
                ranked_codes,
                str(record.get("hs6_label") or ""),
                True,
                False,
            )
            resumed_window = utc_window(seconds=1.0)
            resumed_outputs = build_trace_outputs(
                provider_response=None,
                parsed={
                    "parse_valid": True,
                    "parse_issues": ["resumed_from_submission_without_raw_response"],
                },
                ranked_codes=ranked_codes,
                include_raw_provider_response=include_raw_provider_response,
            )
            resumed_outputs["resumed"] = True
            langsmith.add_external_row(
                sanitized=sanitized,
                actual_outputs=resumed_outputs,
                evaluation_scores=resumed_scores,
                run_metadata={**trace_meta, "resumed": True},
                start_time=resumed_window["start"],
                end_time=resumed_window["end"],
            )
            continue
        jobs.append(RecordJob(
            idx=idx,
            record=record,
            frozen_id=frozen_id,
            system=system,
            user=user,
            candidate_codes=candidate_codes,
            tier_text=tier_text,
            sanitized=sanitized,
            trace_inputs=trace_inputs,
            trace_meta=trace_meta,
            started_iso="",
        ))

    def submit_job(executor: concurrent.futures.ThreadPoolExecutor, job: RecordJob) -> concurrent.futures.Future:
        job.started_iso = utc_now()
        job.trace_id = langsmith.start_trace(name=f"{combo.slug}:{job.frozen_id}", inputs=job.trace_inputs, metadata=job.trace_meta)
        return executor.submit(
            execute_record_job,
            job=job,
            provider=provider,
            combo=combo,
            mode=mode,
            tier=tier,
            open_codes=open_codes,
            top_k=top_k,
            max_retries=max_retries,
            retry_sleep_s=retry_sleep_s,
            temperature=None if temperature is None else float(temperature),
            max_output_tokens=max_output_tokens,
            timeout_s=None if timeout_s is None else float(timeout_s),
            run_cfg=run_cfg,
        )

    job_iter = iter(jobs)
    futures: Dict[concurrent.futures.Future, RecordJob] = {}
    poll_timeout = max(float(timeout_s or 300) * 1.5, 120.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        for _ in range(min(max_parallel_requests, len(jobs))):
            job = next(job_iter)
            futures[submit_job(executor, job)] = job
        try:
            while futures:
                done, _ = concurrent.futures.wait(futures, timeout=poll_timeout, return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:
                    print(
                        f"[{combo.slug}] WARNING: {len(futures)} thread(s) still running after "
                        f"{poll_timeout:.0f}s - threads may be stuck past request_timeout_s"
                    )
                    continue
                for future in done:
                    job = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = RecordResult(
                            job=job,
                            provider_response=None,
                            parsed={"ranked_codes": ["000000"], "parse_valid": False, "parse_issues": ["provider_error"]},
                            error_text=f"{type(exc).__name__}: {exc}",
                            failures=[{
                                "time": utc_now(),
                                "frozen_id": job.frozen_id,
                                "attempt": "worker",
                                "max_retries": max_retries,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            }],
                            ended_iso=utc_now(),
                        )
                    write_result(result)
                    if abort:
                        continue
                    try:
                        next_job = next(job_iter)
                    except StopIteration:
                        continue
                    futures[submit_job(executor, next_job)] = next_job
                if abort:
                    for f in list(futures.keys()):
                        f.cancel()
                    break
        except KeyboardInterrupt:
            print(f"\n[{combo.slug}] interrupted - cancelling pending jobs, saving checkpoint")
            for f in list(futures.keys()):
                f.cancel()
            raise
    submission = write_submission(submission_path, meta, predictions_by_id, record_order)
    try:
        report = score_submission(submission, records, report_md, report_json)
    except SubmissionError as exc:
        append_jsonl(failures_path, {"time": utc_now(), "run": combo.slug, "error": f"scoring_failed: {exc}"})
        raise
    summary_row = report_to_summary_row(combo, run_dir, report, prompts)
    write_summary_csv(run_dir / "summary.csv", [summary_row])
    langsmith_status = langsmith.upload_external_experiment(summary_scores_from_report(report))
    write_json(run_dir / "langsmith.json", langsmith_status)
    if langsmith.warnings:
        write_json(run_dir / "langsmith.warnings.json", langsmith.warnings)
    return summary_row


def run_benchmark(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data_cfg = dict(config.get("data") or {})
    split = str(data_cfg.get("split") or "eval")
    mode = str(data_cfg.get("mode") or "constrained")
    raw_tiers = data_cfg.get("tiers") or [data_cfg.get("tier") or 1]
    tiers = [int(t) for t in raw_tiers]
    paths = resolve_data_paths(config)
    records = read_json(paths["dataset"])
    if not isinstance(records, list):
        raise ValueError(f"{paths['dataset']} must contain a JSON list")
    limit = data_cfg.get("limit")
    if limit is not None:
        records = records[:int(limit)]
    taxonomy = load_taxonomy_csv(paths["taxonomy"])
    open_codes = in_scope_hs6_codes(taxonomy)
    manifest = read_json(paths["manifest"]) if paths["manifest"].exists() else {}
    run_cfg = dict(config.get("run") or {})
    out_root = Path(run_cfg.get("out_dir") or ROOT / "runs" / sanitize_slug(str(config.get("name") or f"{split}-{mode}")))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_top = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "name": config.get("name"),
        "created_at": utc_now(),
        "data": data_cfg,
        "run": run_cfg,
        "tiers": tiers,
        "model_count": len(config.get("models", []) or []),
        "record_count": len(records),
        "paths": {k: str(v) for k, v in paths.items()},
        "env": {
            "dotenv_loaded": list(config.get("_dotenv_loaded") or []),
        },
    }
    write_json(out_root / "benchmark_config.resolved.json", resolved_top)
    base_prompts = load_prompt_bundle(config, mode, web_mode="off")
    dataset_upload_status = upload_langsmith_reference_datasets(
        config=config,
        active_split=split,
        mode=mode,
        tier=tiers[0],
        prompts=base_prompts,
        taxonomy_count=len(open_codes),
        taxonomy_hash=sha256_file(paths["taxonomy"]) if paths["taxonomy"].exists() else None,
        manifest=manifest,
        paths=paths,
    )
    write_json(out_root / "langsmith_datasets.json", dataset_upload_status)
    all_rows: List[Dict[str, Any]] = []
    parallel_models = bool(run_cfg.get("parallel_models", False))

    def _run_combo_seq(tier_config: Mapping[str, Any], combos_for_model: List[Combo]) -> List[Dict[str, Any]]:
        rows = []
        for combo in combos_for_model:
            prompts = load_prompt_bundle(tier_config, mode, web_mode=combo.web_mode)
            print(f"==> Running {combo.slug}")
            rows.append(run_combo(
                config=tier_config,
                combo=combo,
                records=records,
                taxonomy=taxonomy,
                open_codes=open_codes,
                paths=paths,
                manifest=manifest,
                prompts=prompts,
                out_root=out_root,
            ))
        return rows

    for tier in tiers:
        tier_config = {**config, "data": {**data_cfg, "tier": tier}}
        combos = build_combos(tier_config, mode, tier)
        if not combos:
            raise ValueError("No model combinations to run")

        # Group combos by model_name so each model's combos always run sequentially
        groups: Dict[str, List[Combo]] = {}
        for combo in combos:
            groups.setdefault(combo.model_name, []).append(combo)

        if parallel_models and len(groups) > 1:
            print(f"==> Running {len(groups)} models in parallel (parallel_models=true)")
            for name in groups:
                print(f"  -> Model group: {name} ({len(groups[name])} combo(s))")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
                fut_map: Dict[concurrent.futures.Future, str] = {}
                for model_name, model_combos in groups.items():
                    fut = pool.submit(_run_combo_seq, tier_config, model_combos)
                    fut_map[fut] = model_name
                for fut in concurrent.futures.as_completed(fut_map):
                    all_rows.extend(fut.result())
        else:
            all_rows.extend(_run_combo_seq(tier_config, combos))
    write_summary_csv(out_root / "summary.csv", all_rows)
    return all_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Benchmark JSON config")
    parser.add_argument("--split", default=None, help="Dataset split (default: eval)")
    parser.add_argument("--mode", choices=["constrained", "open", "free"], default=None)
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], action="append", default=None,
                        help="Tier(s) to run. Repeat for a sweep: --tier 1 --tier 2 --tier 3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="Override run.out_dir")
    parser.add_argument("--max-parallel-requests", type=int, default=None)
    parser.add_argument("--max-parallel-combos", type=int, default=None,
                        help="Run this many combos concurrently (default: 1, sequential)")
    parser.add_argument("--parallel-models", action="store_true", default=None,
                        help="Run each model's combos in a separate thread (models share no rate limit)")
    parser.add_argument("--rerun-errors", action="store_true", default=False,
                        help="On resume, drop records whose last call errored or have a '000000' prediction and retry them")
    parser.add_argument("--tavily-fallback", action="store_true", default=False,
                        help="For Gemini models: pre-fetch web results via Tavily API instead of native Google search (requires TAVILY_API_KEY)")
    parser.add_argument("--model", action="append", help="Run only models with this config name/model id; repeatable")
    parser.add_argument(
        "--reasoning-level",
        action="append",
        default=None,
        help="Override reasoning levels. Repeat for a sweep. Use 'default' for no explicit setting.",
    )
    parser.add_argument(
        "--web-mode",
        action="append",
        default=None,
        choices=["off", "unrestricted", "restricted", "on"],
        help="Override web modes. Repeat for a sweep.",
    )
    parser.add_argument(
        "--langsmith-mode",
        choices=["off", "trace_only", "dataset_experiment", "both"],
        default=None,
    )
    parser.add_argument("--langsmith-project", default=None)
    parser.add_argument("--langsmith-dataset-name", default=None)
    args = parser.parse_args(argv)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = read_json(config_path)
    loaded_dotenv = load_dotenv_files(config, config_path)
    if loaded_dotenv:
        config["_dotenv_loaded"] = loaded_dotenv
    config = merge_cli_overrides(config, args)
    rows = run_benchmark(config)
    print()
    print("Completed benchmark runs:")
    for row in rows:
        print(
            f"- {row['run']}: n={row.get('n')} "
            f"hs6_top1={row.get('hs6_top1')} mrr={row.get('mrr')} -> {row.get('run_dir')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
