"""Optional LangSmith integration for SemiHS-Bench.

Local artifacts remain canonical. This module only mirrors provider-visible
traces or sanitized external experiments when explicitly configured.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _langsmith_jsonable(value: Any) -> Any:
    """Normalize upload payloads to LangSmith's accepted JSON shape."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                round(float(item), 4)
                if key == "score" and isinstance(item, float)
                else _langsmith_jsonable(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_langsmith_jsonable(item) for item in value]
    return value


def _uuid_for(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _time_window(seconds: float = 1.0) -> Dict[str, str]:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=seconds)
    return {"start": start.isoformat(), "end": end.isoformat()}


def sanitize_record_for_langsmith(
    record: Mapping[str, Any],
    *,
    tier_input: str,
    mode: str,
    candidate_codes: Sequence[str],
    taxonomy_hash: Optional[str],
    taxonomy_count: Optional[int],
    prompt_hash: str,
    prompt_version: Optional[str],
    benchmark_version: str,
) -> Dict[str, Any]:
    """Build the intentionally small benchmark row uploaded to LangSmith."""
    inputs: Dict[str, Any] = {
        "frozen_id": record.get("frozen_id"),
        "mode": mode,
        "tier_input": tier_input,
        "benchmark_version": benchmark_version,
        "prompt_hash": prompt_hash,
        "prompt_version": prompt_version,
    }
    if mode == "constrained":
        inputs["candidate_codes"] = list(candidate_codes)
    elif mode == "open":
        inputs["open_recall"] = True
    elif mode == "free":
        inputs["free_recall"] = True
        inputs["taxonomy_hash"] = taxonomy_hash
        inputs["taxonomy_count"] = taxonomy_count
    outputs = {
        "hs6_label": str(record.get("hs6_label") or ""),
        "hs4_label": str(record.get("hs4_label") or ""),
        "hs2_label": str(record.get("hs2_label") or ""),
    }
    metadata = {
        "benchmark": "SemiHS-Bench",
        "benchmark_version": benchmark_version,
        "schema_version": benchmark_version,
        "mode": mode,
        "prompt_hash": prompt_hash,
        "prompt_version": prompt_version,
        "frozen_id": record.get("frozen_id"),
        "difficulty_tags": list(record.get("difficulty_tags") or []),
        "difficulty_tags_joined": ",".join(record.get("difficulty_tags") or []),
        "has_difficulty_tags": bool(record.get("difficulty_tags") or []),
        "candidate_gold_rank": (record.get("candidate_set") or {}).get("gold_rank_in_candidates"),
        "tier2_classifiable": record.get("tier2_classifiable"),
        "hs2": record.get("hs2_label"),
        "hs4": record.get("hs4_label"),
        "hs6": record.get("hs6_label"),
        "tier1_source": record.get("tier1_source"),
    }
    return {"inputs": inputs, "expected_outputs": outputs, "metadata": metadata}


@dataclass
class LangSmithRecorder:
    config: Mapping[str, Any]
    run_name: str
    warnings: List[str] = field(default_factory=list)
    trace_client: Any = None
    trace_project: Optional[str] = None
    external_rows: List[Dict[str, Any]] = field(default_factory=list)
    start_time: str = field(default_factory=_utc_now)
    end_time: Optional[str] = None

    def __post_init__(self) -> None:
        self.mode = str(self.config.get("mode") or "off")
        self.enabled = self.mode in {"trace_only", "dataset_experiment", "both"}
        self.trace_enabled = self.mode in {"trace_only", "both"}
        self.experiment_enabled = self.mode in {"dataset_experiment", "both"}
        self.trace_project = str(self.config.get("project") or self.run_name)
        self.dataset_name = str(self.config.get("dataset_name") or f"semihs-bench-{self.run_name}")
        self.api_url = str(self.config.get("api_url") or os.environ.get("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
        self.api_key_env = str(self.config.get("api_key_env") or "LANGSMITH_API_KEY")
        if self.trace_enabled:
            self._init_trace_client()

    def _init_trace_client(self) -> None:
        try:
            from langsmith import Client  # type: ignore
        except ImportError:
            msg = "LangSmith SDK requested but 'langsmith' is not installed; continuing with local artifacts only"
            if self.config.get("required"):
                raise RuntimeError(msg)
            self.warnings.append(msg)
            self.trace_enabled = False
            return
        try:
            self.trace_client = Client(
                api_url=self.api_url,
                api_key=os.environ.get(self.api_key_env),
            )
        except Exception as exc:
            msg = f"LangSmith trace client initialization failed: {exc}"
            if self.config.get("required"):
                raise RuntimeError(msg) from exc
            self.warnings.append(msg)
            self.trace_enabled = False

    def start_trace(self, *, name: str, inputs: Mapping[str, Any], metadata: Mapping[str, Any]) -> Optional[str]:
        if not self.trace_enabled or self.trace_client is None:
            return None
        run_id = str(uuid.uuid4())
        try:
            self.trace_client.create_run(
                id=run_id,
                name=name,
                run_type="llm",
                inputs=dict(inputs),
                project_name=self.trace_project,
                extra={"metadata": dict(metadata)},
                start_time=datetime.now(timezone.utc),
            )
            return run_id
        except Exception as exc:
            self.warnings.append(f"LangSmith create_run failed for {name}: {exc}")
            return None

    def end_trace(
        self,
        run_id: Optional[str],
        *,
        outputs: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not run_id or not self.trace_enabled or self.trace_client is None:
            return
        try:
            kwargs: Dict[str, Any] = {"end_time": datetime.now(timezone.utc)}
            if outputs is not None:
                kwargs["outputs"] = dict(outputs)
            if error:
                kwargs["error"] = error
            if metadata:
                kwargs["extra"] = {"metadata": dict(metadata)}
            self.trace_client.update_run(run_id, **kwargs)
        except Exception as exc:
            self.warnings.append(f"LangSmith update_run failed for {run_id}: {exc}")

    @staticmethod
    def _object_id(value: Any) -> Any:
        if isinstance(value, Mapping):
            return value.get("id")
        return getattr(value, "id", value)

    def upload_reference_dataset(
        self,
        *,
        dataset_name: str,
        examples: Sequence[Mapping[str, Any]],
        description: str,
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Seed a full externally managed dataset independent of run limit.

        LangSmith's external experiment endpoint owns externally managed
        datasets. Creating normal SDK datasets first makes experiments appear
        under a different dataset, so reference seeding goes through the same
        endpoint as real benchmark experiments.
        """
        if not self.experiment_enabled:
            return {"dataset_name": dataset_name, "uploaded": False, "reason": "experiment mode disabled"}
        if self.config.get("upload_reference_datasets", True) is False:
            return {"dataset_name": dataset_name, "uploaded": False, "reason": "reference dataset upload disabled"}
        seed_window = _time_window(seconds=2.0)
        row_window = _time_window(seconds=1.0)
        results = []
        for example in examples:
            frozen_id = example["inputs"].get("frozen_id")
            results.append({
                "row_id": _uuid_for(f"{dataset_name}:{frozen_id}"),
                "inputs": dict(example["inputs"]),
                "expected_outputs": dict(example["expected_outputs"]),
                "actual_outputs": {"dataset_seed": True},
                "evaluation_scores": [{"key": "dataset_seed", "score": 1.0, "feedback_source": {"type": "model"}}],
                "start_time": row_window["start"],
                "end_time": row_window["end"],
                "run_name": "reference_dataset_seed",
                "run_metadata": {**dict(example.get("metadata") or {}), "dataset_seed": True},
            })
        body = {
            "experiment_name": str(self.config.get("seed_experiment_name") or f"__seed__{dataset_name}"),
            "experiment_description": "Seed full SemiHS-Bench reference dataset for later experiments.",
            "experiment_start_time": seed_window["start"],
            "experiment_end_time": seed_window["end"],
            "dataset_name": dataset_name,
            "dataset_description": description,
            "experiment_metadata": {**dict(metadata), "dataset_seed": True},
            "summary_experiment_scores": [
                {"key": "summary_seeded_examples", "score": float(len(examples)), "feedback_source": {"type": "model"}}
            ],
            "results": results,
        }
        status = self._post_upload_experiment(body)
        status.update({
            "dataset_name": dataset_name,
            "dataset_id": _uuid_for(dataset_name),
            "n_examples": len(examples),
            "seed_experiment_name": body["experiment_name"],
        })
        return status

    def add_external_row(
        self,
        *,
        sanitized: Mapping[str, Any],
        actual_outputs: Mapping[str, Any],
        evaluation_scores: Sequence[Mapping[str, Any]],
        run_metadata: Mapping[str, Any],
        start_time: str,
        end_time: str,
        error: Optional[str] = None,
    ) -> None:
        if not self.experiment_enabled:
            return
        self.external_rows.append({
            "row_id": _uuid_for(f"{self.dataset_name}:{sanitized['inputs']['frozen_id']}"),
            "inputs": dict(sanitized["inputs"]),
            "expected_outputs": dict(sanitized["expected_outputs"]),
            "actual_outputs": dict(actual_outputs),
            "evaluation_scores": [dict(s) for s in evaluation_scores],
            "run_metadata": {**dict(sanitized.get("metadata") or {}), **dict(run_metadata)},
            "start_time": start_time,
            "end_time": end_time,
            "run_name": self.run_name,
            **({"error": error} if error else {}),
        })

    def _post_upload_experiment(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            msg = f"{self.api_key_env} is not set; skipping LangSmith external experiment upload"
            if self.config.get("required"):
                raise RuntimeError(msg)
            self.warnings.append(msg)
            return {"mode": self.mode, "uploaded": False, "warnings": self.warnings}
        req = urllib.request.Request(
            f"{self.api_url}/api/v1/datasets/upload-experiment",
            data=_json_bytes(_langsmith_jsonable(body)),
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
            },
        )
        try:
            started = time.time()
            with urllib.request.urlopen(req, timeout=float(self.config.get("timeout_s") or 60)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return {
                "mode": self.mode,
                "uploaded": True,
                "elapsed_s": time.time() - started,
                "response": payload,
                "warnings": self.warnings,
            }
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8")
            except Exception:
                body_text = ""
            if exc.code == 409:
                msg = f"LangSmith experiment already exists (409 - skipped): {body_text}"
                self.warnings.append(msg)
                return {"mode": self.mode, "uploaded": False, "already_exists": True, "error": msg, "warnings": self.warnings}
            msg = f"LangSmith external experiment upload failed: HTTP {exc.code} {exc.reason}: {body_text}"
            if self.config.get("required"):
                raise RuntimeError(msg) from exc
            self.warnings.append(msg)
            return {"mode": self.mode, "uploaded": False, "error": msg, "warnings": self.warnings}
        except (urllib.error.URLError, TimeoutError) as exc:
            msg = f"LangSmith external experiment upload failed: {exc}"
            if self.config.get("required"):
                raise RuntimeError(msg) from exc
            self.warnings.append(msg)
            return {"mode": self.mode, "uploaded": False, "error": msg, "warnings": self.warnings}

    def upload_external_experiment(self, summary_scores: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        self.end_time = _utc_now()
        if not self.experiment_enabled:
            return {"mode": self.mode, "uploaded": False, "warnings": self.warnings}
        if not self.external_rows:
            return {"mode": self.mode, "uploaded": False, "warnings": self.warnings + ["No LangSmith external rows to upload"]}
        body = {
            "experiment_name": self.run_name,
            "experiment_description": "SemiHS-Bench externally managed benchmark run",
            "experiment_start_time": self.start_time,
            "experiment_end_time": self.end_time,
            "dataset_name": self.dataset_name,
            "dataset_description": "Sanitized SemiHS-Bench examples; local files are canonical.",
            "experiment_metadata": dict(self.config.get("metadata") or {}),
            "summary_experiment_scores": [dict(s) for s in summary_scores],
            "results": self.external_rows,
        }
        status = self._post_upload_experiment(body)
        status.update({
            "dataset_name": self.dataset_name,
            "dataset_id": _uuid_for(self.dataset_name),
            "experiment_name": self.run_name,
            "n_rows": len(self.external_rows),
        })
        return status
