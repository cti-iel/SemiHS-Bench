#!/usr/bin/env python3
"""SemiHS-Bench - worked example: build a submission from an LLM client.

Demonstrates the end-to-end flow:

    1. Load the gold dataset
    2. For each record, build a constrained-top-k prompt using the templates
    3. Call an LLM (this script uses a deterministic *mock* client)
    4. Parse the model response into a ranked candidate list
    5. Write a valid submission JSON
    6. Print the metrics report

To plug in a real LLM, replace ``MockClient`` with your provider's SDK.
A clearly marked ``# TODO`` block shows where the swap goes.

Run:

    python3 eval/example_client.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from lib import (  # noqa: E402  (vendored)
    get_tier_input,
    load_taxonomy_csv,
    parse_ranked_indices,
    render_candidates_block,
    render_prompt,
)


# ----- LLM client interface --------------------------------------------------


@dataclass
class LLMResponse:
    text: str


class MockClient:
    """Deterministic mock client used in this example.

    Returns a ranking that puts the *first* candidate first. On a 4-way slate
    where gold position is randomly distributed, this yields ~25% top-1
    accuracy - the random baseline. Use this to verify the full pipeline
    without spending API credits.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def complete(self, system: str, user: str) -> LLMResponse:
        return LLMResponse(text="[0, 1, 2, 3]")


# ----- production swap point -------------------------------------------------


def build_real_client(provider: str, model_id: str) -> Any:
    """TODO: replace this stub with a real SDK call.

    Example with the Anthropic SDK::

        import anthropic
        class AnthropicClient:
            def __init__(self, model_id):
                self.model_id = model_id
                self.client = anthropic.Anthropic()
            def complete(self, system, user):
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=512,
                    temperature=0.0,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return LLMResponse(text="".join(b.text for b in msg.content if hasattr(b, "text")))
        return AnthropicClient(model_id)

    Example with the OpenAI SDK::

        from openai import OpenAI
        class OpenAIClient:
            def __init__(self, model_id):
                self.model_id = model_id
                self.client = OpenAI()
            def complete(self, system, user):
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    temperature=0.0,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                )
                return LLMResponse(text=resp.choices[0].message.content or "")
        return OpenAIClient(model_id)
    """
    raise NotImplementedError(
        "build_real_client is a stub. Edit eval/example_client.py and "
        "replace MockClient with your provider's SDK before running for real."
    )


# ----- prompt building -------------------------------------------------------


def build_constrained_prompts(
    record: Mapping[str, Any],
    *,
    tier: int,
    taxonomy: Optional[Mapping[str, Any]] = None,
) -> tuple:
    candidate_codes = list((record.get("candidate_set") or {}).get("codes") or [])
    if not candidate_codes:
        raise ValueError(f"record {record.get('frozen_id')!r} has no candidate_set")
    candidates_block = render_candidates_block(candidate_codes, taxonomy=taxonomy)
    tier_text = get_tier_input(record, tier)
    system = render_prompt("constrained_topk_system")
    user = render_prompt(
        "constrained_topk_user",
        tier_text=tier_text,
        candidates_block=candidates_block,
        num_candidates=len(candidate_codes),
    )
    return system, user, candidate_codes


# ----- main flow -------------------------------------------------------------


def run(
    *,
    data_path: Path,
    taxonomy_path: Path,
    tier: int,
    submission_name: str,
    submission_path: Path,
    client: Any,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    records = json.loads(data_path.read_text())
    if limit is not None:
        records = records[:limit]
    taxonomy = load_taxonomy_csv(taxonomy_path) if taxonomy_path.exists() else None

    predictions: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        system, user, candidate_codes = build_constrained_prompts(
            record, tier=tier, taxonomy=taxonomy
        )
        response = client.complete(system, user)
        ranked_indices = parse_ranked_indices(response.text or "", expected=len(candidate_codes))
        ranked_codes = [candidate_codes[i] for i in ranked_indices]
        predictions.append({
            "frozen_id": record["frozen_id"],
            "ranked_codes": ranked_codes,
        })
        if idx == 0:
            print(f"--- example record {record['frozen_id']} ---")
            print(f"  tier-{tier} input: {get_tier_input(record, tier)[:120]}")
            print(f"  candidates: {candidate_codes}")
            print(f"  model response: {(response.text or '')[:60]!r}")
            print(f"  parsed ranking: {ranked_codes}")
            print(f"  gold: {record['hs6_label']}")
            print()

    submission = {
        "submission": {
            "name": submission_name,
            "model_id": getattr(client, "model_id", "mock-client"),
            "mode": "constrained",
            "tier": tier,
            "schema_version": "2.0.0",
            "notes": "Generated by example_client.py - replace MockClient with a real LLM SDK.",
        },
        "predictions": predictions,
    }
    submission_path.write_text(json.dumps(submission, indent=2) + "\n")
    print(f"wrote {submission_path}")
    return submission


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "eval.json")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "data" / "taxonomy.csv")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument(
        "--submission-name",
        default="mock-client-constrained-tier1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "examples" / "_my_submission.json",
        help="Where to write the produced submission. Default avoids "
             "overwriting the canonical oracle example file.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N records (useful for smoke tests).",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        default=True,
        help="After writing the submission, run score_submission.py on it.",
    )
    parser.add_argument("--no-score", dest="score", action="store_false")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    client = MockClient()
    run(
        data_path=args.data,
        taxonomy_path=args.taxonomy,
        tier=args.tier,
        submission_name=args.submission_name,
        submission_path=args.out,
        client=client,
        limit=args.limit,
    )

    if args.score:
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "eval" / "score_submission.py"),
                "--submission", str(args.out),
                "--data", str(args.data),
            ],
            check=False,
        )
        return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
