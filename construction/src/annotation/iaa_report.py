"""Inter-annotator agreement metrics for SemiHS-Bench annotation fields.

Implements Cohen's κ, weighted Cohen's κ (linear), Krippendorff's α (ordinal),
and a macro-averaged per-tag κ for multilabel boundary tags. All metrics carry
non-parametric bootstrap CIs. Stdlib only (Python 3.9+).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

AMBIGUITY_ORDER: Tuple[str, ...] = ("1", "2", "3+")
DRIVER_VALUES: Tuple[str, ...] = ("material", "function", "use", "combination")
TIER2_ORDER: Tuple[str, ...] = ("no", "partial", "yes")
BOUNDARY_TAG_VOCAB: Tuple[str, ...] = (
    "8541_vs_8542",
    "8542.31_vs_8542.32_vs_8542.33_vs_8542.39",
    "8486_vs_8479",
    "3818_vs_2804",
    "8534_vs_8542",
    "9030_vs_9031",
)


def _paired_bootstrap_ci(
    values: Sequence[Tuple[object, object]],
    metric_fn: Callable[[Sequence[Tuple[object, object]]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 17,
) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    samples: List[float] = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        try:
            samples.append(metric_fn(resample))
        except (ZeroDivisionError, ValueError):
            continue
    if not samples:
        return (float("nan"), float("nan"))
    samples.sort()
    lo_idx = int((alpha / 2) * len(samples))
    hi_idx = int((1 - alpha / 2) * len(samples)) - 1
    hi_idx = max(lo_idx, min(hi_idx, len(samples) - 1))
    return (samples[lo_idx], samples[hi_idx])


def cohens_kappa(pairs: Sequence[Tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    categories = set(a_counts) | set(b_counts)
    p_e = sum((a_counts[c] / n) * (b_counts[c] / n) for c in categories)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def weighted_cohens_kappa(
    pairs: Sequence[Tuple[str, str]],
    order: Sequence[str],
    weight: str = "linear",
) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    idx = {value: i for i, value in enumerate(order)}
    k = len(order)
    if k <= 1:
        return float("nan")
    observed = [[0 for _ in range(k)] for _ in range(k)]
    for a, b in pairs:
        if a not in idx or b not in idx:
            continue
        observed[idx[a]][idx[b]] += 1
    total = sum(sum(row) for row in observed)
    if total == 0:
        return float("nan")
    row_marg = [sum(row) for row in observed]
    col_marg = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        d = abs(i - j) / (k - 1)
        return d if weight == "linear" else d * d

    num = sum(w(i, j) * observed[i][j] for i in range(k) for j in range(k))
    den = sum(
        w(i, j) * row_marg[i] * col_marg[j] / total for i in range(k) for j in range(k)
    )
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1.0 - (num / den)


def krippendorffs_alpha_ordinal(
    pairs: Sequence[Tuple[str, str]],
    order: Sequence[str],
) -> float:
    """Krippendorff's α for ordinal data with exactly two coders, no missing values.

    Uses the ordinal distance δ²(c, k) = (Σ_{g=c..k} n_g − (n_c + n_k)/2)²
    where n_g is the frequency of category g across both coders' annotations.
    """
    n = len(pairs)
    if n == 0:
        return float("nan")
    idx = {value: i for i, value in enumerate(order)}
    k = len(order)
    if k <= 1:
        return float("nan")
    cleaned: List[Tuple[int, int]] = []
    for a, b in pairs:
        if a in idx and b in idx:
            cleaned.append((idx[a], idx[b]))
    if not cleaned:
        return float("nan")

    freq = [0] * k
    for a, b in cleaned:
        freq[a] += 1
        freq[b] += 1
    total_units = sum(freq)

    def ordinal_dist_sq(c: int, g: int) -> float:
        if c == g:
            return 0.0
        lo, hi = (c, g) if c < g else (g, c)
        middle_sum = sum(freq[x] for x in range(lo, hi + 1))
        adj = middle_sum - (freq[c] + freq[g]) / 2.0
        return adj * adj

    d_o = sum(ordinal_dist_sq(a, b) for a, b in cleaned) / len(cleaned)
    d_e_num = 0.0
    for c in range(k):
        for g in range(k):
            d_e_num += freq[c] * freq[g] * ordinal_dist_sq(c, g)
    d_e = d_e_num / (total_units * (total_units - 1)) if total_units > 1 else 0.0
    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0
    return 1.0 - (d_o / d_e)


def _parse_tagset(raw: str) -> frozenset:
    if not raw:
        return frozenset()
    return frozenset(item.strip() for item in raw.split(";") if item.strip())


def multilabel_jaccard(pairs: Sequence[Tuple[frozenset, frozenset]]) -> float:
    if not pairs:
        return float("nan")
    values: List[float] = []
    for a, b in pairs:
        if not a and not b:
            values.append(1.0)
            continue
        union = a | b
        if not union:
            values.append(1.0)
            continue
        values.append(len(a & b) / len(union))
    return sum(values) / len(values)


def multilabel_macro_kappa(
    pairs: Sequence[Tuple[frozenset, frozenset]],
    vocabulary: Sequence[str],
) -> float:
    if not pairs:
        return float("nan")
    kappas: List[float] = []
    for tag in vocabulary:
        tag_pairs = [
            ("1" if tag in a else "0", "1" if tag in b else "0") for a, b in pairs
        ]
        kappas.append(cohens_kappa(tag_pairs))
    finite = [value for value in kappas if value == value]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def confusion_matrix(
    pairs: Sequence[Tuple[str, str]],
    order: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in pairs:
        matrix[a][b] += 1
    if order is None:
        return {row: dict(values) for row, values in matrix.items()}
    ordered: Dict[str, Dict[str, int]] = {}
    for row in order:
        ordered[row] = {col: matrix[row].get(col, 0) for col in order}
    return ordered


def _normalize(value: object) -> str:
    text = str(value or "").strip()
    return text


def _load_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            rid = _normalize(row.get("id"))
            if not rid:
                continue
            rows[rid] = {k: _normalize(v) for k, v in row.items()}
        return rows


def _paired(
    rows_a: Mapping[str, Mapping[str, str]],
    rows_b: Mapping[str, Mapping[str, str]],
    field: str,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for rid, row_a in rows_a.items():
        row_b = rows_b.get(rid)
        if not row_b:
            continue
        a_val = _normalize(row_a.get(field))
        b_val = _normalize(row_b.get(field))
        if not a_val or not b_val:
            continue
        pairs.append((a_val, b_val))
    return pairs


def _paired_tags(
    rows_a: Mapping[str, Mapping[str, str]],
    rows_b: Mapping[str, Mapping[str, str]],
    field: str,
) -> List[Tuple[frozenset, frozenset]]:
    pairs: List[Tuple[frozenset, frozenset]] = []
    for rid, row_a in rows_a.items():
        row_b = rows_b.get(rid)
        if not row_b:
            continue
        pairs.append((_parse_tagset(row_a.get(field, "")), _parse_tagset(row_b.get(field, ""))))
    return pairs


def score(
    rows_a: Mapping[str, Mapping[str, str]],
    rows_b: Mapping[str, Mapping[str, str]],
    n_boot: int = 1000,
    seed: int = 17,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "paired_record_count": sum(1 for rid in rows_a if rid in rows_b),
        "rater_a_record_count": len(rows_a),
        "rater_b_record_count": len(rows_b),
        "bootstrap_iterations": n_boot,
        "fields": {},
    }

    # classification_driver — unweighted κ
    driver_pairs = _paired(rows_a, rows_b, "classification_driver")
    driver_kappa = cohens_kappa(driver_pairs)
    driver_ci = _paired_bootstrap_ci(driver_pairs, cohens_kappa, n_boot=n_boot, seed=seed)
    report["fields"]["classification_driver"] = {
        "metric": "cohens_kappa",
        "value": driver_kappa,
        "ci_lo": driver_ci[0],
        "ci_hi": driver_ci[1],
        "n_pairs": len(driver_pairs),
        "confusion_matrix": confusion_matrix(driver_pairs, DRIVER_VALUES),
    }

    # tier2_classifiable — linear-weighted κ
    tier2_pairs = _paired(rows_a, rows_b, "tier2_classifiable")
    tier2_kappa = weighted_cohens_kappa(tier2_pairs, TIER2_ORDER, weight="linear")
    tier2_ci = _paired_bootstrap_ci(
        tier2_pairs,
        lambda resample: weighted_cohens_kappa(resample, TIER2_ORDER, weight="linear"),
        n_boot=n_boot,
        seed=seed + 1,
    )
    report["fields"]["tier2_classifiable"] = {
        "metric": "weighted_cohens_kappa_linear",
        "value": tier2_kappa,
        "ci_lo": tier2_ci[0],
        "ci_hi": tier2_ci[1],
        "n_pairs": len(tier2_pairs),
        "confusion_matrix": confusion_matrix(tier2_pairs, TIER2_ORDER),
    }

    # ambiguity_score — Krippendorff α (ordinal)
    amb_pairs = _paired(rows_a, rows_b, "ambiguity_score")
    amb_alpha = krippendorffs_alpha_ordinal(amb_pairs, AMBIGUITY_ORDER)
    amb_ci = _paired_bootstrap_ci(
        amb_pairs,
        lambda resample: krippendorffs_alpha_ordinal(resample, AMBIGUITY_ORDER),
        n_boot=n_boot,
        seed=seed + 2,
    )
    report["fields"]["ambiguity_score"] = {
        "metric": "krippendorffs_alpha_ordinal",
        "value": amb_alpha,
        "ci_lo": amb_ci[0],
        "ci_hi": amb_ci[1],
        "n_pairs": len(amb_pairs),
        "confusion_matrix": confusion_matrix(amb_pairs, AMBIGUITY_ORDER),
    }

    # boundary_tags — mean Jaccard + macro per-tag κ
    tag_pairs = _paired_tags(rows_a, rows_b, "boundary_tags")
    jaccard = multilabel_jaccard(tag_pairs)
    jaccard_ci = _paired_bootstrap_ci(
        tag_pairs,
        multilabel_jaccard,
        n_boot=n_boot,
        seed=seed + 3,
    )
    macro_kappa = multilabel_macro_kappa(tag_pairs, BOUNDARY_TAG_VOCAB)
    macro_ci = _paired_bootstrap_ci(
        tag_pairs,
        lambda resample: multilabel_macro_kappa(resample, BOUNDARY_TAG_VOCAB),
        n_boot=n_boot,
        seed=seed + 4,
    )
    per_tag: Dict[str, Dict[str, object]] = {}
    for tag in BOUNDARY_TAG_VOCAB:
        binary_pairs = [
            ("1" if tag in a else "0", "1" if tag in b else "0") for a, b in tag_pairs
        ]
        per_tag[tag] = {
            "n_pairs": len(binary_pairs),
            "kappa": cohens_kappa(binary_pairs),
            "support_a": sum(1 for a, _ in tag_pairs if tag in a),
            "support_b": sum(1 for _, b in tag_pairs if tag in b),
        }
    report["fields"]["boundary_tags"] = {
        "metric": "mean_jaccard_and_macro_cohens_kappa",
        "mean_jaccard": jaccard,
        "mean_jaccard_ci_lo": jaccard_ci[0],
        "mean_jaccard_ci_hi": jaccard_ci[1],
        "macro_kappa": macro_kappa,
        "macro_kappa_ci_lo": macro_ci[0],
        "macro_kappa_ci_hi": macro_ci[1],
        "n_pairs": len(tag_pairs),
        "per_tag": per_tag,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater-a", default="data/intermediate/review/iaa_annotated_rater_a.csv")
    parser.add_argument("--rater-b", default="data/intermediate/review/iaa_annotated_rater_b.csv")
    parser.add_argument("--out", default="data/intermediate/review/iaa_report.json")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows_a = _load_rows(Path(args.rater_a))
    rows_b = _load_rows(Path(args.rater_b))
    report = score(rows_a, rows_b, n_boot=args.n_boot, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("paired records:", report["paired_record_count"])
    for field, payload in report["fields"].items():
        metric = payload.get("metric")
        if metric == "mean_jaccard_and_macro_cohens_kappa":
            print(
                "{0}: jaccard={1:.3f} [{2:.3f},{3:.3f}]  macro_kappa={4:.3f} [{5:.3f},{6:.3f}]".format(
                    field,
                    payload["mean_jaccard"],
                    payload["mean_jaccard_ci_lo"],
                    payload["mean_jaccard_ci_hi"],
                    payload["macro_kappa"],
                    payload["macro_kappa_ci_lo"],
                    payload["macro_kappa_ci_hi"],
                )
            )
        else:
            print(
                "{0} ({1}): {2:.3f} [{3:.3f},{4:.3f}] n={5}".format(
                    field,
                    metric,
                    payload["value"],
                    payload["ci_lo"],
                    payload["ci_hi"],
                    payload["n_pairs"],
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
