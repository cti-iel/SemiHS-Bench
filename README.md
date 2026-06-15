# SemiHS-Bench

**A Semiconductor Benchmark for HS6 Classification from Sparse ERP-Style Text**

Swapnil Bembde, Sai Rama Raju Penmatsa - Hitachi America Ltd
*(CIKM 2026 submission)*

Large language models are increasingly used for product classification in trade
compliance and supply-chain analytics. A persistent challenge is **input
sparsity**: enterprise systems often expose terse descriptions, abbreviated
bill-of-material entries, or bare part numbers, forcing a model to *recover
product identity before classifying it*. Existing HS datasets assume the input
is already informative and evaluate at a single level of detail, so they cannot
isolate this operational regime.

SemiHS-Bench targets exactly that regime for the **semiconductor supply chain**.
Each product is represented through **controlled input tiers derived from the
same observation**, with gold HS6 labels grounded in a held-out multilingual
corpus of customs rulings. We evaluate LLM baselines on **constrained** and
**open** classification and observe substantial degradation as input gets
sparser.

This repository contains the **benchmark dataset**, the **reference ruling
corpus**, the **evaluation tools**, and the **dataset-construction pipeline**.

---

## What's here (repository layout)

| Path | Purpose | License |
|------|---------|---------|
| `data/` | The benchmark: 1,800 records in matched train + eval splits (`train.json`, `eval.json`), reference ruling corpus (889 rulings), taxonomy (`taxonomy.csv`) with HS6 descriptions (`hs6_descriptions.csv`), JSON Schemas, and `MANIFEST.json` with SHA-256 hashes. | CC BY-SA 4.0 |
| `eval/` | Self-contained scoring CLI + library (`score_submission.py`, `lib.py`), worked `example_client.py`, the submission JSON Schema, and `build_manifest.py` (rebuilds `data/MANIFEST.json` from the data files). Python 3.9+ stdlib only. | MIT |
| `bench/` | Multi-provider evaluation runner (Anthropic / OpenAI / Gemini / OpenRouter) with resume, cost accounting, and optional LangSmith tracing — plus its run configs (`bench/configs/`, ready-to-run smoke / pilot / sweep files per provider × mode) and prompt templates (`bench/prompts/`, constrained / open / open-web top-k system + user prompts). | MIT |
| `examples/` | Canonical oracle submission + its scored report (expected output). | MIT |
| `construction/` | The dataset-build pipeline: source parsers, processing, annotation, assembly, audit tooling, and methodology docs. Published for provenance; the live scrapers and raw inputs are not redistributed. | MIT (code) |
| `DATASHEET.md`, `BALANCE_REPORT.md`, `data/STATISTICS.md` | Dataset documentation: intended use, sources, distributions, balance strategy. | CC BY-SA 4.0 |

---

## Quickstart (no API key needed)

```bash
# 1. Verify data integrity (compare to data/MANIFEST.json -> hashes)
shasum -a 256 data/eval.json

# 2. Run the worked example (mock LLM, no key needed)
uv run python3 eval/example_client.py --limit 10

# 3. Score the pre-canned oracle example
uv run python3 eval/score_submission.py \
    --submission examples/submission_constrained.example.json \
    --data data/eval.json
```

The oracle example scores top-1 = 1.0; the mock run scores ~25% (random
baseline). No external dependencies are required for scoring.

---

## The two evaluation modes

| Mode | Candidate space | Random top-1 | What it tests |
|------|-----------------|-------------:|---------------|
| **constrained** | 4-element per-record slate (gold + 3 confusable distractors) | 25.0% | disambiguation between confusable HS6 codes |
| **open** | 73 in-scope HS6 codes (`data/taxonomy.csv`) | ~1.4% | recall - can the model produce the right code unaided? |

Each mode runs against input **tiers** of decreasing information:

- **Tier 1** - full natural-language description (with manufacturer).
- **Tier 2** - MPN / short descriptor + manufacturer only (hardest regime;
  expert review found <1% of records fully classifiable from Tier-2 text alone).

Gold position is randomized per record (anti-position-bias). The taxonomy is
used server-side for scoring only and is never shown to the model in open mode.

Each of the 73 in-scope HS6 codes carries a `segment` label
(`data/taxonomy.csv`, also on every record) marking its OECD production stage:

| Segment | HS6 codes | Definition |
|---------|----------:|------------|
| `material` | 12 | Process inputs, substrates, gases, chemicals, photoresist |
| `equipment` | 8 | Fabrication machinery, furnaces, dedicated equipment parts |
| `metrology` | 15 | Measurement, inspection, test instrumentation |
| `component` | 31 | Semiconductor devices, electronic components, optics |
| `end_product` | 7 | Finished, semiconductor-intensive downstream goods |

---

## Baseline results

We evaluate eleven instruction-tuned LLMs — closed frontier models (Claude
Opus 4.7, Claude Sonnet 4.6, GPT-5.5, GPT-5.4-mini, Gemini 3.1 Pro, Gemini 3
Flash) and open-weight models served through OpenRouter (Qwen3.7-Max,
Nemotron-Ultra, GLM-5.1, MiniMax-M3, Kimi-K2.6) — on the full 900-record eval
split, across both modes and both input tiers, with provider-native web search
disabled and enabled.

The table below is the headline view: **HS6@1 (top-1 accuracy, %), web search
off**, across all four mode/tier conditions, sorted by constrained Tier 1.
Random top-1 = 25.0% constrained / ~1.4% open.

| Model | Constrained T1 | Constrained T2 | Open T1 | Open T2 |
|-------|---:|---:|---:|---:|
| Gemini-3.1-Pro | **85.1** | **76.3** | **71.2** | **51.8** |
| Gemini-3-Flash | 81.6 | 64.6 | 61.9 | 44.1 |
| GPT-5.5 | 80.2 | 66.8 | 62.7 | 40.6 |
| Qwen3.7-Max | 75.9 | 57.7 | 49.9 | 31.0 |
| Claude-Opus-4.7 | 75.3 | 60.9 | 46.4 | 33.1 |
| Nemotron-Ultra | 50.6 | 40.6 | 48.4 | 30.1 |
| GLM-5.1 | 50.2 | 33.9 | 39.7 | 22.3 |
| GPT-5.4-Mini | 48.7 | 40.6 | 46.2 | 31.3 |
| Claude-Sonnet-4.6 | 48.3 | 41.3 | 23.9 | 19.0 |
| MiniMax-M3 | 44.0 | 33.1 | 44.7 | 28.4 |
| Kimi-K2.6 | 40.1 | 35.2 | 43.9 | 29.4 |

Full per-condition results — both web conditions and all six metrics (HS6@1,
HS6@3, HS4@1, HS2@1, MRR, and hierarchical distance) — are in
[`RESULTS.md`](RESULTS.md).

---

## Running model evaluations

Install and configure providers:

```bash
uv venv
uv pip install -e ".[all]"        # all providers + langsmith
cp .env.example .env              # then add the keys you need
```

| Provider | Env var | Config files |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `bench/configs/pilot.anthropic.*.json` |
| OpenAI | `OPENAI_API_KEY` | `bench/configs/pilot.openai.*.json` |
| Google Gemini | `GOOGLE_API_KEY` | `bench/configs/pilot.gemini.*.json` |
| OpenRouter (OSS) | `OPENROUTER_API_KEY` | `bench/configs/pilot.openrouter.*.json` |
| LangSmith (optional) | `LANGSMITH_API_KEY` | any config with `langsmith.mode != "off"` |

```bash
# Smoke test (mock provider, no network)
uv run python3 -m bench.run_benchmark --config bench/configs/smoke.mock.json

# Pilot (25 records, full pipeline)
uv run python3 -m bench.run_benchmark --config bench/configs/pilot.anthropic.constrained.json
```

Runs are resumable. Each `(model × tier × reasoning × web_mode)` combo writes
its own `submission.json`, gzipped call log, and scored report. See
`bench/configs/README.md` for the full parameter reference.

---

## Dataset composition

| Dimension | Value |
|-----------|-------|
| Records | 1,800 - matched **900 train + 900 eval** (identical HS6 coverage) |
| HS4 families / HS6 codes | 38 / 73 |
| HS chapters | 28, 37, 38, 71, 73, 76, 84, 85, 90, 94 |
| Sources (per split) | BOL 318 / catalog 582 |
| Confidence (per split) | high 525 / medium 372 / low 3 |
| Boundary-tagged records | eval 718 / train 753 (~80 % sit on a confusable HS boundary) |
| Reference corpus | 889 rulings (CROSS 400, EBTI 389, JP 100), normalized to HS2022 |
| HS version | HS2022 |

All reported experiments use the **eval** split (`data/eval.json`); the **train**
split (`data/train.json`) is provided for reuse - fine-tuning, retrieval indices -
and mirrors the same HS6 coverage.

Each record carries a `difficulty_tags` list naming the confusable HS boundaries it
sits on, drawn from a closed 25-tag vocabulary in two groups — *within-family sibling
splits* (e.g. processor vs memory IC) and *cross-family frontiers* (e.g. discrete
device vs integrated circuit) — plus a `boundary_note` stating the deciding criterion
in plain language. The scorer reports a per-difficulty accuracy breakdown, so you can
read top-1 on boundary vs non-boundary records and per tag. See `data/STATISTICS.md`
for the full distribution.

**Caveats to report with results:**
- Gold HS6 labels are expert-validated against the HS2022 nomenclature; the reference
  ruling corpus is provided as supporting domain evidence (each record's
  `reference_ruling_ids` lists the corpus rulings that classify comparable goods under
  its gold HS6).
- When publishing results, cite the `eval.json` SHA-256 from `data/MANIFEST.json`
  along with the mode and tier(s) used.
- After editing any file under `data/`, rebuild the manifest: `python3 eval/build_manifest.py`.

See `DATASHEET.md` for sources, anonymization, and limitations.

---

## Anonymization

The brand/manufacturer name never appears in model-facing description text (it
lives only in the structured `manufacturer` field); catalog supplier names and
SKUs are excluded from description text. Authoritative-source records are
recent-only (retained start date 2011-01-01).

> Note: provenance fields (`source_reference`, some `justification_text`) may
> still contain supplier part numbers used for traceability. These are not shown
> to models. Scrub them if your distribution requirements are stricter than the
> model-input anonymization above.

---

## Rebuilding the dataset

The construction pipeline lives in `construction/` and is published for
transparency. Note that the **raw source dumps are not redistributed** (they
contain third-party catalog/BOL data), so the end-to-end pipeline requires
re-sourcing those inputs. See `construction/README.md` for the methodology,
including the adjudication and IAA protocols.

---

## License & citation

Dual-licensed: **code** under the **MIT License** (`LICENSE`); the **dataset**
under **CC BY-SA 4.0** (`LICENSE-DATA`), which also lists upstream source
licenses (EBTI, CROSS, BOL, catalog). If you use this work, cite the paper -
see `CITATION.cff`.
