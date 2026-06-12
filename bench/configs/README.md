# Benchmark Configs

Config files live here. Each one is a self-contained JSON description of a benchmark run:
which models to call, how many records to evaluate, which evaluation axes to sweep, and
where to send traces.

---

## File naming convention

| Pattern | Purpose |
|---|---|
| `smoke.mock.json` | Fast CI check - no API key, mock provider, 5 records |
| `frontier.json` | Quick multi-provider sanity check, 25 records |
| `pilot.<provider>.<mode>.json` | 25-record end-to-end validation before a full run |
| `<provider>.<mode>.json` | Full 900-record production run |

**pilot vs full run** - the only structural difference is `data.limit`. Pilots set `"limit": 25`
to validate the full pipeline (LangSmith tracing, output format, scoring, resume logic) cheaply
before committing to a full run. Once a pilot completes cleanly, duplicate the file, remove
the limit, and run it.

**mode in the filename** - each file covers one eval mode (`constrained` or `open`) because
the mode affects which prompt files are loaded and how outputs are parsed. Keep them separate
so you can run one mode at a time and control cost independently.

---

## Config sections

### `data`

Controls which records the harness loads and how they are presented to the model.

| Key | Type | Description |
|---|---|---|
| `split` | string | `"eval"` (the benchmark split) or `"train"` |
| `mode` | string | `"constrained"` - 4-way MCQ ranking; `"open"` - free HS6 recall |
| `tier` | int | Single tier (1 or 2) - use this for a fixed-tier run |
| `tiers` | int[] | Sweep multiple tiers as an outer loop: `[1, 2]` |
| `limit` | int \| null | Cap records per combo. `null` = all 900. Set to 25 for pilots |

Use either `tier` (single value) or `tiers` (array for Cartesian sweep) - not both.
Tier 1 is a full natural-language description. Tier 2 is MPN or short descriptor + manufacturer
only - the hardest input regime, with no supporting context.

---

### `run`

Execution parameters shared across all models in the run.

| Key | Default | Description |
|---|---|---|
| `out_dir` | required | Output directory, relative to repo root. e.g. `"runs/pilot-anthropic-constrained"` |
| `resume` | `true` | Skip records already written to `submission.json`. Safe to re-run interrupted jobs |
| `max_parallel_requests` | `5` | Worker pool size shared across all combos in this run. Tune per provider rate limits |
| `max_retries` | `2` | Retry count on transient API errors (429, 5xx) |
| `retry_sleep_s` | `2` | Seconds to wait between retries |
| `temperature` | `0.0` | Sampling temperature. Forced to `1.0` automatically for reasoning models |
| `max_output_tokens` | `10000` | Output token budget. Floored at 10000 - do not set lower |
| `request_timeout_s` | `180` | Per-request timeout in seconds. Increase for slow providers (Gemini Pro: 240) |
| `progress_every` | `10` | Print a progress line every N completed records |
| `web_max_uses` | `5` | Max web search calls per record when `web_mode = "unrestricted"` |
| `search_context_size` | `"medium"` | Search result verbosity: `"low"`, `"medium"`, or `"high"` |

**Parallelism guide by provider** (paid tier):

| Provider | Recommended `max_parallel_requests` | Notes |
|---|---|---|
| Anthropic | 20 | Tier 2 allows 1K RPM per model; Sonnet and Opus share the pool |
| OpenAI | 20 | Paid tier; o-series has high latency so workers stay busy |
| Gemini | 25 | Flash is 2K RPM, Pro is 1K RPM; shared pool across combos |
| OpenRouter | 5 | Varies by underlying model; conservative default |

---

### `models[]`

A list of model configs. The harness runs every model in the list, sweeping each one
across all `(tier × reasoning_level × web_mode)` combinations (Cartesian product).

**Common fields (all providers):**

| Key | Description |
|---|---|
| `name` | Short slug used in combo names and output paths |
| `provider` | `"anthropic"`, `"openai"`, `"gemini"`, or `"openai_compatible"` |
| `model` | Provider's model ID string (e.g. `"claude-sonnet-4-6"`) |
| `reasoning_levels` | Array of reasoning effort levels to sweep. `[null]` = no reasoning |
| `web_modes` | Array of web search modes: `["off"]`, `["unrestricted"]`, or `["off", "unrestricted"]` |

**Cartesian product example:**
```json
{
  "reasoning_levels": [null, "medium"],
  "web_modes": ["off", "unrestricted"]
}
```
This produces 4 combos per model per tier: `reasoning-default__web-off`,
`reasoning-default__web-unrestricted`, `reasoning-medium__web-off`,
`reasoning-medium__web-unrestricted`.

---

### Provider-specific model config

#### Anthropic

```json
{
  "name": "claude-sonnet-4-6",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "reasoning_levels": [null],
  "web_modes": ["off", "unrestricted"]
}
```

- API key env var: `ANTHROPIC_API_KEY`
- Reasoning levels: `null`, `"low"`, `"medium"`, `"high"` (maps to Anthropic's `output_config.effort`)
- Temperature is forced to `1.0` when reasoning is enabled
- **`claude-opus-4-7`** is reasoning-native and rejects the `temperature` parameter entirely - the harness omits it automatically

#### OpenAI

```json
{
  "name": "gpt-5.5",
  "provider": "openai",
  "model": "gpt-5.5",
  "reasoning_levels": [null],
  "web_modes": ["off", "unrestricted"]
}
```

- API key env var: `OPENAI_API_KEY`
- For o-series reasoning models, set `reasoning_levels: ["medium"]` or `["low", "medium", "high"]`
- Temperature is auto-forced to `1.0` for o-series

#### Gemini

```json
{
  "name": "gemini-3.1-pro-preview",
  "provider": "gemini",
  "model": "gemini-3.1-pro-preview",
  "reasoning_levels": [null],
  "web_modes": ["off", "unrestricted"],
  "api_key_env": "GOOGLE_API_KEY"
}
```

- API key env var: `GOOGLE_API_KEY` (also accepted as `GEMINI_API_KEY`)
- **Must set `api_key_env` per model entry** - the harness reads whichever env var you specify
- Use `request_timeout_s: 240` for Pro models (slower than Flash)
- Recommended `max_parallel_requests: 5` due to tighter RPM on Pro tier

#### OpenRouter (open-source models)

```json
{
  "name": "llama-3.1-405b",
  "provider": "openai_compatible",
  "model": "meta-llama/llama-3.1-405b-instruct",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_env": "OPENROUTER_API_KEY",
  "reasoning_levels": [null],
  "web_modes": ["off"]
}
```

- API key env var: `OPENROUTER_API_KEY`
- `provider` must be `"openai_compatible"` - not `"openai"`
- Each model entry needs its own `base_url` and `api_key_env`
- `web_modes: ["off"]` only - OpenRouter models have no native web tool in the harness
- Use longer `request_timeout_s` (240-300) for large models

---

### `prompts`

```json
{
  "dir": "bench/prompts",
  "version": "default"
}
```

| Key | Description |
|---|---|
| `dir` | Directory containing prompt template files, relative to repo root |
| `version` | Free-form label stored in LangSmith metadata per experiment row. Bump when prompts change so runs before and after the change are distinguishable in LangSmith |

Prompt files are selected by mode: `constrained_topk_{system,user}.txt` for constrained,
`open_topk_{system,user}.txt` for open without web, `open_web_topk_{system,user}.txt` for
open with web search.

---

### `langsmith`

```json
{
  "mode": "both",
  "project": "semihs-bench",
  "upload_reference_datasets": true,
  "include_raw_provider_response": false
}
```

| Key | Options | Description |
|---|---|---|
| `mode` | `"off"` | No LangSmith integration |
| | `"trace_only"` | SDK-level traces only, no experiment upload |
| | `"dataset_experiment"` | Upload results to LangSmith datasets/experiments, no SDK traces |
| | `"both"` | Traces + experiment upload |
| `project` | string | LangSmith project name |
| `upload_reference_datasets` | bool | Seed the reference dataset on first run (safe to repeat - idempotent) |
| `include_raw_provider_response` | bool | Whether to log the full raw API response. Keep `false` unless debugging |

**Dataset naming:** one dataset is created per `(mode, tier)` pair, e.g.
`semihs-bench-v2.0.0-eval-constrained-tier1`. All runs against the same (mode, tier) share
this dataset. Each individual combo run uploads as a separate named experiment under it.
To isolate a specific prompt version into its own dataset, add
`"dataset_name_template": "semihs-bench-v{benchmark_version}-{split}-{mode}-tier{tier}-{prompt8}"`
to this section.

---

## Adding a new config

The fastest path is to copy the closest existing pilot and edit it:

```bash
cp bench/configs/pilot.anthropic.constrained.json bench/configs/pilot.newprovider.constrained.json
```

Then update: `name`, `data.mode`, `run.out_dir`, `models[]`, and `langsmith.project`.
Run the smoke test first, then the pilot, then remove `data.limit` for the full run.
