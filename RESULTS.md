# SemiHS-Bench Evaluation Results

Full per-condition baselines for the four mode/tier combinations. The README
carries a compact web-off HS6@1 leaderboard; this file holds every row, both web
conditions, and all six metrics.

**Eval split:** 900 records (`data/eval.json`), 73 in-scope HS6 codes
(`data/taxonomy.csv`), schema_version 2.0.0.
**Metrics:** HS6@1, HS6@3, HS4@1, HS2@1 (% accuracy), MRR (%), HD (hierarchical
distance, lower is better).

---

## CONSTRAINED — Tier 1
*4-way MCQ, easier candidate set*

| Model | Web | HS6@1 | HS6@3 | HS4@1 | HS2@1 | MRR | HD |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | web-off | **85.1** | 98.0 | **99.3** | **99.4** | **91.4** | **0.161** |
| Gemini-3.1-Pro | web-unrestricted | 84.1 | **98.1** | 98.4 | 98.8 | 90.9 | 0.187 |
| Gemini-3-Flash | web-off | 81.6 | 96.2 | 98.0 | 99.2 | 89.2 | 0.212 |
| Gemini-3-Flash | web-unrestricted | 81.4 | 96.4 | 98.2 | 99.4 | 89.2 | 0.209 |
| GPT-5.5 | web-off | 80.2 | 96.3 | 97.6 | 99.3 | 88.3 | 0.229 |
| GPT-5.5 | web-unrestricted | 78.8 | 96.1 | 97.1 | 99.4 | 87.6 | 0.247 |
| Qwen3.7-Max | web-off | 75.9 | 94.7 | 96.3 | 98.9 | 85.4 | 0.289 |
| Qwen3.7-Max | web-unrestricted | 75.6 | 93.9 | 95.9 | 98.9 | 85.2 | 0.297 |
| Claude-Opus-4.7 | web-off | 75.3 | 96.1 | 97.2 | 99.3 | 85.8 | 0.281 |
| Claude-Opus-4.7 | web-unrestricted | 74.1 | 95.3 | 96.4 | 99.1 | 84.9 | 0.303 |
| GLM-5.1 | web-unrestricted | 73.7 | 94.1 | 92.9 | 98.3 | 84.1 | 0.351 |
| Claude-Sonnet-4.6 | web-unrestricted | 65.9 | 92.7 | 93.1 | 98.3 | 79.3 | 0.427 |
| Nemotron-Ultra | web-off | 50.6 | 88.7 | 87.3 | 97.0 | 69.7 | 0.651 |
| GLM-5.1 | web-off | 50.2 | 88.7 | 86.4 | 96.6 | 69.6 | 0.668 |
| Claude-Sonnet-4.6 | web-off | 48.3 | 88.1 | 85.6 | 96.7 | 68.4 | 0.694 |
| GPT-5.4-Mini | web-off | 48.7 | 86.8 | 85.3 | 94.9 | 68.6 | 0.711 |
| GPT-5.4-Mini | web-unrestricted | 47.6 | 86.2 | 83.1 | 94.0 | 67.5 | 0.753 |
| Kimi-K2.6 | web-unrestricted | 47.0 | 85.2 | 83.1 | 94.6 | 66.9 | 0.753 |
| MiniMax-M3 | web-off | 44.0 | 86.1 | 80.1 | 93.3 | 65.7 | 0.826 |
| MiniMax-M3 | web-unrestricted | 44.2 | 83.9 | 81.0 | 94.4 | 65.6 | 0.803 |
| Kimi-K2.6 | web-off | 40.1 | 81.6 | 81.2 | 93.8 | 62.3 | 0.849 |

> GLM-5.1, Claude-Sonnet-4.6, and Nemotron-Ultra had low parse rates (46–59%) in the web-unrestricted condition — output format issues without web grounding.

---

## CONSTRAINED — Tier 2
*4-way MCQ, harder candidate set*

| Model | Web | HS6@1 | HS6@3 | HS4@1 | HS2@1 | MRR | HD |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | web-unrestricted | **82.6** | **97.7** | **97.8** | **99.4** | **90.0** | **0.202** |
| Gemini-3.1-Pro | web-off | 76.3 | 97.0 | 93.4 | 99.3 | 86.5 | 0.309 |
| Gemini-3-Flash | web-unrestricted | 71.0 | 93.9 | 89.9 | 98.2 | 82.8 | 0.409 |
| GLM-5.1 | web-unrestricted | 70.6 | 92.3 | 90.4 | 98.1 | 81.7 | 0.409 |
| GPT-5.5 | web-unrestricted | 67.9 | 92.8 | 89.1 | 99.0 | 80.7 | 0.440 |
| GPT-5.5 | web-off | 66.8 | 91.9 | 89.0 | 99.0 | 79.8 | 0.452 |
| Gemini-3-Flash | web-off | 64.6 | 92.9 | 89.2 | 98.3 | 79.3 | 0.479 |
| Qwen3.7-Max | web-unrestricted | 64.0 | 91.8 | 88.4 | 98.4 | 78.1 | 0.491 |
| Claude-Opus-4.7 | web-off | 60.9 | 92.0 | 86.2 | 98.1 | 76.6 | 0.548 |
| Claude-Opus-4.7 | web-unrestricted | 60.9 | 90.4 | 87.6 | 97.9 | 76.3 | 0.537 |
| Nemotron-Ultra | web-unrestricted | 59.3 | 88.4 | 82.1 | 95.3 | 74.4 | 0.632 |
| Qwen3.7-Max | web-off | 57.7 | 89.2 | 82.9 | 98.7 | 74.0 | 0.608 |
| Claude-Sonnet-4.6 | web-unrestricted | 57.0 | 88.1 | 84.4 | 97.8 | 73.0 | 0.608 |
| GPT-5.4-Mini | web-unrestricted | 43.3 | 85.1 | 79.8 | 94.8 | 64.7 | 0.821 |
| Claude-Sonnet-4.6 | web-off | 41.3 | 84.6 | 78.1 | 95.6 | 63.5 | 0.850 |
| GPT-5.4-Mini | web-off | 40.6 | 83.4 | 76.9 | 94.7 | 63.2 | 0.879 |
| Nemotron-Ultra | web-off | 40.6 | 83.4 | 77.7 | 96.1 | 62.9 | 0.857 |
| Kimi-K2.6 | web-unrestricted | 37.8 | 82.3 | 72.2 | 93.4 | 60.7 | 0.966 |
| MiniMax-M3 | web-unrestricted | 36.3 | 80.2 | 73.9 | 92.8 | 59.9 | 0.970 |
| Kimi-K2.6 | web-off | 35.2 | 78.2 | 75.6 | 92.8 | 58.6 | 0.964 |
| GLM-5.1 | web-off | 33.9 | 80.0 | 71.6 | 93.2 | 58.3 | 1.013 |
| MiniMax-M3 | web-off | 33.1 | 78.9 | 72.1 | 93.0 | 57.7 | 1.018 |

---

## OPEN — Tier 1
*Free recall over 73 in-scope HS6 codes, easier*

| Model | Web | HS6@1 | HS6@3 | HS4@1 | HS2@1 | MRR | HD |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | web-off | **71.2** | 82.7 | **82.1** | **88.7** | **77.2** | **0.580** |
| Gemini-3.1-Pro | web-unrestricted | 69.2 | **84.3** | 80.6 | 86.3 | 76.8 | 0.639 |
| Gemini-3-Flash | web-unrestricted | 67.4 | 84.2 | 80.4 | 86.9 | 76.1 | 0.652 |
| Gemini-3-Flash | web-off | 61.9 | 81.0 | 76.6 | 84.6 | 71.5 | 0.770 |
| GPT-5.5 | web-off | 62.7 | 77.1 | 79.6 | 86.2 | 70.1 | 0.716 |
| GPT-5.5 | web-unrestricted | 62.2 | 76.1 | 78.3 | 85.7 | 69.4 | 0.738 |
| Qwen3.7-Max | web-unrestricted | 60.0 | 80.6 | 70.9 | 78.1 | 70.4 | 0.910 |
| Claude-Sonnet-4.6 | web-unrestricted | 55.6 | 69.9 | 74.8 | 85.4 | 62.7 | 0.842 |
| GPT-5.4-Mini | web-unrestricted | 52.1 | 64.6 | 68.9 | 80.4 | 59.1 | 0.986 |
| MiniMax-M3 | web-unrestricted | 50.6 | 69.8 | 65.3 | 74.8 | 60.3 | 1.093 |
| Qwen3.7-Max | web-off | 49.9 | 65.2 | 66.6 | 75.9 | 58.2 | 1.077 |
| Nemotron-Ultra | web-off | 48.4 | 67.1 | 68.6 | 77.6 | 57.9 | 1.054 |
| GLM-5.1 | web-unrestricted | 49.0 | 68.3 | 61.9 | 71.0 | 58.8 | 1.181 |
| GPT-5.4-Mini | web-off | 46.2 | 60.1 | 65.0 | 78.9 | 54.0 | 1.099 |
| Claude-Opus-4.7 | web-off | 46.4 | 61.7 | 69.3 | 81.9 | 54.5 | 1.023 |
| Claude-Opus-4.7 | web-unrestricted | 45.2 | 59.7 | 69.0 | 83.6 | 52.9 | 1.022 |
| MiniMax-M3 | web-off | 44.7 | 60.8 | 64.2 | 78.0 | 53.2 | 1.131 |
| Kimi-K2.6 | web-off | 43.9 | 62.7 | 66.0 | 75.3 | 53.1 | 1.148 |
| GLM-5.1 | web-off | 39.7 | 57.1 | 64.3 | 78.8 | 49.2 | 1.172 |
| Kimi-K2.6 | web-unrestricted | 36.7 | 54.8 | 44.8 | 47.7 | 46.0 | 1.709 |
| Claude-Sonnet-4.6 | web-off | 23.9 | 34.4 | 43.8 | 75.8 | 29.4 | 1.566 |
| Nemotron-Ultra | web-unrestricted | 4.8 | 4.9 | 5.1 | 5.3 | 4.8 | 2.848 |

> Nemotron-Ultra and Kimi-K2.6 web-unrestricted open-tier1 have near-zero parse rates (0.6% / 0%) — output format failure, not a reflection of true capability.

---

## OPEN — Tier 2
*Free recall over 73 in-scope HS6 codes, harder*

| Model | Web | HS6@1 | HS6@3 | HS4@1 | HS2@1 | MRR | HD |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | web-unrestricted | **66.3** | **78.2** | **77.9** | **84.6** | **72.7** | **0.712** |
| GPT-5.5 | web-unrestricted | 61.1 | 75.2 | 74.8 | 82.0 | 68.1 | 0.821 |
| Gemini-3-Flash | web-unrestricted | 58.7 | 77.7 | 73.1 | 79.2 | 68.2 | 0.890 |
| Qwen3.7-Max | web-unrestricted | 55.8 | 74.3 | 68.4 | 76.0 | 65.5 | 0.998 |
| Gemini-3.1-Pro | web-off | 51.8 | 63.8 | 63.0 | 72.6 | 58.0 | 1.127 |
| GPT-5.4-Mini | web-unrestricted | 49.5 | 59.8 | 66.3 | 78.2 | 55.2 | 1.060 |
| GLM-5.1 | web-unrestricted | 46.0 | 64.4 | 58.4 | 67.2 | 55.4 | 1.284 |
| Claude-Opus-4.7 | web-unrestricted | 44.7 | 60.3 | 67.3 | 79.7 | 52.9 | 1.083 |
| Gemini-3-Flash | web-off | 44.1 | 56.8 | 57.6 | 70.6 | 51.1 | 1.278 |
| MiniMax-M3 | web-unrestricted | 41.6 | 59.3 | 57.4 | 69.6 | 50.6 | 1.314 |
| GPT-5.5 | web-off | 40.6 | 55.9 | 54.3 | 70.2 | 48.5 | 1.349 |
| Claude-Sonnet-4.6 | web-unrestricted | 37.6 | 50.0 | 55.1 | 75.4 | 44.3 | 1.319 |
| Claude-Opus-4.7 | web-off | 33.1 | 43.2 | 49.9 | 67.7 | 38.7 | 1.493 |
| Kimi-K2.6 | web-unrestricted | 33.4 | 45.8 | 42.7 | 46.5 | 39.6 | 1.774 |
| Qwen3.7-Max | web-off | 31.0 | 43.3 | 48.6 | 66.6 | 37.7 | 1.539 |
| GPT-5.4-Mini | web-off | 31.3 | 41.9 | 45.9 | 65.9 | 37.4 | 1.569 |
| Nemotron-Ultra | web-off | 30.1 | 42.3 | 48.1 | 65.0 | 37.1 | 1.568 |
| Kimi-K2.6 | web-off | 29.4 | 41.6 | 45.4 | 66.1 | 36.0 | 1.590 |
| MiniMax-M3 | web-off | 28.4 | 40.0 | 42.9 | 63.1 | 34.8 | 1.656 |
| GLM-5.1 | web-off | 22.3 | 34.1 | 42.1 | 64.0 | 28.8 | 1.716 |
| Claude-Sonnet-4.6 | web-off | 19.0 | 26.2 | 34.0 | 64.4 | 23.2 | 1.826 |
| Nemotron-Ultra | web-unrestricted | 8.3 | 8.8 | 9.0 | 9.3 | 8.6 | 2.735 |

> GLM-5.1 and Kimi-K2.6 web-unrestricted open-tier2 are partial runs (n=860). Nemotron-Ultra web-unrestricted has a near-zero parse rate (0.7%) — output format failure.

---

See the [README](README.md) for the benchmark description, the two evaluation
modes, and the compact web-off leaderboard.
