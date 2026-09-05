# Local model benchmark comparison
Snapshot for the Phase 3 local-model spike (agent_docs/TODOS.md). Not a decision - decisions go in decisions.md once the empirical bake-off runs.

## Hardware target
MacBook Pro, Apple M3 Pro, 11-core CPU, 14-core GPU, Metal 4, 18 GB unified memory. Safe model budget: 10-12 GB, 14 GB as the outer edge before the system risks swapping.

## Method
Coding and reasoning scores are published FP16/BF16 numbers from each model's own technical report or official model card, not measurements taken on this machine. Quantisation loss does not scale linearly with bit-width reduction - Q8 costs almost nothing, Q4_K_M costs a small, roughly single-digit percentage on most benchmarks, and loss grows sharply below Q4. The "Q4_K_M est." columns apply a flat 0.95x discount to the FP16 score as a conservative estimate, not a per-model measurement. gpt-oss-20b ships only in its native MXFP4 format, so no FP16 baseline exists to discount from - its published numbers already reflect close to that operating point. A cell reads "not measured" when no published number exists for that exact model on that exact benchmark, not because the model lacks the capability. The real answer for this project comes from the empirical bake-off: run each candidate against the same task the Gemini baseline already does (train a LightGBM model via `src/services/code_execution.py`, ADR-006's text-format handoff, scored against the real holdout).

## Coding benchmarks

| Model | Size at Q4_K_M | HumanEval (FP16) | Q4_K_M est. | MBPP(+) (FP16) | Q4_K_M est. | Source |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder 7B Instruct | ~4.7 GB | 88.4 | 84.0 | 83.5 | 79.3 | Qwen2.5-Coder Technical Report, Table 16 |
| Qwen2.5-Coder 14B Instruct | ~9 GB | 89.6 | 85.1 | 86.2 | 81.9 | Qwen2.5-Coder Technical Report, Table 16 |
| DeepSeek-Coder-V2-Lite Instruct | ~9.5 GB | 81.1 | 77.0 | 68.8 (MBPP+) | 65.4 | DeepSeek-Coder-V2 technical report |
| Qwen3 14B (non-thinking) | ~9 GB | not reported | - | LiveCodeBench v5: 29.0 | 27.6 | Qwen3 Technical Report, Table 16 |
| Llama 3.1 8B Instruct | ~4.9 GB | 72.6 | 69.0 | 72.8 (EvalPlus) | 69.2 | Meta official `eval_details.md` |
| gpt-oss-20b | ~13-14 GB (native) | not reported | n/a, native MXFP4 | SWE-Bench Verified: 60.7 | n/a, native MXFP4 | gpt-oss Model Card, Table 3 |

## Reasoning and tool-use benchmarks

| Model | Reasoning (FP16) | Q4_K_M est. | Tool-use / function calling | Source |
|---|---|---|---|---|
| Qwen2.5-Coder 7B Instruct | MMLU (5-shot): 68.7, GPQA: 35.6 | 65.3, 33.8 | not measured | Qwen2.5-Coder Technical Report, Table 20 |
| Qwen2.5-Coder 14B Instruct | MMLU (5-shot): 71.7, GPQA: 36.8 | 68.1, 35.0 | not measured | Qwen2.5-Coder Technical Report, Table 20 |
| DeepSeek-Coder-V2-Lite Instruct | MMLU: 60.1 | 57.1 | not measured | DeepSeek-Coder-V2 report |
| Qwen3 14B (non-thinking) | GPQA-Diamond: 54.8 | 52.1 | BFCL v3: 61.5 | Qwen3 Technical Report, Table 16 |
| Llama 3.1 8B Instruct | MMLU: 69.4 (5-shot) | 65.9 | not measured (only the 405B variant has a published BFCL score) | Meta official `eval_details.md` |
| gpt-oss-20b | MMLU: 85.3 | n/a, native MXFP4 | Tau-Bench Retail: 54.8, Tau-Bench Airline: 38.0 | gpt-oss Model Card, Table 3 |

## Reading

A benchmark differing between rows (LiveCodeBench versus HumanEval, MMLU versus GPQA, BFCL versus Tau-Bench) is not directly comparable across rows, only against its own FP16 baseline within the same row.

Qwen2.5-Coder-14B-Instruct leads on coding and fits comfortably. On MMLU, its 71.7 is also the highest of the three models with a reported score, ahead of Llama 3.1 8B Instruct's 69.4 and DeepSeek-Coder-V2-Lite-Instruct's 60.1, so it does not trade reasoning away for its coding lead. DeepSeek-Coder-V2-Lite-Instruct is close behind on coding and is the only one of the three shortlisted models with both a coding and a reasoning number from the same official source. Qwen3-14B is the only one with a published function-calling score (BFCL 61.5), which matters directly for `src/agent.py`'s tool-calling loop, but has no directly comparable coding number. gpt-oss-20b has the strongest reasoning and tool-use numbers by a wide margin, but sits at the edge of the 18 GB budget and has no HumanEval/MBPP number on the coding axis.

Shortlist for the empirical bake-off: Qwen2.5-Coder-14B-Instruct, DeepSeek-Coder-V2-Lite-Instruct, Qwen3-14B. gpt-oss-20b is a candidate stretch addition given its tool-use numbers, weighed against its tighter memory fit.

## Measured result: the real spike, 2026-09-05
Not a published benchmark - `scripts/spike_local_model.py` running the exact task `src/agent.py` performs (write and run a LightGBM training script via `src/services/code_execution.py`, scored against the real holdout), against `gemini-3.7-flash`, `ollama_chat/qwen2.5-coder:14b`, and `ollama_chat/qwen2.5-coder:7b`, all through the real `_build_instruction` system prompt, not a synthetic one.

A first minimal test (a bare prompt, no system instruction, asking the model to call an unrelated toy tool) showed Ollama never populating `tool_calls` for either Qwen2.5-Coder size - the model wrote the call as bare JSON instead of the `<tool_call>`-wrapped format its own chat template expects. That turned out to be an artefact of the minimal test, not a real limitation: re-run with the project's actual system instruction, both sizes called `run_training_code` correctly every time. Tool-calling itself is not the gap.

| Model | Trials | Tool called | Ran to completion | Holdout accuracy (successful runs) | Duration |
|---|---|---|---|---|---|
| gemini-3.7-flash | 1 | 1/1 | 1/1 | 0.9649 | 13.1s |
| qwen2.5-coder:14b | 4 | 4/4 | 2/4 | 0.9649, 0.9737 | 35-51s |
| qwen2.5-coder:7b | 4 | 4/4 | 0/4 | - | - |

Every failure on both Qwen2.5-Coder sizes shared one root cause: the generated script called `lgb.train(..., early_stopping_rounds=10)`, a direct keyword argument LightGBM removed in the 4.x line this project already depends on (`pyproject.toml`: `lightgbm>=4.7.0`) - early stopping now needs a `callbacks=[lgb.early_stopping(...)]` argument instead. `_build_instruction` (src/agent.py) never asks for early stopping at all; both Qwen sizes added it unprompted, using an older, once-common LightGBM convention. Gemini has not produced this failure in any observed run this session. One 7B trial also produced a different defect: 27 training features against the holdout's 30, a shape mismatch at scoring caused by the generated script dropping columns beyond what the instruction excludes.

This is exactly the kind of gap a HumanEval/MBPP score cannot surface: both benchmarks test generic code correctness, not compatibility with one specific installed library's current API surface. A model can score well on published coding benchmarks and still fail here.

Not yet tested at the time this was written: whether tightening `_build_instruction` to explicitly forbid `early_stopping_rounds` as a direct argument (or to drop early stopping entirely) closes this gap. That is a prompt-level fix available to any model, not specific to working around a weaker one. The MCP docs-grounding approach below was built and measured instead, per the user's decision to pursue the model-agnostic fix directly rather than the narrow prompt patch.

## Before/after: MCP docs-grounding server, 2026-09-05
`src/services/docs_mcp_server.py` - a local MCP server, no network call and no third-party account, that exposes `get_api_signature(library, symbol)`. It introspects the actual libraries installed in this project's `.venv` via Python's `inspect` and returns the real, current signature and docstring - guaranteed to match what the sandbox will actually run against. Wired into `src/agent.py` behind a `use_docs_tool` flag (default `False`, so today's Gemini path and the Streamlit UI are unaffected). When enabled, `_build_instruction` adds one line telling the agent to call the tool before using any function it isn't certain of - a general instruction, not a hardcoded ban on the one observed kwarg.

Methodology note: the first full run of this comparison shared one training-call budget (`MAX_CALLS_PER_SESSION`, `src/services/code_execution.py`) across all 20 trials, because they ran in one Python process. That module-level counter is meant to bound one run's session, not 20 independent trials - it produced two false failures (`TrainingBudgetExceeded`) in `qwen2.5-coder:7b`'s "after" condition that were a harness bug, not a model result. `scripts/spike_local_model.py` now runs each trial in its own subprocess, giving each a fresh budget; the `qwen2.5-coder:7b` "after" condition was re-run clean under the fixed harness. The other three conditions were unaffected (they exhausted no budget) and are reported as originally measured.

| Model | Condition | Success rate | Mean holdout accuracy | Mean duration |
|---|---|---|---|---|
| qwen2.5-coder:14b | before | 3/5 | 0.9766 | 33.0s |
| qwen2.5-coder:14b | after | 2/5 | 0.9693 | 45.3s |
| qwen2.5-coder:7b | before | 0/5 | - | - |
| qwen2.5-coder:7b | after (re-run, clean) | 0/5 | - | - |

**The MCP tool did not close the gap, and the reason is more informative than the raw numbers.** Checked directly against the run logs: `get_api_signature` was called zero times across all 5 of `qwen2.5-coder:14b`'s "after" trials - not on the 2 successes, not on the 3 failures. Both remaining failures in that condition wrote the identical stale `lgb.train(..., early_stopping_rounds=10)` call the "before" condition produced; the 2 successes happened to sample code that omitted early stopping entirely, the same way some "before" trials did. The tool sat available and unused in every trial. Giving a model a grounding tool is necessary but not sufficient - it also has to reliably choose to call it, and Qwen2.5-Coder at these sizes does not do that on its own initiative for a kwarg it is confident (wrongly) that it already knows.

This reframes the earlier open question. The gap is not "the model doesn't know the current API" - the tool proves current information is reachable - it is "the model doesn't know when to doubt its own memorised API knowledge enough to check." That is a harder problem than docs-grounding alone solves, and matches the pattern already named as the session's running theme: Qwen2.5-Coder's tool-calling and instruction-following is real but not fully reliable at this size, independent of what information is made available to it.

## The actual fix: stop asking the model to decide, 2026-09-05
Given the finding above, the fix moved from "make the tool available" to "remove the decision from the model entirely" - the same "enforce structurally, not politely" principle ADR-005 already used for the held-out split. Two changes to `src/agent.py`, both behind flags, both off by default:

- **`inject_signature`**: the harness itself calls `get_api_signature("lightgbm", "train")` (a plain Python call - `@mcp.tool()` doesn't stop the function being called directly) and pastes the real signature straight into the system instruction before the model writes anything. No tool call for the model to skip, because there's nothing left for it to decide.
- **`allow_retry`**: `_build_instruction`'s tool-call policy changes from "call it exactly once" to "if it fails, read stderr, fix the named defect, and try again" - capped at `MAX_TRAINING_ATTEMPTS = 3`, enforced structurally in `run_training_code` itself (it refuses a 4th call rather than trusting the model to stop), not just asked for in prose. Interpreting an explicit `TypeError` naming the bad argument is a far easier task than proactively doubting confident-but-wrong memorised knowledge.

Both together, 5 trials each, same subprocess-per-trial isolation as the corrected before/after run:

| Model | Condition | Success rate | Mean holdout accuracy | Mean duration |
|---|---|---|---|---|
| qwen2.5-coder:14b | fixed (injection + retry) | 5/5 | 0.9789 | 38.7s |
| qwen2.5-coder:7b | fixed (injection + retry) | 3/5 | 0.9708 | 33.8s |

For `qwen2.5-coder:14b` this closed the gap completely: 5/5 clean runs, every one on the first attempt (`attempts: 1`), zero `early_stopping_rounds` failures - up from 3/5 "before" and 2/5 "after" (optional tool). For `qwen2.5-coder:7b`: 0/5 before, 0/5 with the optional tool, 3/5 fixed - a real improvement, and the `early_stopping_rounds` failure class is gone from this condition entirely, but 2 of the remaining 3 failures are a different, unrelated defect (`RuntimeError: Agent finished without calling run_training_code` - the model not invoking the tool at all), and one success only landed on its second attempt. So the fix fully solved the API-staleness problem it targeted, on both sizes; `qwen2.5-coder:7b`'s remaining unreliability is a separate, more basic instruction-following gap the retry loop cannot help with, since the model has to call the tool at all before a retry is even possible.

## Sources
- [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186)
- [DeepSeek-Coder-V2 GitHub](https://github.com/deepseek-ai/DeepSeek-Coder-V2)
- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- [Llama 3.1 official eval details](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md)
- [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/pdf/2508.10925)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
