# Codebase audit - 2026-09-06 (post feature-proposal loop)

Follow-up full-repo audit after today's critic reject-gate fix (ADR-016) and the feature-proposal loop
build (ADR-017) - not a repeat of the earlier same-day audit (`references/audit-2026-09-06.md`, all 16
findings fixed same day). Four Sonnet 5 subagents ran in parallel, each scoped to a non-overlapping slice,
instructed to read `agent_docs/decisions.md` and the prior audit first so deliberate, documented trade-offs
and already-fixed issues aren't re-flagged. Findings below are only what each agent verified directly - by
reading code, running a small reproduction, or driving the real Streamlit app via `AppTest` - not
speculation.

Scope split:
1. **Evaluation harness and critic gate** (explicit focus of this audit) - `src/evaluation.py`,
   `scripts/run_evaluation.py`, `src/critic.py`'s evidence-gate machinery, `src/checks.py`'s
   `evidence_categories`/`StaticFinding`
2. **Feature-proposal loop internals** - `src/feature_loop.py`, `src/agent.py`'s `RevisionContext`/
   `dataset_id`/stamping additions, `src/report.py`'s new fields and query functions
3. **UI, sandbox, and dataset preparation** - `src/ui/app.py`, `src/services/code_execution.py`,
   `src/services/docs_mcp_server.py`, `src/dataset.py`, `src/config.py`
4. **Docs and ADR/code consistency** - `agent_docs/decisions.md`, `agent_docs/TODOS.md`,
   `references/*`, `CLAUDE.md`, `pyproject.toml`, plus regression spot-checks on the prior audit's fixes

## Summary

| Severity | Count |
|---|---|
| High | 2 |
| Medium | 7 |
| Low | 4 |

---

## High severity

### 1. A `run_code()` exception (e.g. `TrainingBudgetExceeded`) escapes uncaught, leaving no run record and a misleading result - in both the feature loop and the evaluation harness
- **File**: `src/agent.py:356-427` (the `try:` block starts only *after* the `async for event in runner.run_async(...)` loop at 360-365, not around it), `src/agent.py:177-204` (`run_training_code` closure has no try/except around `run_code(code)`), `src/feature_loop.py:81-101` (`except Exception: continue`), `src/evaluation.py:495-511` (`run_evaluation_async`'s fixture loop has no try/except around the build step), `src/services/code_execution.py:45-60`
- **Category**: correctness / gap
- **Finding**: `run_code()` raises `TrainingBudgetExceeded` synchronously once `MAX_CALLS_PER_SESSION` (now 60) is exhausted. Nothing catches it between the sandbox and the caller: not the `run_training_code` tool closure, not ADK's own tool invoker, and not `_run_baseline_async` - whose failure-record `except` block was built specifically so a failed run always leaves a trace (ADR-008) sits *after* the code path this exception takes, so it's bypassed entirely. Two concrete consequences, found independently by two auditors: (a) in `run_feature_loop_async`, the round is silently dropped (`except Exception: continue`, no logging) - if it's the first round, `select_loop_winner([])` returns `None` and the UI reports "no accepted baseline found in N rounds", indistinguishable from a genuine modelling failure; (b) in `scripts/run_evaluation.py --rebuild`, the same exception aborts the whole evaluation run with a raw traceback and no summary JSON written at all (fixtures already built before the failure stay cached, so a re-run isn't a full loss, but the failure itself is undocumented and ungraceful).
- **Evidence**: traced by two independent auditors across all three layers (sandbox → ADK tool invoker → orchestration) with no exception handling found anywhere in the chain; confirmed via direct code reading, not a live repro (avoided the cost/time of actually exhausting the real budget against the local LLM).
- **Recommendation**: catch `TrainingBudgetExceeded` (and other `run_code` failures) at the point closest to the sandbox call - either in the `run_training_code` tool closure or by wrapping the `async for` loop in `_run_baseline_async` - and produce a distinguishable failure record/message, rather than letting it propagate as a bare `Exception` that both the loop and the harness treat identically to "the model failed to produce anything useful."

### 2. The evaluation harness's own "static check fired" reporting conflates confirmation-only findings with real defect evidence
- **File**: `scripts/run_evaluation.py:37` (`static_hit = "yes" if row["static_findings"] else "no"`), `src/evaluation.py:441` (`FixtureOutcome.static_findings` sourced from `run_static_checks`, not `evidence_categories`)
- **Category**: correctness / reporting
- **Finding**: `static_findings` includes confirmation-only checks (`_check_target_excluded_from_features`, the "gap is normal" branch of `_check_val_holdout_gap`) that carry no `category` and are never evidence of a defect - exactly the distinction ADR-016 built `evidence_categories()` to isolate. The CLI table's `static` column treats any non-empty `static_findings` list as "a static check caught this," which is wrong for precisely the two categories the harness's own documentation says have *no* static coverage.
- **Evidence**: ran the checks directly against the cached clean fixture and a reconstructed `temporal_leakage` fixture. Both produced non-empty `static_findings` (confirmation-only text) while `evidence_categories()` returned `set()` for both - so the table prints `static=yes` for the two cases the harness explicitly claims have zero static coverage.
- **Recommendation**: base the `static` column (and any "did static checks catch this" framing) on whether `evidence_categories()` is non-empty, not on `static_findings` being non-empty. Keep `static_findings` for the full display text; report coverage from the evidence set.

---

## Medium severity

### 3. `--trials 0` (or negative) crashes the evaluation harness with an unguarded `ZeroDivisionError`
- **File**: `src/evaluation.py:372-373` (`FixtureOutcome.combined_reject_rate`), `scripts/run_evaluation.py:20` (no bound on `--trials`)
- **Category**: correctness / gap
- **Finding**: `combined_reject_rate` divides by `len(self.trials)` with no guard; `_build_summary` accesses it unconditionally for every fixture.
- **Evidence**: reproduced directly - `FixtureOutcome(..., trials=[]).combined_reject_rate` raises `ZeroDivisionError: division by zero`. `--trials 0` reaches this exact path with no earlier validation.
- **Recommendation**: validate `trials > 0` in argparse, and have `combined_reject_rate` return `None` on an empty trial list, matching the graceful-degradation pattern `llm_only_reject_rate` already uses.

### 4. `_fallback_verdict`'s bundled evidence text can name one defect category while citing evidence for a different one
- **File**: `src/critic.py:253-260`
- **Category**: quality
- **Finding**: `sorted(evidence_categories)[0]` picks a category alphabetically (confirmed deterministic), but the returned `evidence` field is `"; ".join(static_findings)` - every finding, not just those tagged with the chosen category. On code that trips two categories at once (e.g. missing seed and missing stratify), a fallback reject would name one category while its evidence text describes both.
- **Recommendation**: filter `evidence` to findings whose `category` matches the chosen one - needs the underlying `StaticFinding` objects threaded through rather than the pre-flattened text list.

### 5. The evaluation harness's own fixture-report cache write bypasses the atomic-write fix
- **File**: `src/evaluation.py:343-359` (`load_or_build_fixture_report`), compare to the correct `_atomic_write_json` usage at `src/evaluation.py:482-483` in the same file
- **Category**: regression / gap
- **Finding**: the prior audit's finding #3 (corrupted/partially-written JSON crashes the app) was fixed via `report.py`'s `_atomic_write_json`, and `evaluation.py`'s own summary writer correctly uses it - but the fixture-cache write, 30 lines away in the same file, is a plain `cache_path.write_text(json.dumps(...))` with no try/except on the read side either. An interrupted `--rebuild` leaves a truncated cache file that crashes every subsequent evaluation run with an uncaught `JSONDecodeError`.
- **Evidence**: `evaluation.py:353` uses raw `write_text`; `evaluation.py:482-483` in the same module correctly calls `_atomic_write_json` - an inconsistency within the file that imports the atomic helper.
- **Recommendation**: route the fixture-cache write through `_atomic_write_json`, and make the read fall back to rebuilding on a parse failure rather than raising.

### 6. `run_feature_loop_async`'s per-round exception handling silently drops failed rounds from the UI's own round count
- **File**: `src/ui/app.py:234-237`, `src/feature_loop.py:66-108`
- **Category**: quality / correctness
- **Finding**: the sidebar's "no accepted baseline" warning uses `len(loop_result.attempts)`, but any round that raises (a `MAX_TRAINING_ATTEMPTS`-exhausted failure, or the budget exhaustion in finding #1) is caught and never appended to `attempts`. If round 1 fails and rounds 2-3 reject, the message says "in 2 rounds" though 3 ran; if every round fails, it says "in 0 rounds."
- **Evidence**: confirmed `attempts.append` only happens after the try block succeeds - a caught exception never contributes to the displayed count. The CLI `__main__` block avoids this by hardcoding `MAX_REVISION_ROUNDS` instead of counting attempts.
- **Recommendation**: track a separate `rounds_attempted` counter incremented once per loop iteration regardless of outcome, and report that instead of `len(attempts)`.

### 7. `prior_summaries` carries no accept/reject tag, so a rejected round's summary reads identically to an accepted one in a later prompt
- **File**: `src/agent.py:132-141`, `src/feature_loop.py:78, 105`
- **Category**: quality
- **Finding**: `agent_summary` is the modeller's own one-sentence description, written before critique runs, so it's verdict-neutral by construction - but `prior_summaries` is built unconditionally for every non-crashing round and shown as an undifferentiated list under "Prior attempts so far" when a later round revises after an accept.
- **Evidence**: confirmed against the real saved loop `c7e5e8e34e7a` - round 1 (rejected, 0.9825) and round 2 (accepted, 0.9649)'s summaries read in identical style with nothing distinguishing them.
- **Recommendation**: prefix each summary with its verdict when building the prompt text, e.g. `"[rejected: <category>] ..."` / `"[accepted, acc=X] ..."`.

### 8. "Persistent per-dataset run history" TODO checkbox overstates what the run list actually shows
- **Location**: `agent_docs/TODOS.md` (Phase 4 and "Next up"), `src/ui/app.py:88-89` (`_run_label`)
- **Category**: todo-accuracy
- **Finding**: the dataset filter and per-run critique detail genuinely work - selecting a specific run does show its full verdict/defect/evidence. But the filtered "Past runs" list itself, which is what someone would actually scan to answer "what did we try and why was it rejected," labels every plain single-shot run with only `f"{timestamp} · acc {accuracy:.3f}"` - no accept/reject status at all. Only loop entries show accept/reject in their label; the majority of run history (everything before today's loop feature) has no such signal in the list view.
- **Recommendation**: append accept/reject (and category, if rejected) to `_run_label`'s string, matching the pattern already used for loop entries.

### 9. Dataset filter dropdown shows indistinguishable duplicate labels when two datasets share a name
- **File**: `src/ui/app.py:434-437, 438-443`
- **Category**: quality / UX correctness
- **Finding**: `dataset_labels` keys by `dataset_id` (correct - filtering is never ambiguous under the hood), but the dropdown's visible text is `dataset_name`, so two distinct `dataset_id`s sharing a name render as two options with identical, unlabelled text.
- **Evidence**: injected a synthetic run record with `dataset_id="synthetic_ds_2"` and `dataset_name="breast_cancer_wisconsin"` (colliding with the real dataset's name) and confirmed via live `AppTest` rendering that the selectbox shows `['All datasets', 'breast_cancer_wisconsin', 'breast_cancer_wisconsin']` - selecting either resolves correctly, but a user can't tell them apart beforehand.
- **Recommendation**: disambiguate the label on a name collision, e.g. append a short `dataset_id` suffix.

---

## Low severity

### 10. `fallback_count` is computed but never shown in the evaluation harness's printed table
- **File**: `scripts/run_evaluation.py:33-43`, `src/evaluation.py:445`
- **Category**: quality / gap
- **Finding**: the table prints `gated` (total gated rounds) but omits `fallback_count` (trials that ultimately fell back to a static-only verdict) - a reader can't tell "the critic recovered after being gated" from "this is a static-only fallback" from the table alone.
- **Recommendation**: add a `fallback` column alongside `gated`.

### 11. Redundant double computation of static checks at every critic call site
- **File**: `src/agent.py:430-435, 458-463`, `src/evaluation.py:404-409`
- **Category**: quality
- **Finding**: every call site invokes `run_static_checks` and `evidence_categories` back-to-back with identical arguments, each independently re-running all 10 checks (including a CSV disk read for the correlation check). Not a correctness risk - both derive from the same `_run_all_checks()` and can't drift - just repeated I/O/CPU.
- **Recommendation**: expose one call returning `(text_list, evidence_set)` from a single `_run_all_checks()` pass.

### 12. Loop-count sidebar label doesn't pluralise ("1 rounds")
- **File**: `src/ui/app.py:463-467`
- **Category**: quality
- **Finding**: the real 1-round loop on disk (`9c80e685e9e3`) renders as "Loop · 1 rounds · winner acc 0.965" - the spinner text two lines away already handles this correctly (`round{'s' if max_rounds != 1 else ''}`), so this is an inconsistency within the same file, not a missed pattern.
- **Recommendation**: apply the same singular/plural pattern already used for the spinner text.

### 13. `_render_loop_result` has an unenforced (currently unreachable) empty-attempts precondition
- **File**: `src/ui/app.py:226-250`
- **Category**: quality / robustness
- **Finding**: `st.tabs([])` raises `StreamlitAPIException`; `_render_loop_result` has no guard against `loop_result.attempts == []`. The sole call site already checks `if loop_attempts:` first, so this isn't reachable today, but the function itself carries the precondition silently.
- **Recommendation**: not urgent given the existing guard; add a one-line note or guard if a future call site (e.g. a comparison view) is added.

---

## Verified clean (checked, no issue found)

- **Reject-gate correctness (ADR-016)**: traced every path through `critique_run_async`'s retry loop - no way for an unevidenced, non-`UNGATED_CATEGORIES` reject to become the final verdict; `_fallback_verdict` only ever selects from `evidence_categories`, so it can't bypass its own gate.
- **`evidence_categories()`/`run_static_checks()`**: share one `_run_all_checks()` pass - structurally cannot drift apart.
- **Fixture injection correctness**: re-measured correlations directly - legitimate ceiling 0.786 ("worst concave points"), `target_leakage` 0.9998, `temporal_leakage` 0.827 (ADR states 0.830, immaterial rounding) - the `0.97` threshold's margin is real on both sides. Every other fixture trips exactly its intended category with no cross-contamination from other checks.
- **`llm_only_reject_rate`**: returns `None` gracefully on an empty trial set; the CLI handles it correctly (`"n/a"`).
- **Fixture caching cost**: matches ADR-010 exactly - one `run_code()` call per fixture on `--rebuild`, zero on a cache hit.
- **`dataset_id` determinism**: called `dataset.build_train_artifact()` twice in the same process - byte-identical output, identical SHA-256 hash both times. Grouping by content hash is sound.
- **Stamping order**: both `RunReport` construction sites in `_run_baseline_async` (success and failure paths) set the four new fields before their respective `save_run()` call - no stale-on-disk risk.
- **Backward compatibility**: an old-shape dict missing the four new `RunReport` fields still constructs successfully with defaults.
- **`revised_from_run_id` chain**: verified against the real 3-round loop on disk - each round's `revised_from_run_id` equals the immediately preceding round's `run_id`, in strict order.
- **`select_loop_winner` tie-break and leak safety**: confirmed `max()` returns the first-encountered tied round; confirmed no rejected attempt, however high its raw accuracy, ever surfaces as a "winner" anywhere in the loop or the UI - both derive the displayed winner strictly through `select_loop_winner`.
- **`RevisionContext` field gating**: reject-only fields are populated only when `previous_verdict == "reject"`; accuracy/summaries are always populated; `Critique.to_dict()` always includes all keys, no `KeyError` risk on either branch.
- **Session-state precedence (`selected_run_id`/`selected_loop_id`)**: drove real interaction sequences through `AppTest` (select-loop-then-unrelated-rerun, dataset-filter-excludes-current-selection) - self-correcting by construction, no desync found.
- **Loop-grouping in the sidebar**: both real loops on disk (3-round and 1-round) collapse to exactly one entry each, correct round count and winner accuracy, no attempt duplicated or dropped.
- **Round-count slider staleness**: not a risk - Streamlit applies pending widget values atomically before the script that reads them runs.
- **Security spot-check on today's UI changes**: no new subprocess or file-path construction introduced; the new loop code only calls existing, already-audited `dataset.py`/`feature_loop.py` functions.
- **Regression re-checks, all still intact**: `code_execution.py`'s lock-guarded call-count check-and-increment (re-verified live with 20 concurrent threads, exact ceiling enforced, no overshoot); `docs_mcp_server.py`'s scope-leak fix (re-verified live: `get_api_signature("sklearn", "os.system")` correctly returns "not found"); `dataset.py`'s post-split class-presence assertion; `code_execution.py`'s 50MB artifact cap; `_run_baseline_async`'s failure-path record persistence (still correctly extended with today's stamping, not broken by it); `report.py`'s atomic writes for every path within that file; `_check_foreign_file_path`'s read-only-mode fix.
- **ADR-016/017 vs. code**: every specific claim checked (field names, defaults, thresholds, the `UNGATED_CATEGORIES` set, the verified loop numbers against the real stored JSON records) matches exactly.
- **`pyproject.toml` consistency**: `src/feature_loop.py`'s imports are all stdlib or already-declared dependencies.
- **`references/project-proposal.md`**: correctly left untouched by design (own header: "Draft, expected to change"; live status tracked in TODOS.md/decisions.md instead).
- **CLAUDE.md's own new resumify rule**: confirmed actually followed - `agent_docs/achievements.md` has entries for today's ADR-016/017 work, written in the same session the rule was added.

---

## Suggested priority

1. **#1 (uncaught budget-exhaustion exception)** - the highest-value fix: it makes both the feature loop and the evaluation harness produce a misleading result (rather than a clear error) under a condition that's increasingly plausible now that the loop and the harness share one process-wide sandbox budget.
2. **#2 (harness's own static-coverage reporting is wrong for the two categories it says have no coverage)** - directly undermines trust in the harness's headline numbers for exactly the categories where trust matters most; cheap to fix (swap one field for another already computed).
3. **#5 (fixture-cache non-atomic write)** - narrow blast radius (CLI-only), but a genuine regression against an already-fixed class of bug, and cheap to close.
4. **#3, #4, #6, #7, #8, #9** - real, verified issues, but each needs either an unusual input (`--trials 0`), a rare multi-category defect, a failed round, or a name collision to actually bite. Reasonable to batch together.
5. **#10-13** - low severity, cosmetic or currently-unreachable, defer freely.
