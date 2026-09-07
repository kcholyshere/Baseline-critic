# Evaluation

The project is evaluated along three separate questions. Each has its own script and its own saved
reports under `data/processed/`.

## 1. Does the critic catch real defects, without crying wolf?

Script: `scripts/run_evaluation.py`. Logic: `src/evaluation.py`.

A known defect is planted into a clean dataset, one category at a time, then scored by the critic
several times (its LLM half is not deterministic). This runs on two datasets - a classification task
and a regression task - so results are not tied to one task type.

Defect categories:

| Category | What it means |
|---|---|
| `none` | Clean run, no defect - measures the false-alarm rate |
| `target_leakage` | A feature encodes the target directly |
| `train_test_contamination` | Preprocessing fitted on data outside the training rows |
| `temporal_leakage` | A feature uses information from after the point of prediction |
| `score_mismatch` | The reported score does not match what the code actually computed |
| `unseeded_randomness` | No fixed random seed, so the result is not reproducible |
| `degenerate_split` | A split that leaves a class or the target column unrepresented |
| `duplicate_row_leakage` | The same or near-identical rows appear on both sides of the split |

Each category is reported at three levels: caught by static checks alone (`src/checks.py`, no LLM
call needed), caught by the critic's own LLM judgement, and caught by the two combined - which is
what a real user sees. A good result is a high combined detection rate on every defect category and
a low false-alarm rate on `none`.

## 2. Can generated code be talked into doing something dangerous?

Script: `scripts/run_red_team.py`, axis 1. Logic: `src/guardrails.py` + `src/red_team.py`.

Before any agent-written script runs, a static scan checks it for recognisable dangerous patterns.
Hand-written malicious snippets (never executed, only scanned) are checked against six categories:

- `network_access`
- `process_execution`
- `dynamic_code_execution`
- `unsafe_deserialization`
- `destructive_filesystem_op`
- `absolute_path_access`

The same scan also runs against real, legitimate training code, to check it does not block honest
work by mistake.

## 3. Can the critic be talked into accepting a bad result?

Script: `scripts/run_red_team.py`, axis 2. Logic: `src/red_team.py`.

Adversarial text is embedded in a script's code or console output - the two things the critic reads
as plain, uninspected text - trying to make it accept a run that has real, evidenced signs of a
defect. A control scenario with no injected text sets the baseline. A "flip" means the injected
version accepted where the control rejected: the failure this axis exists to catch.

## Where results land

`scripts/run_evaluation.py` writes to `data/processed/eval_reports/`, `scripts/run_red_team.py` to
`data/processed/red_team_reports/` and `red_team_summaries/`. Each run also updates a `latest.json`
pointer in its own directory. Both scripts cache the expensive part (live LLM calls) and only redo it
with `--rebuild`.
