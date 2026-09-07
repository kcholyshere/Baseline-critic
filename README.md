# Baseline critic
Point it at a tabular dataset and a target column. One agent profiles the
data, writes and trains a LightGBM baseline, and hands back a scored result -
a separate critic agent reviews every result before it is trusted, looking
specifically for leakage, contamination, and other reasons a good score
might not be real.

Everything runs on a local model via Ollama - no cloud LLM call, no API key.
Every run is traced to Langfuse.

## Why
Anyone can wire an LLM to scikit-learn. The interesting design problem is the
reviewer: an agent whose success is measured by rejecting work, given
different information from the agent that produced it, and bounded so it
cannot argue forever.

The business case is cost optimisation: analyst days saved per dataset, plus
the more expensive cost avoided - a leaked score reaching production before
anyone notices.

## Functionalities
- Upload a CSV and pick a target column - binary classification or
  regression, auto-detected and overridable.
- Profiles the dataset: types, cardinality, missingness, target balance,
  obvious identifier columns, and an optional time column for a
  chronological (rather than random) train/holdout split.
- Writes and trains a LightGBM baseline in an isolated subprocess sandbox,
  scored against a holdout it never sees during training.
- A critic agent reviews every result before it is accepted, checking for
  target leakage, train/test contamination, temporal leakage, duplicate-row
  leakage, unseeded randomness, and a mismatched score - most categories are
  backed by a deterministic check, not just the model's own judgement.
- A feature-proposal loop: the agent revises across several rounds instead
  of running once, pausing after each round so you can steer the next one
  with a free-text instruction.
- Persistent run history per dataset, browsable in the Historical runs tab,
  so a later run can see what was already tried and why it was rejected.
- A downloadable markdown report for any run or loop round - the same
  information the UI shows for it, in one file.
- An evaluation harness that plants one known defect per category into a
  reference dataset and measures the critic's detection rate against its
  false-alarm rate on a clean run.
- A red-team harness that checks the pipeline resists misuse: a static scan
  of agent-generated code, and prompt-injection scenarios against the critic.

## Limitations
- One model family only - LightGBM. No choice of algorithm yet.
- `qwen2.5-coder:14b` is not fully reliable even with the fixes already
  applied (signature injection, a bounded retry on its own runtime errors);
  the critic is the mitigation for that, not a fix.
- Temporal leakage has no deterministic static check by design - no
  correlation threshold safely separates it from a genuinely strong feature,
  so it stays critic-judgement-only.
- A repeat entity with differing values per visit (panel/longitudinal data)
  has no safe leakage check yet, and group-aware splitting isn't built.
- A high-cardinality categorical or mean encoding computed over the whole
  file before splitting is a real leakage route with no check yet.
- The evaluation harness reports detection/false-alarm rates combined across
  the classification and regression fixtures, not split by task type - each
  row still carries its own task type for a reader to split by hand.
- The sandbox enforces one training-call budget per app session; once it is
  exhausted, every further run in that session fails until the app restarts.
- Binary classification or regression only - no multiclass targets.

## Setup
```bash
uv sync
cp .env.example .env
ollama pull qwen2.5-coder:14b
```
`.env` only needs Langfuse keys, and only if you want tracing - leave them
unset and the app still runs, just without traces. There is no LLM API key
anywhere in this project; the model is local-only via Ollama.

## Running it locally
```bash
uv run python -m streamlit run src/ui/app.py
```
Use `python -m`, not the bare `streamlit` binary - the console-script shim's
shebang hard-codes the venv's absolute path at `uv sync` time, so it breaks
if the project directory is renamed or moved without recreating `.venv`.
`python -m` resolves through the interpreter instead.

Then open <http://localhost:8501>, upload a CSV, pick a target column, and
click **Run baseline**.

## Running it with Docker
```bash
docker compose up --build --wait
```
- <http://localhost:8501> - the Streamlit UI

The model stays on the **host** via Ollama, not in the container - pull it
there first (`ollama pull qwen2.5-coder:14b`). The container reaches it at
`host.docker.internal:11434`, which `docker-compose.yml` sets up for you.
Run history, uploads, and evaluation fixtures persist under `./data`, bind-
mounted into the container.

## Evaluation
```bash
uv run python -m scripts.run_evaluation          # critic detection/false-alarm rates
uv run python -m scripts.run_red_team            # misuse-resistance red team
```
The evaluation harness is also available from the UI's **Evaluation
harness** tab (**Run evaluation now**); the red-team harness is
command-line only. Both cache their results under `data/processed/` and can
be re-run with `--rebuild` to force fresh sandbox runs.

## Layout
```
src/
├── config.py                  <- paths, the local model URI
├── dataset.py                 <- train/holdout split, upload validation
├── profiling.py                 <- deterministic dataset summary for the prompt
├── agent.py                     <- the modeller agent, one baseline cycle
├── feature_loop.py              <- the multi-round revision loop
├── checks.py                    <- deterministic static checks for leakage/contamination
├── critic.py                    <- the critic agent - accepts or rejects with one named defect
├── report.py                    <- RunReport, the shared run-record contract, markdown rendering
├── evaluation.py                <- leak-injection harness: detection/false-alarm rate
├── guardrails.py                <- static misuse scan for agent-generated code
├── red_team.py                  <- misuse and prompt-injection red-team harness
├── services/code_execution.py   <- isolated subprocess sandbox for generated code
├── services/observability.py    <- Langfuse tracing, wired before any agent is built
├── services/docs_mcp_server.py  <- MCP server exposing library API signatures
└── ui/app.py                    <- the Streamlit dashboard

scripts/         <- CLI entry points for the evaluation and red-team harnesses
data/processed/  <- run records, uploads, evaluation and red-team fixtures (gitignored)
```

Project tracking lives in `agent_docs/`: `TODOS.md` for the live checklist
and `decisions.md` for the architectural decision log.
