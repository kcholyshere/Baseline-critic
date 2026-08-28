# Architecture

Two views below: what is built right now (Phase 2, one dataset, one baseline), and how the final product (proposal phases 3-6, `references/project-proposal.md`) would extend it. Diagrams first, notes are short.

## Functionalities

The six things the finished product should do, from `references/project-proposal.md`. Time risk is marked on the ones that could realistically slip past a deadline.

- Profile a dataset: a deterministic summary of types, cardinality, missingness, target balance, and obvious keys, so the model reads facts instead of guessing at schema. Low time risk.
- Propose features in plain language, then implement them as code: the agent states an idea before it writes the script for it, rather than jumping straight to a training script the way today's single-shot agent does. Time risk: needs the profiling tool as an input and a real feature-proposal step, not just reused scaffolding.
- Train and score several baseline models under a compute budget: several attempts compared, not the one baseline built today, capped by the hard ceiling already in `code_execution.py`. Moderate time risk, mostly extends what already runs.
- Critique each accepted result for leakage and contamination, and reject with a named reason: the actual point of the project, a second agent given different information than the one that produced the result. Highest time risk of the six, since calibrating real detections against false alarms is genuinely hard, but it is the core differentiator - it should get first claim on remaining time, not be the one cut if time runs short.
- Report what was tried, what won, and what was rejected and why: a written account of the whole run. Low time risk, today's run report and UI already extend directly into this.
- Emit a runnable script or notebook that reproduces the winning result from scratch: a standalone reproducibility artefact, separate from the interactive run. Highest risk of being cut - real engineering effort for a feature the demo itself does not depend on.

## Current architecture

```mermaid
flowchart TD
    A["sklearn load_breast_cancer"] --> B["dataset.py: stratified split, seed 42"]
    B --> C["data/processed/train.csv - 455 rows"]
    B -.never written to disk.-> D["holdout - 114 rows, rebuilt in memory only"]

    C --> E["baseline_agent: gemini-3.7-flash via ADK"]
    E -->|"writes one Python script"| F["code_execution.py sandbox: fresh subprocess, scratch tempdir"]
    F -->|"model.txt, LightGBM text format, not pickle"| G["parent process loads lgb.Booster"]

    D --> H["score: predict on holdout, sklearn metrics"]
    G --> H
    H --> I["report.py: RunReport saved as JSON"]
    I --> J["Streamlit UI"]

    E -.traced via OpenTelemetry.-> K["Langfuse"]
```

```mermaid
sequenceDiagram
    participant U as Streamlit UI
    participant D as dataset.py
    participant A as baseline_agent
    participant S as sandbox
    participant P as parent process

    U->>D: build_train_artifact()
    D-->>U: train.csv written
    U->>A: run_baseline(train.csv path, target column)
    A->>A: writes one training script
    A->>S: run_training_code(script)
    S->>S: subprocess.run in fresh scratch dir
    S-->>A: stdout, stderr, model.txt
    A-->>U: one-sentence summary
    U->>D: get_holdout()
    U->>P: lgb.Booster(model_str=...), predict, score
    P->>U: RunReport saved and rendered
```

Notes:

- One agent, one shot. No planner/modeller split yet, no critic, no revision loop. It writes one script and stops.
- The holdout never touches disk and never enters the sandbox. The agent only ever sees `train.csv`.
- The model crosses the sandbox boundary as LightGBM's own text format, not pickle, so a malicious or buggy generated script cannot execute code in the trusted parent process by way of a poisoned pickle.
- No profiling tool. The agent is told the CSV path and target column directly, not a deterministic schema summary.
- No separate scoring tool component. Scoring is inline in `agent.py`, but it already respects the same rule a real scoring tool would: it is the only thing that ever sees the holdout.
- Run records are plain JSON files under `data/processed/runs/`, one per run, which is what the UI's run history reads.

## Final product would add

```mermaid
flowchart TD
    Start(["Dataset + target column"]) --> Profile["Profiling tool: deterministic schema summary"]
    Profile --> Plan["Planner/modeller agent"]
    Plan -->|"proposes features and training code"| CodeExec["Code execution tool"]
    CodeExec --> Score["Scoring tool: fixed holdout split"]
    Score --> Critique["Critic agent: different information, reviews for leakage"]
    Critique -->|"names a defect"| Plan
    Critique -->|"accepted"| Report["Report tool: renders the run as a document"]
    Report --> End(["Winning model + written account of what was tried"])

    HardCap["Hard cap: training runs per session"] -.bounds.-> CodeExec
    SoftCap["Soft cap: critique rounds per request"] -.bounds.-> Critique
```

Notes:

- The one agent splits into two: a planner/modeller that proposes and builds, and a critic that reviews. They get different information on purpose, so the critic cannot just agree with the modeller's framing.
- The loop can revise: a named defect sends the modeller back to try again, not just a pass or fail.
- A profiling tool replaces telling the agent the schema directly, so the model never guesses at types, cardinality, or missingness.
- The evaluation harness plants known leaks in a clean dataset and measures the critic's detection rate against its false-alarm rate on clean runs. That is what actually tests whether the critic is doing its job.
- Two budgets, not one: a hard ceiling on training runs (already built, `code_execution.py`, 20 calls per session) and a new soft ceiling on critique rounds per request.
- Optional, later: the critic runs as its own A2A service with dataset access over MCP, rather than in-process.
