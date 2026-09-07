# Architecture

One diagram, medium detail. For the reasoning behind each choice, see `agent_docs/decisions.md`.

```mermaid
flowchart TD
    User["User - Streamlit UI (src/ui/app.py)"]
    Dataset["dataset.py<br/>split: stratified or chronological<br/>holdout kept in memory only"]
    Profiling["profiling.py<br/>deterministic dataset summary"]
    Modeller["Modeller agent (agent.py)<br/>ADK + qwen2.5-coder:14b, local via Ollama/LiteLLM"]
    Guardrail["guardrails.py<br/>pre-execution misuse scan"]
    Sandbox["code_execution.py<br/>isolated subprocess, capped calls"]
    Score["Score on the holdout"]
    Checks["checks.py<br/>deterministic static leakage checks"]
    Critic["Critic agent (critic.py)<br/>different information than the modeller"]
    Loop["feature_loop.py<br/>revise on reject, up to N rounds<br/>optional free-text steering per round"]
    Report["report.py<br/>RunReport, saved as JSON"]
    Langfuse["Langfuse<br/>via OpenTelemetry"]

    User -->|"upload CSV or pick demo dataset, pick target"| Dataset
    Dataset --> Profiling
    Profiling --> Modeller
    Modeller -->|"writes a training script"| Guardrail
    Guardrail -->|"clean"| Sandbox
    Guardrail -.->|"flagged - blocked, no execution"| Modeller
    Sandbox --> Score
    Score --> Checks
    Checks --> Critic
    Modeller -.->|"code + stdout, for review"| Critic
    Critic -->|"accept"| Report
    Critic -->|"reject + named defect"| Loop
    Loop -->|"revision prompt"| Modeller
    Report --> User
    Loop -->|"winner or exhausted"| Report
    Modeller -. traced .-> Langfuse
    Critic -. traced .-> Langfuse
```

Notes:

- The critic sees the generated code, its output, and the static checks' findings, but not the modeller's own reasoning - it is deliberately kept independent, so it cannot just agree with the modeller's framing.
- The holdout is only ever touched by the scoring step, never by the modeller or the sandbox.
- The evaluation harness and red-team harness (`src/evaluation.py`, `src/red_team.py`, run via `scripts/`) reuse this same pipeline offline, against hand-written fixtures instead of live data. See `EVALUATION.md`.
