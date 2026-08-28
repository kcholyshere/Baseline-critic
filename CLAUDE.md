# Project Context

## What
The Baseline Model Agent: point it at a tabular dataset and a target column, and a planner/modeller agent profiles the data, proposes and builds features, and trains a handful of baselines under a fixed budget. A separate critic agent reviews every accepted result before it is trusted - looking specifically for leakage, train/test contamination, and other reasons a good score might not be real. Full brief: references/project-proposal.md.

## Why
Anyone can wire an LLM to scikit-learn. The interesting design problem is the reviewer: an agent whose success is measured by rejecting work, given different information from the agent that produced it, bounded so it cannot argue forever. The business case is cost optimisation - analyst days saved per dataset, and the more expensive cost avoided: a leaked score reaching production before anyone notices.

## How
- Agent framework: Google ADK. LLM: `gemini-3.7-flash` via Vertex AI, ADC auth, no API keys (ADR-002).
- Data/modelling: pandas, scikit-learn + LightGBM.
- Code execution: agent-generated feature/training code runs in an isolated subprocess with a throwaway scratch directory, not a container - `src/services/code_execution.py` (ADR-001, ADR-003).
- Observability: Langfuse via OpenTelemetry (`openinference-instrumentation-google-adk`), wired in `src/services/observability.py` before any `Agent` is constructed, so every entrypoint imports it once (ADR-000).
- UI: Streamlit with custom CSS, not a separate TypeScript frontend - time-boxed decision (ADR-004).
- Dependency management: uv + `pyproject.toml`, no packaging - `uv run` for everything, no manual venv activation.
- Run records: plain JSON, same shape as Research-agent's evaluation harness.

### Phased approach
1. Scaffolding and scoping.
2. Initial build: one dataset (Breast Cancer Wisconsin, target `diagnosis`), one baseline, a report.
3. Critic: leakage and contamination detection.
4. Feature proposal loop: agent revises rather than running once.
5. Evaluation harness: injected leaks, detection/false-alarm rate, cost budget.
6. Improvements, including optionally the critic as its own A2A service with dataset access over MCP.

Live task list: agent_docs/TODOS.md. Full decision log with rationale: agent_docs/decisions.md.

## Critical rules
- Commit and push at reasonable intervals
- Final project due Monday 2026-09-07. Deliver as much as possible, but keep it explainable - don't outrun what can be walked through without a big comprehension debt.
