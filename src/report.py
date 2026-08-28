"""The report shape the baseline agent produces and the UI renders.

Defined once here so the agent (src/agent.py) and the UI (app.py) build
against the same contract instead of each inventing their own dict shape.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import PROCESSED_DATA_DIR

RUNS_DIR = PROCESSED_DATA_DIR / "runs"


@dataclass
class RunReport:
    run_id: str
    timestamp: str
    dataset_name: str
    target_column: str
    train_rows: int
    holdout_rows: int
    agent_summary: str
    generated_code: str
    stdout: str
    holdout_accuracy: float
    classification_report: dict[str, Any]
    duration_seconds: float
    positive_class: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def save_run(report: RunReport) -> Path:
    """Writes the run record as plain JSON, matching Research-agent's evaluation harness."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{report.run_id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def load_latest_run() -> RunReport | None:
    """Returns the most recently written run record, or None if no run exists yet."""
    if not RUNS_DIR.exists():
        return None
    run_files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not run_files:
        return None
    data = json.loads(run_files[-1].read_text())
    return RunReport(**data)


def list_runs() -> list[RunReport]:
    """Returns all saved run records, most recent first."""
    if not RUNS_DIR.exists():
        return []
    run_files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [RunReport(**json.loads(p.read_text())) for p in run_files]
