"""The report shape the baseline agent produces and the UI renders.

Defined once here so the agent (src/agent.py) and the UI (app.py) build
against the same contract instead of each inventing their own dict shape.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROCESSED_DATA_DIR

RUNS_DIR = PROCESSED_DATA_DIR / "runs"
RESCORINGS_DIR = PROCESSED_DATA_DIR / "rescorings"


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
    model: str = ""
    attempts: int = 0
    critique: dict | None = None
    modeller_prompt_tokens: int = 0
    modeller_completion_tokens: int = 0

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


def save_rescoring(run_id: str, critique: dict, model: str) -> Path:
    """Writes a re-score result as its own record, keyed by the original run's
    id plus a fresh rescoring id - the original run's `critique` field is left
    untouched, so past critic verdicts stay a stable record you can compare a
    later critic version against, rather than being silently overwritten.
    """
    RESCORINGS_DIR.mkdir(parents=True, exist_ok=True)
    rescoring_id = uuid.uuid4().hex[:12]
    record = {
        "rescoring_id": rescoring_id,
        "run_id": run_id,
        "rescored_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "critique": critique,
    }
    path = RESCORINGS_DIR / f"{run_id}-{rescoring_id}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def list_rescorings(run_id: str | None = None) -> list[dict]:
    """Returns saved rescoring records, most recent first, optionally filtered to one run_id."""
    if not RESCORINGS_DIR.exists():
        return []
    files = sorted(RESCORINGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    records = [json.loads(p.read_text()) for p in files]
    if run_id is not None:
        records = [r for r in records if r["run_id"] == run_id]
    return records
