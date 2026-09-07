"""The report shape the baseline agent produces and the UI renders.

Defined once here so the agent (src/agent.py) and the UI (app.py) build
against the same contract instead of each inventing their own dict shape.
"""

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROCESSED_DATA_DIR

RUNS_DIR = PROCESSED_DATA_DIR / "runs"
RESCORINGS_DIR = PROCESSED_DATA_DIR / "rescorings"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Writes JSON atomically: a full write to a temp file in the same
    directory (so the rename is on the same filesystem), then os.replace()
    into place. A killed process or an interrupted Streamlit rerun can never
    leave a truncated file at `path` - either the old content is there, or
    the new content is, never a half-written mix of both."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2))
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _read_json_or_none(path: Path) -> dict | None:
    """Reads and parses JSON, returning None (instead of raising) if the
    file is missing, unreadable, or corrupt - so one bad file can be
    skipped rather than crashing every caller that lists runs/rescorings."""
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


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
    failed: bool = False
    failure_reason: str = ""
    dataset_id: str = ""
    loop_id: str | None = None
    round_index: int = 0
    revised_from_run_id: str | None = None
    time_column: str = ""
    # "classification" or "regression". Defaulted so every run recorded
    # before this field existed still loads via RunReport(**data) - list_runs
    # silently skips a TypeError on unknown/missing fields, so a field with
    # no default would make every historical run vanish from the UI with no
    # error shown.
    task_type: str = "classification"
    # {"rmse": float, "mae": float, "r2": float} for a regression run, empty
    # for classification - mirrors classification_report staying {} for a
    # regression run. holdout_accuracy still holds the headline higher-is-
    # better metric for both task types (accuracy, or r2 for regression) -
    # feature_loop.select_loop_winner relies on that invariant.
    regression_metrics: dict = field(default_factory=dict)
    # Free-text instruction the person reviewing this loop's progress left
    # for this specific round, or "" if none was given (every round before
    # this feature existed, and round 1 of any loop - see
    # agent.RevisionContext.user_instruction, ADR-021).
    mid_loop_instruction: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def save_run(report: RunReport) -> Path:
    """Writes the run record as plain JSON, matching Research-agent's evaluation harness."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{report.run_id}.json"
    _atomic_write_json(path, report.to_dict())
    return path


def load_latest_run() -> RunReport | None:
    """Returns the most recently written run record, or None if no run exists yet
    (or the most recent file on disk fails to parse)."""
    if not RUNS_DIR.exists():
        return None
    run_files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in run_files:
        data = _read_json_or_none(path)
        if data is not None:
            try:
                return RunReport(**data)
            except TypeError:
                continue
    return None


def list_runs(dataset_id: str | None = None) -> list[RunReport]:
    """Returns all saved run records, most recent first. A file that fails to
    parse (e.g. left truncated by an interrupted write) is skipped rather
    than raised, so one corrupted run can't take down the whole list.
    Pass dataset_id to restrict to runs trained against that dataset."""
    if not RUNS_DIR.exists():
        return []
    run_files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for path in run_files:
        data = _read_json_or_none(path)
        if data is None:
            continue
        try:
            runs.append(RunReport(**data))
        except TypeError:
            continue
    if dataset_id is not None:
        runs = [r for r in runs if r.dataset_id == dataset_id]
    return runs


def list_loop_attempts(loop_id: str) -> list[RunReport]:
    """Returns every saved run belonging to one feature-proposal-loop
    invocation, ordered by round_index rather than recency - callers need
    the attempt sequence a loop took, not when each attempt happened to
    be written to disk."""
    runs = [r for r in list_runs() if r.loop_id == loop_id]
    return sorted(runs, key=lambda r: r.round_index)


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
    _atomic_write_json(path, record)
    return path


def list_rescorings(run_id: str | None = None) -> list[dict]:
    """Returns saved rescoring records, most recent first, optionally filtered
    to one run_id. A file that fails to parse is skipped rather than raised."""
    if not RESCORINGS_DIR.exists():
        return []
    files = sorted(RESCORINGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    records = [r for r in (_read_json_or_none(p) for p in files) if r is not None]
    if run_id is not None:
        records = [r for r in records if r["run_id"] == run_id]
    return records
