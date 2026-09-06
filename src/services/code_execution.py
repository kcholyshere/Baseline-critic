"""Runs agent-generated Python code in an isolated scratch directory.

Filesystem and network access are NOT technically restricted here - isolation
is by convention (a throwaway cwd) only, not enforcement. See ADR-001 for the
accepted risk and the container fallback this defers to.
"""

import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_SECONDS = 120
MAX_CALLS_PER_SESSION = 20
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024  # 50 MB per artifact file

_call_count = 0
_call_count_lock = threading.Lock()


class TrainingBudgetExceeded(Exception):
    """Raised once a session exceeds MAX_CALLS_PER_SESSION code-execution calls."""


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    artifacts: dict[str, bytes]  # files the code wrote, keyed by relative path


def run_code(code: str) -> ExecutionResult:
    """Run `code` as a standalone Python script in a fresh scratch directory.

    Uses this project's own interpreter, so pandas/scikit-learn/LightGBM are
    importable without a second environment. Raises TrainingBudgetExceeded
    once MAX_CALLS_PER_SESSION is reached - a hard ceiling, not agent-
    adjustable. The scratch directory is always deleted before returning;
    any file the code wrote is captured into `artifacts` first.
    """
    global _call_count
    with _call_count_lock:
        if _call_count >= MAX_CALLS_PER_SESSION:
            raise TrainingBudgetExceeded(
                f"Training budget of {MAX_CALLS_PER_SESSION} calls exhausted this session."
            )
        _call_count += 1

    scratch_dir = Path(tempfile.mkdtemp(prefix="baseline-critic-run-"))
    script_path = scratch_dir / "run.py"
    script_path.write_text(code)

    timed_out = False
    try:
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=scratch_dir,
                timeout=TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n[timed out after {TIMEOUT_SECONDS}s]"
            returncode = -1

        artifacts: dict[str, bytes] = {}
        skipped: list[str] = []
        for f in scratch_dir.rglob("*"):
            if not f.is_file() or f == script_path:
                continue
            size = f.stat().st_size
            if size > MAX_ARTIFACT_BYTES:
                skipped.append(f"{f.relative_to(scratch_dir)} ({size} bytes)")
                continue
            artifacts[str(f.relative_to(scratch_dir))] = f.read_bytes()

        if skipped:
            stderr += (
                f"\n[skipped {len(skipped)} artifact(s) exceeding "
                f"{MAX_ARTIFACT_BYTES} byte cap: {', '.join(skipped)}]"
            )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
        artifacts=artifacts,
    )
