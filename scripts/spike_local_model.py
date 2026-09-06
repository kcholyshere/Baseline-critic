"""Local-model spike: repeated-trial harness comparing src/agent.py's real
task (not a published benchmark) across model/instruction configurations.

qwen2.5-coder:7b was dropped entirely after this harness's own results
(references/local-model-benchmarks.md, 2026-09-05) - it stayed unreliable
even with every fix applied. qwen2.5-coder:14b is the project's default
model (config.DEFAULT_MODEL_URI, ADR-008); this script's job now is to
keep measuring its actual reliability, not to pick between candidates.

"inject_signature" pastes the real, currently-installed lightgbm.train
signature straight into the instruction, rather than exposing it as an
optional tool the model has to remember to call - the earlier optional-tool
version was tested and shown not to help, since the model never called it.
"allow_retry" lets the agent see a failed script's stderr and try again,
capped at MAX_TRAINING_ATTEMPTS (src/agent.py) - enforced in code, not
just asked for in the instruction.

Each trial runs in its own subprocess, not in-process in a loop: the
code-execution sandbox's training-call budget (MAX_CALLS_PER_SESSION,
src/services/code_execution.py) is a module-level counter meant to bound
one run's session, not N independent trials sharing one Python process -
an earlier in-process version of this script hit that ceiling partway
through and produced false failures that were a harness bug, not a model
result.

Requires Ollama running locally with the model already pulled:
    ollama pull qwen2.5-coder:14b
"""

import json
import subprocess
import sys
from dataclasses import dataclass

TRIALS_PER_CONDITION = 5

MODEL_URIS = {
    "qwen2.5-coder:14b": "ollama_chat/qwen2.5-coder:14b",
}


@dataclass
class ConditionResult:
    model: str
    use_docs_tool: bool
    successes: int
    trials: int
    accuracies: list[float]
    durations: list[float]

    @property
    def success_rate(self) -> str:
        return f"{self.successes}/{self.trials}"

    @property
    def mean_accuracy(self) -> str:
        return f"{sum(self.accuracies) / len(self.accuracies):.4f}" if self.accuracies else "-"

    @property
    def mean_duration(self) -> str:
        return f"{sum(self.durations) / len(self.durations):.1f}" if self.durations else "-"


def _run_one_trial(model_uri: str, use_docs_tool: bool, inject_signature: bool, allow_retry: bool) -> dict:
    """Runs exactly one baseline trial in a fresh subprocess and returns its
    result as a dict, parsed from the single JSON line the subprocess prints."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.spike_local_model",
            "--trial",
            model_uri,
            str(use_docs_tool),
            str(inject_signature),
            str(allow_retry),
        ],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("TRIAL_RESULT "):
            return json.loads(line[len("TRIAL_RESULT ") :])
    return {"ok": False, "error": f"no result line; stderr tail: {proc.stderr[-500:]}"}


def _trial_subprocess_main(model_uri: str, use_docs_tool: bool, inject_signature: bool, allow_retry: bool) -> None:
    """Entry point for the child subprocess: one trial, one process, one
    fresh training-call budget, then print the result as JSON and exit."""
    from google.adk.models.lite_llm import LiteLlm

    from src.agent import run_baseline

    try:
        report = run_baseline(
            model=LiteLlm(model=model_uri),
            use_docs_tool=use_docs_tool,
            inject_signature=inject_signature,
            allow_retry=allow_retry,
        )
        result = {
            "ok": True,
            "accuracy": report.holdout_accuracy,
            "duration": report.duration_seconds,
            "attempts": report.attempts,
        }
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[-300:]}"}
    print(f"TRIAL_RESULT {json.dumps(result)}")


def run_condition(
    label: str, use_docs_tool: bool = False, inject_signature: bool = False, allow_retry: bool = False, tag: str = ""
) -> ConditionResult:
    result = ConditionResult(
        model=label, use_docs_tool=use_docs_tool, successes=0, trials=TRIALS_PER_CONDITION, accuracies=[], durations=[]
    )
    for trial in range(1, TRIALS_PER_CONDITION + 1):
        print(f"\n=== {label} [{tag}] trial {trial}/{TRIALS_PER_CONDITION} ===")
        trial_result = _run_one_trial(MODEL_URIS[label], use_docs_tool, inject_signature, allow_retry)
        if trial_result["ok"]:
            result.successes += 1
            result.accuracies.append(trial_result["accuracy"])
            result.durations.append(trial_result["duration"])
            print(
                f"SUCCESS holdout accuracy: {trial_result['accuracy']:.4f}, "
                f"duration: {trial_result['duration']:.1f}s, attempts: {trial_result['attempts']}"
            )
        else:
            print(f"FAILED: {trial_result['error']}")
    return result


def main() -> None:
    results: list[tuple[str, ConditionResult]] = []
    for label in MODEL_URIS:
        results.append(("fixed", run_condition(label, inject_signature=True, allow_retry=True, tag="fixed")))

    print("\n=== fixed-condition comparison (signature injection + bounded retry) ===")
    print(f"{'model':<20}{'condition':<10}{'success rate':<14}{'mean accuracy':<16}{'mean duration (s)':<18}")
    for tag, r in results:
        print(f"{r.model:<20}{tag:<10}{r.success_rate:<14}{r.mean_accuracy:<16}{r.mean_duration:<18}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--trial":
        # Internal-only: always invoked by _run_one_trial with exactly these
        # five args in this order. Asserted rather than left to a bare
        # IndexError, so a future edit to either side that drifts out of
        # sync fails with a clear message instead of a confusing traceback.
        assert len(sys.argv) == 6, f"--trial expects 5 args, got {len(sys.argv) - 2}: {sys.argv[2:]}"
        _trial_subprocess_main(
            model_uri=sys.argv[2],
            use_docs_tool=sys.argv[3] == "True",
            inject_signature=sys.argv[4] == "True",
            allow_retry=sys.argv[5] == "True",
        )
    else:
        main()
