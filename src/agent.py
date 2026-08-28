"""Minimal baseline agent: writes and runs one LightGBM training script.

Demo-scoped (2026-08-28): a single-shot agent, not yet the
planner/modeller/critic loop the proposal describes (references/project-proposal.md).
It writes one training script, runs it in the code-execution sandbox
(src/services/code_execution.py), and hands the fitted model back to this
module, which scores it against the holdout in-process - the holdout is
never passed to the agent or the sandbox, per ADR-005.

The model crosses the sandbox boundary as LightGBM's own text model format
(Booster.save_model/model_str), not pickle: pickle.load on bytes the agent's
own generated code produced would let a malicious or buggy script run
arbitrary code in this trusted parent process, defeating the isolation
ADR-001/ADR-003 put around the sandbox in the first place. The text format
is tree weights, not executable bytecode.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from sklearn.metrics import accuracy_score, classification_report

from src import config, dataset
from src.report import RunReport, save_run
from src.services.code_execution import ExecutionResult, run_code
from src.services.observability import langfuse  # noqa: F401 - instruments ADK before Agent() below

APP_NAME = "baseline_critic"
USER_ID = "demo-user"
POSITIVE_CLASS = "malignant"

INSTRUCTION = f"""\
You are a baseline modelling agent for a tabular classification task.

You will be told the path to a training CSV and its target column, which
holds the string values "benign" and "malignant". Write one self-contained
Python script that:
1. Loads the CSV at the given path with pandas. Do not read, write, or
   reference any other file.
2. Builds a binary label column: 1 where the target equals "{POSITIVE_CLASS}",
   else 0.
3. Splits off its own internal validation slice from that CSV only.
4. Trains with the low-level `lightgbm.train()` API (not LGBMClassifier) on
   a `lightgbm.Dataset` built from every column except the target column and
   the label column you added, with params {{"objective": "binary",
   "metric": "binary_logloss", "verbosity": -1}}.
5. Prints its own validation accuracy at a 0.5 probability threshold.
6. Saves the trained booster with `booster.save_model("model.txt")` -
   LightGBM's own text format. Do not use pickle or joblib.

Call the run_training_code tool exactly once with the full script as a
single string. After the tool call returns, reply with one sentence
summarising what the script did (features used, model type). Do not call
the tool again once it succeeds.
"""


def _build_agent(capture: dict) -> Agent:
    def run_training_code(code: str) -> dict:
        """Runs a self-contained Python training script in an isolated sandbox.

        Args:
            code: A full Python script, as a single string, that trains a
                model and saves it to "model.txt" in the current directory.
        """
        result = run_code(code)
        capture["result"] = result
        capture["code"] = code
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-2000:],
            "timed_out": result.timed_out,
            "artifact_names": list(result.artifacts),
        }

    return Agent(
        name="baseline_agent",
        model=config.GEMINI_MODEL,
        instruction=INSTRUCTION,
        tools=[run_training_code],
    )


async def _run_baseline_async() -> RunReport:
    started = time.monotonic()
    dataset.build_train_artifact()

    capture: dict = {}
    agent = _build_agent(capture)
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    prompt = (
        f"Training data CSV path: {dataset.TRAIN_PATH}\n"
        f"Target column: {dataset.TARGET_COLUMN}\n"
        "Train a LightGBM baseline as instructed."
    )
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])

    final_text = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts)

    execution: ExecutionResult | None = capture.get("result")
    if execution is None:
        raise RuntimeError("Agent finished without calling run_training_code.")
    if execution.returncode != 0:
        raise RuntimeError(f"Training script failed:\n{execution.stderr}")

    model_bytes = execution.artifacts.get("model.txt")
    if model_bytes is None:
        raise RuntimeError("Training script did not save model.txt.")
    booster = lgb.Booster(model_str=model_bytes.decode())
    generated_code = capture.get("code", "")

    holdout = dataset.get_holdout()
    X_holdout = holdout.drop(columns=[dataset.TARGET_COLUMN])
    y_holdout = (holdout[dataset.TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
    y_pred = (booster.predict(X_holdout) >= 0.5).astype(int)
    train_rows = len(pd.read_csv(dataset.TRAIN_PATH))

    report = RunReport(
        run_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name="breast_cancer_wisconsin",
        target_column=dataset.TARGET_COLUMN,
        train_rows=train_rows,
        holdout_rows=len(holdout),
        agent_summary=final_text,
        generated_code=generated_code,
        stdout=execution.stdout[-4000:],
        holdout_accuracy=accuracy_score(y_holdout, y_pred),
        classification_report=classification_report(
            y_holdout, y_pred, target_names=["benign", "malignant"], output_dict=True
        ),
        duration_seconds=time.monotonic() - started,
    )
    save_run(report)
    return report


def run_baseline() -> RunReport:
    """Runs one full baseline cycle: agent trains a model, scored against the holdout."""
    return asyncio.run(_run_baseline_async())


if __name__ == "__main__":
    result = run_baseline()
    print(f"Holdout accuracy: {result.holdout_accuracy:.4f}")
    print(result.agent_summary)
