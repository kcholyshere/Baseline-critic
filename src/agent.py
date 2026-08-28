"""Baseline agent: writes and runs one LightGBM training script.

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

Generalised to any binary-classification CSV, not just the built-in demo
dataset: `run_baseline()` and `run_baseline_for()` are two thin wrappers
over the same core, supplying the demo dataset's fixed values or an
uploaded dataset's values respectively. Scoped to binary classification
only, matching the proposal's own first-version scope.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from sklearn.metrics import accuracy_score, classification_report

from src import config, dataset
from src.profiling import format_profile_for_prompt, profile_dataframe
from src.report import RunReport, save_run
from src.services.code_execution import ExecutionResult, run_code
from src.services.observability import langfuse  # noqa: F401 - instruments ADK before Agent() below

APP_NAME = "baseline_critic"
USER_ID = "demo-user"


def _build_instruction(target_column: str, positive_class: str, profile_text: str) -> str:
    return f"""\
You are a baseline modelling agent for a tabular binary classification task.

Here is a deterministic profile of the dataset, computed with pandas - trust
these facts over any assumption you might otherwise make about the data:

{profile_text}

You will be told the path to a training CSV and its target column, which
holds two string values. The positive class (the outcome of interest) is
"{positive_class}". Write one self-contained Python script that:
1. Loads the CSV at the given path with pandas. Do not read, write, or
   reference any other file.
2. Builds a binary label column: 1 where the target equals "{positive_class}",
   else 0.
3. Splits off its own internal validation slice from that CSV only.
4. Trains with the low-level `lightgbm.train()` API (not LGBMClassifier) on
   a `lightgbm.Dataset` built from every column except the target column
   "{target_column}" and the label column you added - exclude any column the
   profile above flags as a likely identifier - with params
   {{"objective": "binary", "metric": "binary_logloss", "verbosity": -1}}.
5. Prints its own validation accuracy at a 0.5 probability threshold.
6. Saves the trained booster with `booster.save_model("model.txt")` -
   LightGBM's own text format. Do not use pickle or joblib.

Call the run_training_code tool exactly once with the full script as a
single string. After the tool call returns, reply with one sentence
summarising what the script did (features used, model type). Do not call
the tool again once it succeeds.
"""


def _build_agent(capture: dict, instruction: str) -> Agent:
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
        instruction=instruction,
        tools=[run_training_code],
    )


async def _run_baseline_async(
    train_path: Path,
    target_column: str,
    positive_class: str,
    negative_class: str,
    holdout: pd.DataFrame,
    dataset_name: str,
) -> RunReport:
    started = time.monotonic()

    train_df = pd.read_csv(train_path)
    profile = profile_dataframe(train_df, target_column)
    profile_text = format_profile_for_prompt(profile)

    capture: dict = {}
    instruction = _build_instruction(target_column, positive_class, profile_text)
    agent = _build_agent(capture, instruction)
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    prompt = (
        f"Training data CSV path: {train_path}\n"
        f"Target column: {target_column}\n"
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

    X_holdout = holdout.drop(columns=[target_column])
    y_holdout = (holdout[target_column] == positive_class).astype(int)
    y_pred = (booster.predict(X_holdout) >= 0.5).astype(int)

    report = RunReport(
        run_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name=dataset_name,
        target_column=target_column,
        train_rows=len(train_df),
        holdout_rows=len(holdout),
        agent_summary=final_text,
        generated_code=generated_code,
        stdout=execution.stdout[-4000:],
        holdout_accuracy=accuracy_score(y_holdout, y_pred),
        classification_report=classification_report(
            y_holdout, y_pred, target_names=[negative_class, positive_class], output_dict=True
        ),
        duration_seconds=time.monotonic() - started,
        positive_class=positive_class,
    )
    save_run(report)
    return report


def run_baseline() -> RunReport:
    """Runs one full baseline cycle against the built-in Breast Cancer Wisconsin demo dataset."""
    dataset.build_train_artifact()
    return asyncio.run(
        _run_baseline_async(
            train_path=dataset.TRAIN_PATH,
            target_column=dataset.TARGET_COLUMN,
            positive_class="malignant",
            negative_class="benign",
            holdout=dataset.get_holdout(),
            dataset_name="breast_cancer_wisconsin",
        )
    )


def run_baseline_for(
    train_path: Path,
    target_column: str,
    positive_class: str,
    negative_class: str,
    holdout: pd.DataFrame,
    dataset_name: str,
) -> RunReport:
    """Runs one full baseline cycle against an uploaded dataset (see dataset.prepare_uploaded_dataset)."""
    return asyncio.run(
        _run_baseline_async(
            train_path=train_path,
            target_column=target_column,
            positive_class=positive_class,
            negative_class=negative_class,
            holdout=holdout,
            dataset_name=dataset_name,
        )
    )


if __name__ == "__main__":
    result = run_baseline()
    print(f"Holdout accuracy: {result.holdout_accuracy:.4f}")
    print(result.agent_summary)
