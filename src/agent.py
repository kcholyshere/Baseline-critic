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
import hashlib
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types as genai_types
from mcp import StdioServerParameters
from sklearn.metrics import accuracy_score, classification_report

from src import config, dataset
from src.checks import evidence_categories, run_static_checks
from src.critic import Critique, critique_run_async
from src.profiling import format_profile_for_prompt, profile_dataframe
from src.report import RunReport, save_run
from src.services.code_execution import ExecutionResult, run_code
from src.services.docs_mcp_server import get_api_signature
from src.services.observability import langfuse  # noqa: F401 - instruments ADK before Agent() below

MAX_TRAINING_ATTEMPTS = 3

APP_NAME = "baseline_critic"
USER_ID = "demo-user"


@dataclass
class RevisionContext:
    """Carries what happened on a previous loop attempt into the next one's
    instruction, so a revision round can react to the actual verdict rather
    than repeating the same script blind."""

    previous_code: str
    previous_verdict: str  # "accept" or "reject"
    defect_category: str
    defect: str
    evidence: str
    previous_accuracy: float
    prior_summaries: list[str]


def _build_instruction(
    target_column: str,
    positive_class: str,
    profile_text: str,
    use_docs_tool: bool = False,
    inject_signature: bool = False,
    allow_retry: bool = False,
    revision_context: RevisionContext | None = None,
) -> str:
    docs_tool_line = (
        "\nYou also have a get_api_signature tool that returns the real, "
        "currently installed signature and docstring for a pandas, sklearn, "
        "or lightgbm function. Call it before using any function whose exact "
        "current keyword arguments you are not certain of - library APIs "
        "change between versions, and your training data may be stale "
        "relative to what is actually installed here.\n"
        if use_docs_tool
        else ""
    )
    signature_block = (
        "\nGround truth for the low-level lightgbm.train() API installed in "
        "this exact environment - use only these keyword arguments, since your "
        "training data may reflect an older, incompatible version:\n\n"
        f"{get_api_signature('lightgbm', 'train')}\n"
        if inject_signature
        else ""
    )
    tool_call_policy = (
        f"Call the run_training_code tool with the full script as a single "
        f"string. If it succeeds, reply with one sentence summarising what the "
        f"script did (features used, model type) and stop. If it fails (a "
        f"non-zero returncode), read stderr, fix the exact defect it names, "
        f"and call the tool again - up to {MAX_TRAINING_ATTEMPTS} attempts in "
        f"total. Do not call the tool again once it succeeds."
        if allow_retry
        else "Call the run_training_code tool exactly once with the full script "
        "as a single string. After the tool call returns, reply with one "
        "sentence summarising what the script did (features used, model "
        "type). Do not call the tool again once it succeeds."
    )
    revision_block = ""
    if revision_context is not None:
        if revision_context.previous_verdict == "reject":
            revision_block = f"""
Your previous attempt at this dataset was REJECTED by the critic. Do not
resubmit the same approach.
Defect category: {revision_context.defect_category}
Defect: {revision_context.defect}
Evidence: {revision_context.evidence}

Previous script:
```python
{revision_context.previous_code}
```

Write a new script that fixes this specific problem rather than repeating it.
"""
        else:
            prior_summaries_text = "\n".join(f"- {s}" for s in revision_context.prior_summaries)
            revision_block = f"""
You already have an accepted baseline scoring {revision_context.previous_accuracy:.4f} holdout accuracy.
Prior attempts so far:
{prior_summaries_text}

If you believe a genuinely different feature approach could improve on this
score, try it. If you cannot think of a meaningfully different approach, it
is fine to resubmit a similar one.
"""
    return f"""\
You are a baseline modelling agent for a tabular binary classification task.

Here is a deterministic profile of the dataset, computed with pandas - trust
these facts over any assumption you might otherwise make about the data:

{profile_text}
{docs_tool_line}{signature_block}
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
{revision_block}
{tool_call_policy}
"""


def _build_agent(
    capture: dict, instruction: str, model: str | LiteLlm, use_docs_tool: bool = False, allow_retry: bool = False
) -> Agent:
    max_attempts = MAX_TRAINING_ATTEMPTS if allow_retry else 1

    def run_training_code(code: str) -> dict:
        """Runs a self-contained Python training script in an isolated sandbox.

        Args:
            code: A full Python script, as a single string, that trains a
                model and saves it to "model.txt" in the current directory.
        """
        attempts = capture.get("attempts", 0)
        if attempts >= max_attempts:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"No attempts remaining ({max_attempts} allowed this run). Stop and report failure.",
                "timed_out": False,
                "artifact_names": [],
            }
        capture["attempts"] = attempts + 1

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

    tools: list = [run_training_code]
    if use_docs_tool:
        tools.append(
            McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=sys.executable, args=["-m", "src.services.docs_mcp_server"]
                    )
                )
            )
        )

    return Agent(
        name="baseline_agent",
        model=model,
        # A callable instruction bypasses ADK's regex-based {var} templating
        # entirely (it only scans a plain-string instruction) - the dataset
        # profile or the docs-tool signature block could otherwise contain a
        # bare {identifier}-shaped substring ADK would try, and fail, to
        # resolve from session state. See src/critic.py for the concrete
        # failure this was found from (an f-string placeholder in
        # generated code/stdout, not this agent's own instruction).
        instruction=lambda _ctx: instruction,
        tools=tools,
    )


def _normalise_feature_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]", "", name).lower()


def build_run_report_from_execution(
    execution: ExecutionResult,
    generated_code: str,
    run_id: str,
    dataset_name: str,
    target_column: str,
    train_rows: int,
    holdout: pd.DataFrame,
    positive_class: str,
    negative_class: str,
    duration_seconds: float,
    model_name: str,
    attempts: int,
    agent_summary: str,
    modeller_prompt_tokens: int = 0,
    modeller_completion_tokens: int = 0,
) -> RunReport:
    """Turns a finished sandbox execution into a scored RunReport - the same
    booster-load, feature-reindex, and holdout-scoring steps _run_baseline_async
    uses, factored out so src/evaluation.py's hand-written defect fixtures are
    scored through the identical path a real agent run goes through, rather
    than a second copy that could quietly drift from it.
    """
    if execution.returncode != 0:
        raise RuntimeError(f"Training script failed:\n{execution.stderr}")

    model_bytes = execution.artifacts.get("model.txt")
    if model_bytes is None:
        raise RuntimeError("Training script did not save model.txt.")
    booster = lgb.Booster(model_str=model_bytes.decode())

    X_holdout = holdout.drop(columns=[target_column])
    y_holdout = (holdout[target_column] == positive_class).astype(int)
    X_holdout = _reindex_to_booster_feature_order(X_holdout, booster)
    y_pred = (booster.predict(X_holdout) >= 0.5).astype(int)

    return RunReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name=dataset_name,
        target_column=target_column,
        train_rows=train_rows,
        holdout_rows=len(holdout),
        agent_summary=agent_summary,
        generated_code=generated_code,
        stdout=execution.stdout[-4000:],
        holdout_accuracy=accuracy_score(y_holdout, y_pred),
        classification_report=classification_report(
            y_holdout, y_pred, target_names=[negative_class, positive_class], output_dict=True
        ),
        duration_seconds=duration_seconds,
        positive_class=positive_class,
        model=model_name,
        attempts=attempts,
        modeller_prompt_tokens=modeller_prompt_tokens,
        modeller_completion_tokens=modeller_completion_tokens,
    )


def _reindex_to_booster_feature_order(X: pd.DataFrame, booster: lgb.Booster) -> pd.DataFrame:
    """Booster.predict() on a DataFrame matches columns by position, not name.
    LightGBM's saved feature names are sanitised (e.g. spaces become
    underscores), so they can't be compared to the holdout's real column
    names directly - normalise both sides before matching. A generated
    script that reorders its feature columns (observed in practice: building
    the feature list via `.columns.difference(...)`, which sorts
    alphabetically) would otherwise silently misalign every prediction
    against the wrong feature, with no exception raised."""
    real_by_normalised = {_normalise_feature_name(c): c for c in X.columns}
    if len(real_by_normalised) < len(X.columns):
        raise RuntimeError(
            "Two or more feature columns normalise to the same name (case/punctuation "
            "collision, e.g. 'Age' and 'age') - the booster-feature reindex can only map "
            "one real column per normalised key, so proceeding would silently misalign "
            "or drop a feature with no exception, the same class of bug ADR-009 fixed. "
            f"Columns: {list(X.columns)}"
        )
    ordered_columns = [real_by_normalised[_normalise_feature_name(fn)] for fn in booster.feature_name()]
    return X[ordered_columns]


async def _run_baseline_async(
    train_path: Path,
    target_column: str,
    positive_class: str,
    negative_class: str,
    holdout: pd.DataFrame,
    dataset_name: str,
    model: str | LiteLlm,
    use_docs_tool: bool = False,
    inject_signature: bool = True,
    allow_retry: bool = True,
    run_critic: bool = True,
    loop_id: str | None = None,
    round_index: int = 0,
    revised_from_run_id: str | None = None,
    revision_context: RevisionContext | None = None,
) -> RunReport:
    started = time.monotonic()

    train_df = pd.read_csv(train_path)
    dataset_id = hashlib.sha256(train_path.read_bytes()).hexdigest()[:12]
    profile = profile_dataframe(train_df, target_column)
    profile_text = format_profile_for_prompt(profile)

    capture: dict = {}
    instruction = _build_instruction(
        target_column, positive_class, profile_text, use_docs_tool, inject_signature, allow_retry, revision_context
    )
    agent = _build_agent(capture, instruction, model, use_docs_tool, allow_retry)
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    prompt = (
        f"Training data CSV path: {train_path}\n"
        f"Target column: {target_column}\n"
        "Train a LightGBM baseline as instructed."
    )
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])

    final_text = ""
    modeller_prompt_tokens = 0
    modeller_completion_tokens = 0
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        if event.usage_metadata is not None:
            modeller_prompt_tokens += event.usage_metadata.prompt_token_count or 0
            modeller_completion_tokens += event.usage_metadata.candidates_token_count or 0
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts)

    execution: ExecutionResult | None = capture.get("result")
    run_id = uuid.uuid4().hex[:12]

    try:
        if execution is None:
            raise RuntimeError("Agent finished without calling run_training_code.")

        report = build_run_report_from_execution(
            execution=execution,
            generated_code=capture.get("code", ""),
            run_id=run_id,
            dataset_name=dataset_name,
            target_column=target_column,
            train_rows=len(train_df),
            holdout=holdout,
            positive_class=positive_class,
            negative_class=negative_class,
            duration_seconds=time.monotonic() - started,
            model_name=model if isinstance(model, str) else model.model,
            attempts=capture.get("attempts", 0),
            agent_summary=final_text,
            modeller_prompt_tokens=modeller_prompt_tokens,
            modeller_completion_tokens=modeller_completion_tokens,
        )
        report.dataset_id = dataset_id
        report.loop_id = loop_id
        report.round_index = round_index
        report.revised_from_run_id = revised_from_run_id
    except Exception as exc:
        # A permanently failed run (non-zero returncode, missing model.txt, or
        # the agent never calling the tool at all) must still leave a JSON
        # record behind - otherwise this failure mode leaves no trace, unlike
        # every successful run (see ADR-008's repeated early_stopping_rounds
        # crashes, which left nothing to inspect afterwards).
        failure_report = RunReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=dataset_name,
            target_column=target_column,
            train_rows=len(train_df),
            holdout_rows=len(holdout),
            agent_summary=final_text,
            generated_code=capture.get("code", ""),
            stdout=execution.stdout[-4000:] if execution is not None else "",
            holdout_accuracy=0.0,
            classification_report={},
            duration_seconds=time.monotonic() - started,
            positive_class=positive_class,
            model=model if isinstance(model, str) else model.model,
            attempts=capture.get("attempts", 0),
            modeller_prompt_tokens=modeller_prompt_tokens,
            modeller_completion_tokens=modeller_completion_tokens,
            failed=True,
            failure_reason=str(exc),
        )
        failure_report.dataset_id = dataset_id
        failure_report.loop_id = loop_id
        failure_report.round_index = round_index
        failure_report.revised_from_run_id = revised_from_run_id
        save_run(failure_report)
        raise

    if run_critic:
        static_findings = run_static_checks(
            report.generated_code, train_path, target_column, report.stdout, report.holdout_accuracy
        )
        static_evidence = evidence_categories(
            report.generated_code, train_path, target_column, report.stdout, report.holdout_accuracy
        )
        critique = await critique_run_async(report, profile_text, static_findings, model, static_evidence)
        report.critique = critique.to_dict()

    save_run(report)
    return report


async def rescore_run_async(
    report: RunReport, train_path: Path, model: str | LiteLlm | None = None
) -> Critique:
    """Re-runs the static checks and critic against an already-saved
    RunReport, without touching the sandbox or the modeller agent (Phase 5
    TODO: "everything should be deterministic and re-scorable from stored
    runs"). `train_path` must point at the same train CSV the run was scored
    against - for the demo dataset this is dataset.TRAIN_PATH regenerated via
    dataset.build_train_artifact(), never the (gitignored, possibly absent)
    original file. Returns a fresh Critique; callers decide whether to keep
    it via report.save_rescoring - the original run's own `critique` field is
    never overwritten, so past and re-scored verdicts stay comparable.
    """
    train_df = pd.read_csv(train_path)
    profile_text = format_profile_for_prompt(profile_dataframe(train_df, report.target_column))
    static_findings = run_static_checks(
        report.generated_code, train_path, report.target_column, report.stdout, report.holdout_accuracy
    )
    static_evidence = evidence_categories(
        report.generated_code, train_path, report.target_column, report.stdout, report.holdout_accuracy
    )
    return await critique_run_async(report, profile_text, static_findings, _resolve_model(model), static_evidence)


def _resolve_model(model: str | LiteLlm | None) -> str | LiteLlm:
    """Defaults to a fresh LiteLlm pointed at the local model (config.DEFAULT_MODEL_URI)
    rather than a single shared instance, since a default argument value is only
    constructed once at function-definition time."""
    return model if model is not None else LiteLlm(model=config.DEFAULT_MODEL_URI)


def run_baseline(
    model: str | LiteLlm | None = None,
    use_docs_tool: bool = False,
    inject_signature: bool = True,
    allow_retry: bool = True,
    run_critic: bool = True,
) -> RunReport:
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
            model=_resolve_model(model),
            use_docs_tool=use_docs_tool,
            inject_signature=inject_signature,
            allow_retry=allow_retry,
            run_critic=run_critic,
        )
    )


def run_baseline_for(
    train_path: Path,
    target_column: str,
    positive_class: str,
    negative_class: str,
    holdout: pd.DataFrame,
    dataset_name: str,
    model: str | LiteLlm | None = None,
    use_docs_tool: bool = False,
    inject_signature: bool = True,
    allow_retry: bool = True,
    run_critic: bool = True,
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
            model=_resolve_model(model),
            use_docs_tool=use_docs_tool,
            inject_signature=inject_signature,
            allow_retry=allow_retry,
            run_critic=run_critic,
        )
    )


if __name__ == "__main__":
    result = run_baseline()
    print(f"Holdout accuracy: {result.holdout_accuracy:.4f}")
    print(result.agent_summary)
