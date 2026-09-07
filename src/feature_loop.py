"""The feature-proposal loop (Phase 4): runs several revision rounds of the
baseline agent against the same dataset, feeding each round's outcome -
accepted score or the critic's named defect - into the next round's prompt
via agent.RevisionContext, then picks the best accepted attempt.

This module only orchestrates sequential src.agent._run_baseline_async
calls and the winner-selection rule; the revision prompt itself, the
sandbox, and the critic are unchanged - see src/agent.py and src/critic.py.

run_loop_round_async runs exactly one round against a LoopState, so a
caller can pause between rounds - src/ui/app.py's "talk with your data"
flow (ADR-021) drives it one round per user click, threading a free-text
user_instruction into the next round's RevisionContext. run_feature_loop_async
is a thin, unchanged-behaviour wrapper over the same function for the
non-interactive CLI/single-call callers below.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from google.adk.models.lite_llm import LiteLlm

from src import dataset
from src.agent import RevisionContext, _resolve_model, _run_baseline_async
from src.report import RunReport

# Bounded the same way agent.py's MAX_TRAINING_ATTEMPTS and critic.py's
# MAX_CRITIQUE_ROUNDS are: a code-enforced ceiling, not a request. 5 rounds
# at up to MAX_TRAINING_ATTEMPTS=3 sandbox calls each is 15 calls per loop
# invocation against code_execution.MAX_CALLS_PER_SESSION=60 - comfortably
# leaves room for several loop invocations (or a loop plus standalone runs)
# within one process's lifetime, per the reasoning bumping that budget.
# Raised from 3 to 5 (2026-09-06) after live use showed the local model
# sometimes burns a round or two failing to call the training tool at all
# (references/local-model-benchmarks.md) - more rounds gives the loop a
# realistic chance to still get a few genuine feature-iteration attempts in.
MAX_REVISION_ROUNDS = 5


@dataclass
class LoopResult:
    loop_id: str
    attempts: list[RunReport]
    winner: RunReport | None
    # Counted once per loop iteration regardless of outcome, unlike
    # len(attempts) - a round that raises (MAX_TRAINING_ATTEMPTS exhausted,
    # sandbox budget exhaustion) never joins `attempts` but still ran, and
    # the UI's "no accepted baseline" message needs the true round count
    # (audit-2026-09-06-post-loop.md finding #6).
    rounds_attempted: int = 0


def select_loop_winner(attempts: list[RunReport]) -> RunReport | None:
    """Picks the highest-holdout-accuracy attempt among those the critic
    actually accepted. Returns None if every attempt was rejected or every
    attempt failed before critique ran (critique is None either way) - a
    loop with no accepted result has no winner to report, not a fallback
    to the least-bad reject."""
    accepted = [a for a in attempts if a.critique is not None and a.critique["verdict"] == "accept"]
    if not accepted:
        return None
    return max(accepted, key=lambda a: a.holdout_accuracy)


@dataclass
class LoopState:
    """Accumulated state carried from one loop round to the next - what
    run_feature_loop_async's `for` loop used to hold as local variables,
    now extracted so a caller can run one round at a time (src/ui/app.py's
    interactive "talk with your data" flow, ADR-021) and inspect or persist
    the state in between, rather than only after every round has run."""

    loop_id: str
    attempts: list[RunReport] = field(default_factory=list)
    last_successful: RunReport | None = None
    prior_summaries: list[str] = field(default_factory=list)
    rounds_attempted: int = 0

    def to_result(self) -> LoopResult:
        return LoopResult(
            loop_id=self.loop_id,
            attempts=self.attempts,
            winner=select_loop_winner(self.attempts),
            rounds_attempted=self.rounds_attempted,
        )


def new_loop_state() -> LoopState:
    return LoopState(loop_id=uuid.uuid4().hex[:12])


async def run_loop_round_async(
    state: LoopState,
    train_path: Path,
    target_column: str,
    positive_class: str,
    negative_class: str,
    holdout: pd.DataFrame,
    dataset_name: str,
    model: str | LiteLlm,
    time_column: str | None = None,
    task_type: str = "classification",
    user_instruction: str = "",
) -> LoopState:
    """Runs exactly one round against `state`, mutating and returning it.

    round_index is derived from state.rounds_attempted, not taken as a
    parameter - a caller-supplied round_index that must be kept in sync
    with state is an unforced positional-mismatch bug class (ADR-009 found
    one already in this codebase).

    user_instruction, if non-empty, is threaded into this round's
    RevisionContext.user_instruction (ADR-021, "talk with your data") - it
    has no effect when state.last_successful is None (round 1, or any round
    following an unbroken run of failures), since revision_context stays
    None in that case. Callers should disable instruction input in that
    state rather than silently accepting text that will never reach the
    modeller."""
    round_index = state.rounds_attempted + 1
    state.rounds_attempted += 1
    revision_context: RevisionContext | None = None
    if state.last_successful is not None:
        critique = state.last_successful.critique
        is_reject = critique["verdict"] == "reject"
        revision_context = RevisionContext(
            previous_code=state.last_successful.generated_code,
            previous_verdict=critique["verdict"],
            defect_category=critique["defect_category"] if is_reject else "",
            defect=critique["defect"] if is_reject else "",
            evidence=critique["evidence"] if is_reject else "",
            previous_accuracy=state.last_successful.holdout_accuracy,
            prior_summaries=list(state.prior_summaries),
            user_instruction=user_instruction,
        )

    try:
        report = await _run_baseline_async(
            train_path=train_path,
            target_column=target_column,
            positive_class=positive_class,
            negative_class=negative_class,
            holdout=holdout,
            dataset_name=dataset_name,
            model=model,
            run_critic=True,
            loop_id=state.loop_id,
            round_index=round_index,
            revised_from_run_id=state.last_successful.run_id if state.last_successful is not None else None,
            revision_context=revision_context,
            time_column=time_column,
            task_type=task_type,
        )
    except Exception:
        # _run_baseline_async has already saved its own failure RunReport
        # to disk (src/agent.py) before re-raising - this round simply
        # doesn't join `attempts` or become the next round's revision
        # basis, it stays recoverable from disk like any other failure.
        # Callers must not assume attempts grew after this call.
        return state

    state.attempts.append(report)
    state.last_successful = report
    # agent_summary is written before critique runs, so it's
    # verdict-neutral by construction - tag it here so a rejected
    # round doesn't read identically to an accepted one once it lands
    # in a later round's "Prior attempts so far" prompt text (audit
    # finding #7).
    critique = report.critique
    metric_label = "r2" if report.task_type == "regression" else "acc"
    if critique["verdict"] == "accept":
        tag = f"[accepted, {metric_label}={report.holdout_accuracy:.4f}]"
    else:
        tag = f"[rejected: {critique['defect_category']}]"
    state.prior_summaries.append(f"{tag} {report.agent_summary}")
    return state


async def run_feature_loop_async(
    train_path: Path,
    target_column: str,
    positive_class: str,
    negative_class: str,
    holdout: pd.DataFrame,
    dataset_name: str,
    model: str | LiteLlm,
    max_rounds: int = MAX_REVISION_ROUNDS,
    time_column: str | None = None,
    task_type: str = "classification",
) -> LoopResult:
    state = new_loop_state()
    for _ in range(max_rounds):
        state = await run_loop_round_async(
            state,
            train_path,
            target_column,
            positive_class,
            negative_class,
            holdout,
            dataset_name,
            model,
            time_column,
            task_type,
        )
    return state.to_result()


def run_feature_loop(model: str | LiteLlm | None = None, max_rounds: int = MAX_REVISION_ROUNDS) -> LoopResult:
    """Runs one full feature-proposal loop against the built-in Breast Cancer
    Wisconsin demo dataset."""
    dataset.build_train_artifact()
    return asyncio.run(
        run_feature_loop_async(
            train_path=dataset.TRAIN_PATH,
            target_column=dataset.TARGET_COLUMN,
            positive_class="malignant",
            negative_class="benign",
            holdout=dataset.get_holdout(),
            dataset_name="breast_cancer_wisconsin",
            model=_resolve_model(model),
            max_rounds=max_rounds,
            task_type="classification",
        )
    )


def run_feature_loop_for(
    train_path: Path,
    target_column: str,
    positive_class: str,
    negative_class: str,
    holdout: pd.DataFrame,
    dataset_name: str,
    model: str | LiteLlm | None = None,
    max_rounds: int = MAX_REVISION_ROUNDS,
    time_column: str | None = None,
    task_type: str = "classification",
) -> LoopResult:
    """Runs one full feature-proposal loop against an uploaded dataset (see
    dataset.prepare_uploaded_dataset)."""
    return asyncio.run(
        run_feature_loop_async(
            train_path=train_path,
            target_column=target_column,
            positive_class=positive_class,
            negative_class=negative_class,
            holdout=holdout,
            dataset_name=dataset_name,
            model=_resolve_model(model),
            max_rounds=max_rounds,
            time_column=time_column,
            task_type=task_type,
        )
    )


if __name__ == "__main__":
    result = run_feature_loop()
    print(f"Loop ID: {result.loop_id}")
    for attempt in result.attempts:
        verdict = attempt.critique["verdict"] if attempt.critique else None
        print(f"Round {attempt.round_index}: verdict={verdict} holdout_accuracy={attempt.holdout_accuracy:.4f}")
    if result.winner is not None:
        print(f"Winner: round {result.winner.round_index}, holdout_accuracy={result.winner.holdout_accuracy:.4f}")
    else:
        print(f"No accepted baseline found in {MAX_REVISION_ROUNDS} rounds")
