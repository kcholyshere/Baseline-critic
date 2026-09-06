"""Phase 5 evaluation harness: plants one known defect per defect_category
into the Breast Cancer Wisconsin dataset (ADR-005's seeded split) and
measures the critic's detection rate on each, against its false-alarm rate
on a clean run (references/project-proposal.md, "How it is evaluated").

Design choice, agreed with Krzysztof before building: defects are
hand-written training scripts, run once for real through the same sandbox
and scoring path a live agent run uses (src.agent.build_run_report_from_execution),
not requested from the local modeller LLM. ADR-008 already shows that model
isn't reliable enough to produce a specific bug type on demand - asking it to
would make the modeller's unreliability, not the critic's judgement, the
thing being measured. This keeps every fixture deterministic and re-runnable,
the same principle ADR-005 used for the held-out split itself.

Each fixture is scored by the critic N times (LLM sampling is not
deterministic), reporting three numbers per category, not one:
- static_findings_fired: whether src.checks caught it with no LLM call at all.
- llm_only_reject_rate: the reject rate among trials where the LLM actually
  answered (Critique.fallback_used is False) - the number that answers
  "is the critic itself doing anything", not the static checks alongside it.
- combined_reject_rate: what a user actually sees (static checks + LLM,
  falling back to static-only when the LLM's JSON never parses).
Two categories - target_leakage and temporal_leakage - have no static check
at all (src/checks.py targets neither), so their combined and llm_only rates
are identical: they are the harness's only real test of the LLM critic in
isolation, not of the deterministic checks sitting beside it.

temporal_leakage is a synthetic scenario: neither dataset this project uses
has a real time axis. A "record_date" column and a target-rate encoding
computed over the *entire* dataset (train and holdout) before the split
stand in for it - a realistic case of a feature-engineering step leaking
future information backward, agreed with Krzysztof as the closest honest
proxy rather than declaring the category untestable.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from google.adk.models.lite_llm import LiteLlm

from src import config, dataset
from src.agent import build_run_report_from_execution
from src.checks import run_static_checks
from src.critic import Critique, critique_run_async
from src.profiling import format_profile_for_prompt, profile_dataframe
from src.report import RunReport, _atomic_write_json, _read_json_or_none
from src.services.code_execution import run_code

FIXTURES_DIR = config.PROCESSED_DATA_DIR / "eval_fixtures"
SUMMARIES_DIR = config.PROCESSED_DATA_DIR / "eval_reports"
LATEST_SUMMARY_PATH = SUMMARIES_DIR / "latest.json"

DEFAULT_TRIALS_PER_FIXTURE = 5
POSITIVE_CLASS = "malignant"
NEGATIVE_CLASS = "benign"

_TRAIN_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = (df[{target_column!r}] == {positive_class!r}).astype(int)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = (booster.predict(X_val) >= 0.5).astype(int)
accuracy = (preds == y_val).mean()
print(f"Validation accuracy: {{accuracy:.4f}}")
booster.save_model("model.txt")
'''

_TRAIN_TEST_CONTAMINATION_SCRIPT = '''\
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df[{target_column!r}] = data.target_names[data.target]
y = (df[{target_column!r}] == {positive_class!r}).astype(int)
X = df.drop(columns=[{target_column!r}, "target"], errors="ignore")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = (booster.predict(X_val) >= 0.5).astype(int)
accuracy = (preds == y_val).mean()
print(f"Validation accuracy: {{accuracy:.4f}}")
booster.save_model("model.txt")
'''

_UNSEEDED_RANDOMNESS_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = (df[{target_column!r}] == {positive_class!r}).astype(int)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "binary", "metric": "binary_logloss", "verbosity": -1}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = (booster.predict(X_val) >= 0.5).astype(int)
accuracy = (preds == y_val).mean()
print(f"Validation accuracy: {{accuracy:.4f}}")
booster.save_model("model.txt")
'''

_DEGENERATE_SPLIT_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = (df[{target_column!r}] == {positive_class!r}).astype(int)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = (booster.predict(X_val) >= 0.5).astype(int)
accuracy = (preds == y_val).mean()
print(f"Validation accuracy: {{accuracy:.4f}}")
booster.save_model("model.txt")
'''

_SCORE_MISMATCH_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = (df[{target_column!r}] == {positive_class!r}).astype(int)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = (booster.predict(X_train) >= 0.5).astype(int)
accuracy = (preds == y_train).mean()
print(f"Validation accuracy: {{accuracy:.4f}}")
booster.save_model("model.txt")
'''


def _inject_target_leakage(df: pd.DataFrame, target_column: str, positive_class: str) -> pd.DataFrame:
    """A column that is (almost) a direct copy of the label - the classic
    "the answer got left in a feature" bug. No src/checks.py check targets
    this; only the critic can catch it."""
    df = df.copy()
    label = (df[target_column] == positive_class).astype(float)
    df["diagnosis_score"] = label + np.random.RandomState(0).normal(0, 0.01, len(df))
    return df


def _inject_temporal_leakage(df: pd.DataFrame, target_column: str, positive_class: str) -> pd.DataFrame:
    """A target-rate encoding computed over the whole dataset - train and
    holdout together - before the split, plus a synthetic date column
    standing in for the time axis neither real dataset has (module
    docstring). Also uncatchable by src/checks.py."""
    df = df.copy().reset_index(drop=True)
    label = (df[target_column] == positive_class).astype(int)
    bucket = pd.qcut(df["worst area"], 4, duplicates="drop")
    df["area_bucket_target_rate"] = label.groupby(bucket).transform("mean")
    df["record_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    return df


@dataclass
class FixtureSpec:
    category: str
    ground_truth_verdict: str  # "accept" or "reject"
    description: str
    script_template: str
    transform: object | None = None  # (df, target_column, positive_class) -> df, applied before the split


FIXTURES: list[FixtureSpec] = [
    FixtureSpec("none", "accept", "Clean baseline: seeded, stratified, no leakage.", _TRAIN_SCRIPT),
    FixtureSpec(
        "target_leakage",
        "reject",
        "A feature is (almost) a copy of the label, baked into the data before training.",
        _TRAIN_SCRIPT,
        _inject_target_leakage,
    ),
    FixtureSpec(
        "temporal_leakage",
        "reject",
        "A target-rate encoding computed over the full timeline, including rows that come after "
        "the point being predicted.",
        _TRAIN_SCRIPT,
        _inject_temporal_leakage,
    ),
    FixtureSpec(
        "train_test_contamination",
        "reject",
        "Script ignores the given train CSV and trains on the full public dataset, including the "
        "held-out rows.",
        _TRAIN_TEST_CONTAMINATION_SCRIPT,
    ),
    FixtureSpec(
        "unseeded_randomness",
        "reject",
        "No seed or random_state anywhere in the script.",
        _UNSEEDED_RANDOMNESS_SCRIPT,
    ),
    FixtureSpec(
        "degenerate_split",
        "reject",
        "train_test_split() called without stratify=.",
        _DEGENERATE_SPLIT_SCRIPT,
    ),
    FixtureSpec(
        "score_mismatch",
        "reject",
        "Script scores itself on the training rows it just fit on, then prints that number "
        "labelled 'Validation accuracy' - a genuine, code-visible gap between what the code "
        "claims to measure and what it actually measures. Deliberately independent of "
        "holdout_accuracy: an earlier version of this fixture only reordered feature columns "
        "(the ADR-009 pattern, already defended against by src/agent.py's reindexing fix), which "
        "left no live scoring error to detect and left the fixture numerically identical to the "
        "clean fixture on the one signal the critic actually used - see agent_docs/decisions.md.",
        _SCORE_MISMATCH_SCRIPT,
    ),
]


def fixture_train_path(category: str) -> Path:
    # Neither the category name nor any substring of it may appear in this
    # path: it gets embedded verbatim in the generated script via
    # pd.read_csv(...), and src/checks.py's checks are naive substring
    # searches over that script's text. "unseeded_randomness" contains the
    # literal substring "seed" - even tucked inside a subdirectory name, it
    # defeated _check_missing_seed. A short hash of the category, unrelated
    # to its spelling, removes the whole class of collision, while staying
    # a stable, deterministic path for caching.
    digest = hashlib.sha256(category.encode()).hexdigest()[:12]
    return FIXTURES_DIR / digest / "data.csv"


def _fixture_cache_path(category: str) -> Path:
    return FIXTURES_DIR / f"{category}.json"


def _standard_train_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset.build_train_artifact()
    return pd.read_csv(dataset.TRAIN_PATH), dataset.get_holdout()


def get_fixture_train_holdout(spec: FixtureSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cheap, deterministic, no LLM/sandbox call - safe to call every time
    metrics are computed, even when the fixture's scored RunReport is
    loaded from cache."""
    if spec.transform is None:
        return _standard_train_holdout()
    full_df = dataset._load_source()
    transformed = spec.transform(full_df, dataset.TARGET_COLUMN, POSITIVE_CLASS)
    train_idx, holdout_idx = dataset._split_indices(transformed, dataset.TARGET_COLUMN)
    train_df = transformed.loc[train_idx].reset_index(drop=True)
    holdout_df = transformed.loc[holdout_idx].reset_index(drop=True)
    return train_df, holdout_df


def _build_fixture_report(spec: FixtureSpec) -> RunReport:
    """Runs the fixture's script through the real sandbox exactly once."""
    train_df, holdout_df = get_fixture_train_holdout(spec)
    train_path = fixture_train_path(spec.category)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)

    code = spec.script_template.format(
        train_path=str(train_path), target_column=dataset.TARGET_COLUMN, positive_class=POSITIVE_CLASS
    )
    started = time.monotonic()
    execution = run_code(code)
    duration = time.monotonic() - started

    return build_run_report_from_execution(
        execution=execution,
        generated_code=code,
        run_id=f"fixture-{spec.category}",
        dataset_name="breast_cancer_wisconsin_fixture",
        target_column=dataset.TARGET_COLUMN,
        train_rows=len(train_df),
        holdout=holdout_df,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
        duration_seconds=duration,
        model_name="fixture-script (hand-written, no LLM)",
        attempts=1,
        agent_summary=f"Hand-written fixture script for defect_category={spec.category!r}.",
    )


def load_or_build_fixture_report(spec: FixtureSpec, rebuild: bool = False) -> RunReport:
    """Caches the expensive part (one real sandbox run) to disk, keyed by
    category - repeat evaluation runs re-score the same cached result
    against the critic rather than re-executing it, so they don't compete
    with a live demo for src/services/code_execution.py's session budget
    (ADR-003, MAX_CALLS_PER_SESSION)."""
    cache_path = _fixture_cache_path(spec.category)
    if not rebuild and cache_path.exists():
        return RunReport(**json.loads(cache_path.read_text())["report"])
    report = _build_fixture_report(spec)
    cache_path.write_text(
        json.dumps(
            {"ground_truth_verdict": spec.ground_truth_verdict, "description": spec.description, "report": report.to_dict()},
            indent=2,
        )
    )
    return report


@dataclass
class FixtureOutcome:
    category: str
    ground_truth_verdict: str
    description: str
    holdout_accuracy: float
    static_findings: list[str]
    trials: list[Critique] = field(default_factory=list)

    @property
    def combined_reject_rate(self) -> float:
        return sum(1 for t in self.trials if t.verdict == "reject") / len(self.trials)

    @property
    def llm_only_trials(self) -> list[Critique]:
        return [t for t in self.trials if not t.fallback_used]

    @property
    def llm_only_reject_rate(self) -> float | None:
        llm_trials = self.llm_only_trials
        if not llm_trials:
            return None
        return sum(1 for t in llm_trials if t.verdict == "reject") / len(llm_trials)

    @property
    def fallback_count(self) -> int:
        return sum(1 for t in self.trials if t.fallback_used)


async def _run_fixture_trials(
    report: RunReport, spec: FixtureSpec, trials_per_fixture: int, model: str | LiteLlm
) -> FixtureOutcome:
    train_df, _ = get_fixture_train_holdout(spec)
    train_path = fixture_train_path(spec.category)
    static_findings = run_static_checks(
        report.generated_code, train_path, dataset.TARGET_COLUMN, report.stdout, report.holdout_accuracy
    )
    profile_text = format_profile_for_prompt(profile_dataframe(train_df, dataset.TARGET_COLUMN))

    trials = [
        await critique_run_async(report, profile_text, static_findings, model) for _ in range(trials_per_fixture)
    ]
    return FixtureOutcome(
        category=spec.category,
        ground_truth_verdict=spec.ground_truth_verdict,
        description=spec.description,
        holdout_accuracy=report.holdout_accuracy,
        static_findings=static_findings,
        trials=trials,
    )


def _build_summary(outcomes: list[FixtureOutcome], trials_per_fixture: int, model_name: str) -> dict:
    rows = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for outcome in outcomes:
        prompt_tokens = sum(t.prompt_tokens for t in outcome.trials)
        completion_tokens = sum(t.completion_tokens for t in outcome.trials)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        rows.append(
            {
                "defect_category": outcome.category,
                "description": outcome.description,
                "ground_truth_verdict": outcome.ground_truth_verdict,
                "holdout_accuracy": outcome.holdout_accuracy,
                "static_findings": outcome.static_findings,
                "n_trials": len(outcome.trials),
                "combined_reject_rate": outcome.combined_reject_rate,
                "llm_only_reject_rate": outcome.llm_only_reject_rate,
                "fallback_count": outcome.fallback_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                # What the critic actually said on each rejecting trial - without
                # this, a false alarm and a real catch look identical in the
                # aggregate rate alone. Needed to tell "critic is trigger-happy"
                # apart from "critic is suspicious of a genuinely easy dataset"
                # (the proposal's own named hard problem).
                "rejecting_trials": [
                    {"defect_category": t.defect_category, "defect": t.defect}
                    for t in outcome.trials
                    if t.verdict == "reject"
                ],
            }
        )

    reject_rows = [r for r in rows if r["ground_truth_verdict"] == "reject"]
    clean_row = next(r for r in rows if r["ground_truth_verdict"] == "accept")
    detection_rate = sum(r["combined_reject_rate"] for r in reject_rows) / len(reject_rows) if reject_rows else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trials_per_fixture": trials_per_fixture,
        "model": model_name,
        "overall_detection_rate": detection_rate,
        "overall_false_alarm_rate": clean_row["combined_reject_rate"],
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "categories": rows,
    }


def _save_summary(summary: dict) -> Path:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = summary["generated_at"].replace(":", "-")
    timestamped_path = SUMMARIES_DIR / f"{stamp}.json"
    _atomic_write_json(timestamped_path, summary)
    _atomic_write_json(LATEST_SUMMARY_PATH, summary)
    return timestamped_path


def load_latest_summary() -> dict | None:
    """Returns the latest evaluation summary, or None if no evaluation has
    run yet (or the latest summary file on disk fails to parse)."""
    if not LATEST_SUMMARY_PATH.exists():
        return None
    return _read_json_or_none(LATEST_SUMMARY_PATH)


async def run_evaluation_async(
    trials_per_fixture: int = DEFAULT_TRIALS_PER_FIXTURE,
    rebuild: bool = False,
    model: str | LiteLlm | None = None,
) -> dict:
    model = model if model is not None else LiteLlm(model=config.DEFAULT_MODEL_URI)
    model_name = model if isinstance(model, str) else model.model

    outcomes = []
    for spec in FIXTURES:
        report = load_or_build_fixture_report(spec, rebuild=rebuild)
        outcome = await _run_fixture_trials(report, spec, trials_per_fixture, model)
        outcomes.append(outcome)

    summary = _build_summary(outcomes, trials_per_fixture, model_name)
    _save_summary(summary)
    return summary


def run_evaluation(trials_per_fixture: int = DEFAULT_TRIALS_PER_FIXTURE, rebuild: bool = False) -> dict:
    return asyncio.run(run_evaluation_async(trials_per_fixture, rebuild))
