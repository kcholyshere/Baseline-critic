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

Everything above this point describes the classification track (Breast
Cancer Wisconsin). A second, parallel track runs the same seven
defect_category fixtures against sklearn's bundled diabetes regression
dataset, reusing this module's own machinery - FixtureSpec,
_run_fixture_trials, _build_summary - rather than a separate harness: the
whole point of the critic being generic to task_type is that its evaluation
should be too. Every function that touches a specific dataset is threaded
with the fixture's own task_type rather than assuming classification, and
fixture_train_path/_fixture_cache_path key on (task_type, category) so the
two tracks' identically-named categories (e.g. "target_leakage") never
collide on disk.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from google.adk.models.lite_llm import LiteLlm
from sklearn.datasets import load_diabetes

from src import config, dataset
from src.agent import build_run_report_from_execution
from src.checks import run_checks
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

# The regression track's target column, renamed from diabetes's bare
# "target" the same way dataset.TARGET_COLUMN names the classification
# track's - so a reader looking at a fixture's train.csv sees a column name
# that says what it is, not sklearn's generic default.
REGRESSION_TARGET_COLUMN = "disease_progression"

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

# Regression counterparts to the seven classification script templates
# above, same .format(train_path=..., target_column=...) templating pattern.
# None of these reference positive_class - a continuous target needs no
# binarisation - so a stray positive_class="" kwarg passed at format time
# (kept for a single call-site shape across both tracks) is simply unused
# rather than raising.

_REGRESSION_TRAIN_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = df[{target_column!r}].astype(float)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = booster.predict(X_val)
rmse = ((preds - y_val) ** 2).mean() ** 0.5
print(f"Validation RMSE: {{rmse:.4f}}")
booster.save_model("model.txt")
'''

_REGRESSION_TRAIN_TEST_CONTAMINATION_SCRIPT = '''\
import lightgbm as lgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
df = data.frame.rename(columns={{"target": {target_column!r}}})
y = df[{target_column!r}].astype(float)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = booster.predict(X_val)
rmse = ((preds - y_val) ** 2).mean() ** 0.5
print(f"Validation RMSE: {{rmse:.4f}}")
booster.save_model("model.txt")
'''

_REGRESSION_UNSEEDED_RANDOMNESS_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = df[{target_column!r}].astype(float)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "regression", "metric": "rmse", "verbosity": -1}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = booster.predict(X_val)
rmse = ((preds - y_val) ** 2).mean() ** 0.5
print(f"Validation RMSE: {{rmse:.4f}}")
booster.save_model("model.txt")
'''

_REGRESSION_DEGENERATE_SPLIT_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = df[{target_column!r}].astype(float)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.02, random_state=42)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = booster.predict(X_val)
rmse = ((preds - y_val) ** 2).mean() ** 0.5
print(f"Validation RMSE: {{rmse:.4f}}")
booster.save_model("model.txt")
'''

_REGRESSION_SCORE_MISMATCH_SCRIPT = '''\
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv({train_path!r})
y = df[{target_column!r}].astype(float)
X = df.drop(columns=[{target_column!r}]).select_dtypes(include="number")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
booster = lgb.train(
    {{"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42}},
    train_set,
    num_boost_round=100,
    valid_sets=[val_set],
)
preds = booster.predict(X_train)
rmse = ((preds - y_train) ** 2).mean() ** 0.5
print(f"Validation RMSE: {{rmse:.4f}}")
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


def _load_regression_source() -> pd.DataFrame:
    """Regression-track equivalent of dataset._load_source() - that one is
    classification-only (breast cancer) and frozen, so this is its own
    private helper rather than a change to src/dataset.py. sklearn's
    diabetes dataset ships bundled with scikit-learn, so - like
    dataset._load_source() - this needs no network call."""
    data = load_diabetes(as_frame=True)
    return data.frame.rename(columns={"target": REGRESSION_TARGET_COLUMN})


def _inject_regression_target_leakage(df: pd.DataFrame, target_column: str, _positive_class: str) -> pd.DataFrame:
    """Regression counterpart to _inject_target_leakage: a feature that is
    (almost) a direct copy of the continuous label - no binarisation needed,
    unlike the classification version, since the target is already
    continuous. _positive_class is unused; kept only so both tracks' inject
    functions share the same (df, target_column, positive_class) -> df
    shape FixtureSpec.transform documents."""
    df = df.copy()
    noise = np.random.RandomState(0).normal(0, df[target_column].std() * 0.1, len(df))
    df["progression_score"] = df[target_column] + noise
    return df


def _inject_regression_temporal_leakage(df: pd.DataFrame, target_column: str, _positive_class: str) -> pd.DataFrame:
    """Regression counterpart to _inject_temporal_leakage: a target-mean
    encoding (not target-*rate*, since there's no class to have a rate of)
    computed over the whole dataset - train and holdout together - before
    the split, bucketing 'bmi' the way the classification version buckets
    'worst area'. Also uncatchable by src/checks.py, for the same reason
    (ADR-010): no correlation threshold safely separates this from a
    genuinely strong predictor (see _check_feature_target_correlation's
    docstring)."""
    df = df.copy().reset_index(drop=True)
    bucket = pd.qcut(df["bmi"], 4, duplicates="drop")
    df["bmi_bucket_target_mean"] = df[target_column].groupby(bucket).transform("mean")
    df["record_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    return df


@dataclass
class FixtureSpec:
    category: str
    ground_truth_verdict: str  # "accept" or "reject"
    description: str
    script_template: str
    transform: object | None = None  # (df, target_column, positive_class) -> df, applied before the split
    task_type: str = "classification"


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
    # The regression track: same seven defect_category fixtures, run against
    # sklearn's bundled diabetes dataset instead of Breast Cancer Wisconsin.
    # Descriptions are reworded only where the classification phrasing
    # doesn't carry over (e.g. "stratified"/"target-rate" assume a class
    # label) - the underlying defect each fixture stands for is identical.
    FixtureSpec(
        "none",
        "accept",
        "Clean baseline: seeded, plain random split, no leakage.",
        _REGRESSION_TRAIN_SCRIPT,
        task_type="regression",
    ),
    FixtureSpec(
        "target_leakage",
        "reject",
        "A feature is (almost) a copy of the continuous label, baked into the data before training.",
        _REGRESSION_TRAIN_SCRIPT,
        _inject_regression_target_leakage,
        task_type="regression",
    ),
    FixtureSpec(
        "temporal_leakage",
        "reject",
        "A target-mean encoding computed over the full timeline, including rows that come after "
        "the point being predicted.",
        _REGRESSION_TRAIN_SCRIPT,
        _inject_regression_temporal_leakage,
        task_type="regression",
    ),
    FixtureSpec(
        "train_test_contamination",
        "reject",
        "Script ignores the given train CSV and trains on the full public dataset, including the "
        "held-out rows.",
        _REGRESSION_TRAIN_TEST_CONTAMINATION_SCRIPT,
        task_type="regression",
    ),
    FixtureSpec(
        "unseeded_randomness",
        "reject",
        "No seed or random_state anywhere in the script.",
        _REGRESSION_UNSEEDED_RANDOMNESS_SCRIPT,
        task_type="regression",
    ),
    FixtureSpec(
        "degenerate_split",
        "reject",
        "train_test_split() called with a tiny test_size (0.02), too few rows to reliably "
        "estimate generalisation error.",
        _REGRESSION_DEGENERATE_SPLIT_SCRIPT,
        task_type="regression",
    ),
    FixtureSpec(
        "score_mismatch",
        "reject",
        "Script scores itself on the training rows it just fit on, then prints that number "
        "labelled 'Validation RMSE' - a genuine, code-visible gap between what the code claims to "
        "measure and what it actually measures. Same structural defect as the classification "
        "track's score_mismatch fixture, caught by the same task-type-agnostic "
        "_check_scored_on_training_rows.",
        _REGRESSION_SCORE_MISMATCH_SCRIPT,
        task_type="regression",
    ),
]


def fixture_train_path(task_type: str, category: str) -> Path:
    # Neither the category name nor any substring of it may appear in this
    # path: it gets embedded verbatim in the generated script via
    # pd.read_csv(...), and src/checks.py's checks are naive substring
    # searches over that script's text. "unseeded_randomness" contains the
    # literal substring "seed" - even tucked inside a subdirectory name, it
    # defeated _check_missing_seed. A short hash, unrelated to the spelling
    # of either input, removes the whole class of collision, while staying a
    # stable, deterministic path for caching. task_type is part of the hash
    # input - not just category - because the classification and regression
    # tracks share category names (e.g. both have "target_leakage"); without
    # it the two tracks would silently overwrite each other's cached CSV.
    digest = hashlib.sha256(f"{task_type}:{category}".encode()).hexdigest()[:12]
    return FIXTURES_DIR / digest / "data.csv"


def _fixture_cache_path(task_type: str, category: str) -> Path:
    # Same task_type/category collision as fixture_train_path above, but for
    # the scored RunReport cache rather than the training CSV.
    return FIXTURES_DIR / f"{task_type}_{category}.json"


def _standard_train_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset.build_train_artifact()
    return pd.read_csv(dataset.TRAIN_PATH), dataset.get_holdout()


def get_fixture_train_holdout(spec: FixtureSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cheap, deterministic, no LLM/sandbox call - safe to call every time
    metrics are computed, even when the fixture's scored RunReport is
    loaded from cache."""
    if spec.task_type == "regression":
        full_df = _load_regression_source()
        if spec.transform is not None:
            full_df = spec.transform(full_df, REGRESSION_TARGET_COLUMN, "")
        train_idx, holdout_idx = dataset._split_indices(full_df, REGRESSION_TARGET_COLUMN, task_type="regression")
        train_df = full_df.loc[train_idx].reset_index(drop=True)
        holdout_df = full_df.loc[holdout_idx].reset_index(drop=True)
        return train_df, holdout_df
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
    train_path = fixture_train_path(spec.task_type, spec.category)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)

    if spec.task_type == "regression":
        target_column = REGRESSION_TARGET_COLUMN
        positive_class = negative_class = ""
        dataset_name = "diabetes_regression_fixture"
    else:
        target_column = dataset.TARGET_COLUMN
        positive_class, negative_class = POSITIVE_CLASS, NEGATIVE_CLASS
        dataset_name = "breast_cancer_wisconsin_fixture"

    code = spec.script_template.format(
        train_path=str(train_path), target_column=target_column, positive_class=positive_class
    )
    started = time.monotonic()
    execution = run_code(code)
    duration = time.monotonic() - started

    return build_run_report_from_execution(
        execution=execution,
        generated_code=code,
        # task_type is part of run_id, not just category, for the same
        # collision reason as fixture_train_path/_fixture_cache_path above.
        run_id=f"fixture-{spec.task_type}-{spec.category}",
        dataset_name=dataset_name,
        target_column=target_column,
        train_rows=len(train_df),
        holdout=holdout_df,
        positive_class=positive_class,
        negative_class=negative_class,
        duration_seconds=duration,
        model_name="fixture-script (hand-written, no LLM)",
        attempts=1,
        agent_summary=f"Hand-written fixture script for defect_category={spec.category!r}.",
        task_type=spec.task_type,
    )


def load_or_build_fixture_report(spec: FixtureSpec, rebuild: bool = False) -> RunReport:
    """Caches the expensive part (one real sandbox run) to disk, keyed by
    (task_type, category) - repeat evaluation runs re-score the same cached
    result against the critic rather than re-executing it, so they don't
    compete with a live demo for src/services/code_execution.py's session
    budget (ADR-003, MAX_CALLS_PER_SESSION)."""
    cache_path = _fixture_cache_path(spec.task_type, spec.category)
    if not rebuild and cache_path.exists():
        cached = _read_json_or_none(cache_path)
        if cached is not None:
            return RunReport(**cached["report"])
        # An interrupted --rebuild can leave a truncated cache file. Rather
        # than crash every later run with an uncaught JSONDecodeError, treat
        # this the same as a cache miss and rebuild.
    report = _build_fixture_report(spec)
    _atomic_write_json(
        cache_path,
        {
            "ground_truth_verdict": spec.ground_truth_verdict,
            "description": spec.description,
            "report": report.to_dict(),
        },
    )
    return report


@dataclass
class FixtureOutcome:
    category: str
    ground_truth_verdict: str
    description: str
    holdout_accuracy: float
    static_findings: list[str]
    # "classification" or "regression" - the two tracks share category
    # names (e.g. both have "target_leakage"), so a reader needs this to
    # tell which track a given row belongs to (_build_summary).
    task_type: str
    # The defect_category values a static check actually found real evidence
    # for (src.checks.StaticFinding.category) - distinct from static_findings
    # being non-empty, which also includes confirmation-only text with no
    # category (ADR-016's evidence_categories() concept, kept here as data
    # rather than recomputed by every reader).
    static_evidence_categories: set[str] = field(default_factory=set)
    trials: list[Critique] = field(default_factory=list)

    @property
    def combined_reject_rate(self) -> float | None:
        if not self.trials:
            return None
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

    @property
    def gated_reject_count(self) -> int:
        """How many rounds, across all trials, the reject-gate (ADR-016)
        discarded an unevidenced reject and forced a retry - distinct from
        fallback_count, which also counts rounds that failed to parse for
        unrelated reasons (bad JSON, self-inconsistent verdict)."""
        return sum(t.gated_rejects for t in self.trials)


async def _run_fixture_trials(
    report: RunReport, spec: FixtureSpec, trials_per_fixture: int, model: str | LiteLlm
) -> FixtureOutcome:
    train_df, _ = get_fixture_train_holdout(spec)
    train_path = fixture_train_path(spec.task_type, spec.category)
    target_column = REGRESSION_TARGET_COLUMN if spec.task_type == "regression" else dataset.TARGET_COLUMN
    profile = profile_dataframe(train_df, target_column, task_type=spec.task_type)
    # No imbalance concept for a continuous target - _check_missing_imbalance_correction
    # is itself silent for anything other than a real imbalance flag, so this
    # just supplies the "no" answer rather than reading a key the regression
    # profile's target summary doesn't have.
    is_imbalanced = False if spec.task_type == "regression" else profile["target"]["is_imbalanced"]
    static_findings = run_checks(
        report.generated_code,
        train_path,
        target_column,
        report.stdout,
        report.holdout_accuracy,
        is_imbalanced,
        task_type=spec.task_type,
    )
    profile_text = format_profile_for_prompt(profile)

    trials = [
        await critique_run_async(report, profile_text, static_findings, model)
        for _ in range(trials_per_fixture)
    ]
    return FixtureOutcome(
        category=spec.category,
        ground_truth_verdict=spec.ground_truth_verdict,
        description=spec.description,
        holdout_accuracy=report.holdout_accuracy,
        static_findings=[f.text for f in static_findings],
        task_type=spec.task_type,
        static_evidence_categories={f.category for f in static_findings if f.category is not None},
        trials=trials,
    )


def _build_summary(
    outcomes: list[FixtureOutcome],
    trials_per_fixture: int,
    model_name: str,
    failed_fixtures: list[dict] | None = None,
) -> dict:
    """Builds the JSON-serialisable evaluation summary from every fixture
    outcome across both tracks.

    Simplification, deliberately out of scope for this pass: overall_detection_rate
    and overall_false_alarm_rate are computed across ALL fixtures combined -
    classification and regression together, not reported separately per
    task_type. Each row does carry its own "task_type" so a reader can still
    split the categories list by hand if they want the two tracks' numbers
    apart.
    """
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
                "task_type": outcome.task_type,
                "description": outcome.description,
                "ground_truth_verdict": outcome.ground_truth_verdict,
                "holdout_accuracy": outcome.holdout_accuracy,
                "static_findings": outcome.static_findings,
                # Which defect_category values a static check found real
                # evidence for - distinct from static_findings being
                # non-empty, which also includes confirmation-only text with
                # no category (ADR-016's evidence_categories() concept). This
                # is what "did a static check catch this" should mean.
                "static_evidence_categories": sorted(outcome.static_evidence_categories),
                "n_trials": len(outcome.trials),
                "combined_reject_rate": outcome.combined_reject_rate,
                "llm_only_reject_rate": outcome.llm_only_reject_rate,
                "fallback_count": outcome.fallback_count,
                "gated_reject_count": outcome.gated_reject_count,
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
    # Two tracks now each contribute one "accept" (clean) row - "none" for
    # classification and "none" for regression - where there used to be
    # exactly one, so this can no longer just take the first match: doing
    # that would silently drop the regression track's false-alarm signal
    # entirely. Averaged the same way detection_rate averages across
    # multiple reject rows below, for the same combined-across-both-tracks
    # reason (see this function's docstring).
    clean_rows = [r for r in rows if r["ground_truth_verdict"] == "accept"]
    # combined_reject_rate is None only when a fixture's trial list is empty
    # (see FixtureOutcome.combined_reject_rate) - excluded here rather than
    # summed, so one such row can't turn the whole detection rate into a
    # TypeError.
    reject_rates = [r["combined_reject_rate"] for r in reject_rows if r["combined_reject_rate"] is not None]
    detection_rate = sum(reject_rates) / len(reject_rates) if reject_rates else None
    false_alarm_rates = [r["combined_reject_rate"] for r in clean_rows if r["combined_reject_rate"] is not None]
    false_alarm_rate = sum(false_alarm_rates) / len(false_alarm_rates) if false_alarm_rates else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trials_per_fixture": trials_per_fixture,
        "model": model_name,
        "overall_detection_rate": detection_rate,
        "overall_false_alarm_rate": false_alarm_rate,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "categories": rows,
        # Fixtures whose sandbox build raised (e.g. TrainingBudgetExceeded)
        # before any critic trial could run - kept distinct from `categories`
        # so a reader can see which categories are simply missing from this
        # run's numbers, rather than the failure being swallowed silently.
        "failed_fixtures": failed_fixtures or [],
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
    failed_fixtures = []
    for spec in FIXTURES:
        try:
            # Scoped to the build step only: this is where run_code can raise
            # TrainingBudgetExceeded (or any other sandbox failure) during
            # --rebuild. src.agent's _run_baseline_async already leaves a
            # failure RunReport on disk for the live-agent path; this fixture
            # path calls run_code directly (see _build_fixture_report), so
            # this is the only place that failure is recorded for the
            # harness. Deliberately not wrapping _run_fixture_trials below:
            # a critic/LLM failure there is a different, unrelated failure
            # mode this fix isn't meant to mask.
            report = load_or_build_fixture_report(spec, rebuild=rebuild)
        except Exception as exc:
            failed_fixtures.append(
                {
                    "defect_category": spec.category,
                    "task_type": spec.task_type,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        outcome = await _run_fixture_trials(report, spec, trials_per_fixture, model)
        outcomes.append(outcome)

    summary = _build_summary(outcomes, trials_per_fixture, model_name, failed_fixtures)
    _save_summary(summary)
    return summary


def run_evaluation(trials_per_fixture: int = DEFAULT_TRIALS_PER_FIXTURE, rebuild: bool = False) -> dict:
    return asyncio.run(run_evaluation_async(trials_per_fixture, rebuild))
