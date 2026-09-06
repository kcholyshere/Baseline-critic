"""Deterministic dataset summary - the proposal's "Profiling tool".

A plain pandas summary, not an LLM call: the proposal's own component table
lists this as a tool alongside code execution and scoring, separate from the
two agents (planner/modeller, critic). An LLM asked to summarise a
DataFrame can hallucinate counts; pandas cannot.
"""

import pandas as pd

# A minority class below this share of rows is flagged as imbalanced -
# roughly a 65:35 split or worse. Chosen as a conventional, well-documented
# cutoff (not tuned against this project's own data) since profiling.py is a
# plain deterministic summary, not a place to encode dataset-specific
# judgement.
IMBALANCE_THRESHOLD = 0.35

# A column qualifies as a time-column candidate if at least this share of
# its non-null values parse as a date/time - high enough that an ordinary
# numeric or text feature column won't pass by chance (pd.to_datetime is
# surprisingly permissive with plain integers), while still tolerating a
# handful of genuinely malformed values in an otherwise real time column.
MIN_TIME_PARSE_RATE = 0.95


def detect_candidate_time_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    """Returns feature columns usable as a chronological-split key.

    Datetime-dtype columns qualify outright; anything else must parse as a
    date/time for at least MIN_TIME_PARSE_RATE of its non-null values via
    pd.to_datetime. Deliberately does not try to guess a bare numeric
    period id (e.g. a "week_number" column with no date strings) as a time
    column - too easy to false-positive on an ordinary numeric feature, the
    same reasoning src/checks.py already applies to correlation thresholds
    (agent_docs/decisions.md ADR-014). A constant column is excluded since
    it carries no ordering information.
    """
    candidates = []
    for column in df.columns:
        if column == target_column:
            continue
        series = df[column].dropna()
        if series.nunique() <= 1:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= MIN_TIME_PARSE_RATE:
            candidates.append(column)
    return candidates


def _classification_target_summary(target_series: pd.Series) -> dict:
    value_counts = target_series.value_counts(dropna=True)
    minority_class = None
    minority_fraction = None
    is_imbalanced = False
    if len(value_counts) > 0:
        minority_class = str(value_counts.idxmin())
        minority_fraction = round(float(value_counts.min() / value_counts.sum()), 4)
        is_imbalanced = minority_fraction < IMBALANCE_THRESHOLD
    return {
        "dtype": str(target_series.dtype),
        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
        "missing_count": int(target_series.isna().sum()),
        "minority_class": minority_class,
        "minority_fraction": minority_fraction,
        # Imbalance is a classification concept only - always False for
        # regression, which is what keeps checks._check_missing_imbalance_correction
        # silent on a regression run with no extra guard needed there.
        "is_imbalanced": is_imbalanced,
    }


def _regression_target_summary(target_series: pd.Series) -> dict:
    described = target_series.dropna().describe()
    return {
        "dtype": str(target_series.dtype),
        "missing_count": int(target_series.isna().sum()),
        "count": int(described.get("count", 0)),
        "mean": round(float(described.get("mean", float("nan"))), 4),
        "std": round(float(described.get("std", float("nan"))), 4),
        "min": round(float(described.get("min", float("nan"))), 4),
        "p25": round(float(described.get("25%", float("nan"))), 4),
        "p50": round(float(described.get("50%", float("nan"))), 4),
        "p75": round(float(described.get("75%", float("nan"))), 4),
        "max": round(float(described.get("max", float("nan"))), 4),
        # Regression has no minority-class notion - kept present and False
        # so downstream code (checks.py) can read this key regardless of
        # task_type without a separate branch.
        "is_imbalanced": False,
    }


def profile_dataframe(
    df: pd.DataFrame, target_column: str, time_column: str | None = None, task_type: str = "classification"
) -> dict:
    """Returns a JSON-serialisable summary: per-column stats plus target balance
    (classification) or target distribution (regression).

    Reports missing values and exact-zero values as separate counts. A
    column can read as fully populated by isna() while really encoding
    missingness as zero (e.g. Pima Indians Diabetes' blood-pressure and
    insulin columns) - that gap is exactly the kind of thing a naive
    "any nulls?" check misses and worth surfacing on its own.
    """
    feature_columns = {}
    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        feature_columns[column] = {
            "dtype": str(series.dtype),
            "cardinality": int(series.nunique(dropna=True)),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(100 * series.isna().mean(), 1),
            "zero_count": int((series == 0).sum()) if is_numeric else None,
            "likely_identifier": bool(series.nunique(dropna=True) == len(df)),
        }

    target_series = df[target_column]
    if task_type == "regression":
        target_summary = _regression_target_summary(target_series)
    else:
        target_summary = _classification_target_summary(target_series)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "target_column": target_column,
        "task_type": task_type,
        "target": target_summary,
        "features": feature_columns,
        "time_column": time_column,
        "candidate_time_columns": detect_candidate_time_columns(df, target_column),
    }


def format_profile_for_prompt(profile: dict) -> str:
    """Renders a profile as a compact text block for the agent's instruction."""
    lines = [f"Rows: {profile['row_count']}, feature columns: {len(profile['features'])}"]
    if profile["task_type"] == "regression":
        t = profile["target"]
        lines.append(
            f"Target '{profile['target_column']}' is continuous (regression): mean={t['mean']}, "
            f"std={t['std']}, min={t['min']}, p25={t['p25']}, p50={t['p50']}, p75={t['p75']}, max={t['max']}"
        )
    else:
        lines.append(f"Target '{profile['target_column']}' balance: {profile['target']['value_counts']}")
        if profile["target"]["is_imbalanced"]:
            lines.append(
                f"Target is IMBALANCED - minority class '{profile['target']['minority_class']}' is only "
                f"{profile['target']['minority_fraction'] * 100:.1f}% of rows. Correct for this "
                "(e.g. LightGBM's is_unbalance=True or scale_pos_weight, or scikit-learn's "
                "class_weight) rather than training on the raw class counts."
            )
    if profile["time_column"]:
        lines.append(
            f"This dataset is TIME-ORDERED by '{profile['time_column']}'. The train/holdout split "
            "is already chronological (earlier rows train, later rows holdout) - do not shuffle rows "
            "back together. If your script builds its own internal train/validation split, make it "
            "chronological too (e.g. train_test_split(..., shuffle=False), keeping the data sorted by "
            f"'{profile['time_column']}' first), not a random shuffle - and do not engineer a feature "
            "using information from later rows (e.g. a global mean/rate encoding computed over the "
            "whole file), since that leaks future information into the past."
        )
    lines.append("Feature columns (dtype, cardinality, missing, zeros, likely_identifier):")
    for name, stats in profile["features"].items():
        flag = " [LIKELY IDENTIFIER - consider excluding]" if stats["likely_identifier"] else ""
        lines.append(
            f"  {name}: {stats['dtype']}, cardinality={stats['cardinality']}, "
            f"missing={stats['missing_count']} ({stats['missing_pct']}%), "
            f"zeros={stats['zero_count']}{flag}"
        )
    return "\n".join(lines)
