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


def profile_dataframe(df: pd.DataFrame, target_column: str) -> dict:
    """Returns a JSON-serialisable summary: per-column stats plus target balance.

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
    value_counts = target_series.value_counts(dropna=True)
    minority_class = None
    minority_fraction = None
    is_imbalanced = False
    if len(value_counts) > 0:
        minority_class = str(value_counts.idxmin())
        minority_fraction = round(float(value_counts.min() / value_counts.sum()), 4)
        is_imbalanced = minority_fraction < IMBALANCE_THRESHOLD
    target_summary = {
        "dtype": str(target_series.dtype),
        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
        "missing_count": int(target_series.isna().sum()),
        "minority_class": minority_class,
        "minority_fraction": minority_fraction,
        "is_imbalanced": is_imbalanced,
    }

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "target_column": target_column,
        "target": target_summary,
        "features": feature_columns,
    }


def format_profile_for_prompt(profile: dict) -> str:
    """Renders a profile as a compact text block for the agent's instruction."""
    lines = [
        f"Rows: {profile['row_count']}, feature columns: {len(profile['features'])}",
        f"Target '{profile['target_column']}' balance: {profile['target']['value_counts']}",
    ]
    if profile["target"]["is_imbalanced"]:
        lines.append(
            f"Target is IMBALANCED - minority class '{profile['target']['minority_class']}' is only "
            f"{profile['target']['minority_fraction'] * 100:.1f}% of rows. Correct for this "
            "(e.g. LightGBM's is_unbalance=True or scale_pos_weight, or scikit-learn's "
            "class_weight) rather than training on the raw class counts."
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
