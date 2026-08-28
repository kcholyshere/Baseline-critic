"""Deterministic dataset summary - the proposal's "Profiling tool".

A plain pandas summary, not an LLM call: the proposal's own component table
lists this as a tool alongside code execution and scoring, separate from the
two agents (planner/modeller, critic). An LLM asked to summarise a
DataFrame can hallucinate counts; pandas cannot.
"""

import pandas as pd


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
    target_summary = {
        "dtype": str(target_series.dtype),
        "value_counts": {str(k): int(v) for k, v in target_series.value_counts(dropna=True).items()},
        "missing_count": int(target_series.isna().sum()),
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
        "Feature columns (dtype, cardinality, missing, zeros, likely_identifier):",
    ]
    for name, stats in profile["features"].items():
        flag = " [LIKELY IDENTIFIER - consider excluding]" if stats["likely_identifier"] else ""
        lines.append(
            f"  {name}: {stats['dtype']}, cardinality={stats['cardinality']}, "
            f"missing={stats['missing_count']} ({stats['missing_pct']}%), "
            f"zeros={stats['zero_count']}{flag}"
        )
    return "\n".join(lines)
