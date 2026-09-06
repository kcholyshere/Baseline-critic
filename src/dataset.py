"""Loads Breast Cancer Wisconsin, and generalises the held-out split to any
uploaded binary-classification CSV.

The holdout is never written to disk, for either dataset. For the built-in
demo dataset it is rebuilt in memory from the fixed seed each time the
scoring path needs it - see ADR-005. For an uploaded dataset there is no
reusable public loader to regenerate from, so the same principle is applied
directly instead: the full uploaded file is simply never persisted, and only
the train split is written to disk. Either way, there is no holdout file for
agent-generated code to read, glob, or join against.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR

SEED = 42
HOLDOUT_FRACTION = 0.2
TARGET_COLUMN = "diagnosis"
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
UPLOADS_DIR = PROCESSED_DATA_DIR / "uploads"
MIN_ROWS_PER_CLASS = 2


def _load_source() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df[TARGET_COLUMN] = data.target_names[data.target]
    return df.drop(columns=["target"])


def _split_indices(df: pd.DataFrame, target_column: str) -> tuple[pd.Index, pd.Index]:
    train_idx, holdout_idx = train_test_split(
        df.index,
        test_size=HOLDOUT_FRACTION,
        random_state=SEED,
        stratify=df[target_column],
    )

    all_classes = set(df[target_column].dropna().unique())
    train_classes = set(df.loc[train_idx, target_column].dropna().unique())
    holdout_classes = set(df.loc[holdout_idx, target_column].dropna().unique())

    missing_from_train = all_classes - train_classes
    missing_from_holdout = all_classes - holdout_classes
    if missing_from_train or missing_from_holdout:
        problems = []
        if missing_from_train:
            values = ", ".join(f'"{v}"' for v in sorted(missing_from_train, key=str))
            problems.append(f"missing from the train split: {values}")
        if missing_from_holdout:
            values = ", ".join(f'"{v}"' for v in sorted(missing_from_holdout, key=str))
            problems.append(f"missing from the holdout split: {values}")
        raise ValueError(
            f'"{target_column}" has a class too small for a reliable train/holdout split - '
            + "; ".join(problems)
            + ". Add more rows for the affected class, or use a larger holdout fraction."
        )

    return train_idx, holdout_idx


def build_train_artifact(path=TRAIN_PATH) -> None:
    """Write the agent-facing train split to disk. Run once at setup, never from agent code."""
    df = _load_source()
    train_idx, _ = _split_indices(df, TARGET_COLUMN)
    train_df = df.loc[train_idx].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(path, index=False)


def get_holdout() -> pd.DataFrame:
    """Rebuild the holdout split in memory. Scoring-side only - never call from agent-generated code."""
    df = _load_source()
    _, holdout_idx = _split_indices(df, TARGET_COLUMN)
    return df.loc[holdout_idx].reset_index(drop=True)


def validate_binary_target(df: pd.DataFrame, target_column: str) -> tuple[str, str]:
    """Checks a target column is usable for a binary baseline; returns its two class values.

    Raises ValueError with a message fit to show directly in the UI, rather
    than letting a downstream sklearn error (e.g. from a stratified split
    with too few members of a class) surface as a stack trace.
    """
    counts = df[target_column].dropna().value_counts()
    if len(counts) != 2:
        raise ValueError(
            f'"{target_column}" has {len(counts)} distinct values; pick a column with exactly '
            "two values - this version only supports binary classification."
        )
    if counts.min() < MIN_ROWS_PER_CLASS:
        sparse_class = counts.idxmin()
        raise ValueError(
            f'"{target_column}" has only {counts.min()} row(s) with value "{sparse_class}"; '
            f"each class needs at least {MIN_ROWS_PER_CLASS} rows for a train/holdout split."
        )
    return tuple(counts.index.astype(str))


@dataclass
class UploadedDataset:
    train_path: Path
    holdout: pd.DataFrame
    target_column: str
    dataset_name: str


def prepare_uploaded_dataset(df: pd.DataFrame, target_column: str, dataset_name: str) -> UploadedDataset:
    """Splits an uploaded dataset and writes only the train rows to disk.

    The holdout is returned in memory and never persisted - see this
    module's docstring. Call validate_binary_target first; this function
    assumes the target column is already known-good.
    """
    train_idx, holdout_idx = _split_indices(df, target_column)
    train_df = df.loc[train_idx].reset_index(drop=True)
    holdout_df = df.loc[holdout_idx].reset_index(drop=True)

    upload_id = uuid.uuid4().hex[:12]
    train_path = UPLOADS_DIR / upload_id / "train.csv"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)

    return UploadedDataset(
        train_path=train_path,
        holdout=holdout_df,
        target_column=target_column,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    build_train_artifact()
