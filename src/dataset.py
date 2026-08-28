"""Loads Breast Cancer Wisconsin and derives the held-out split.

The holdout is never written to disk. It is rebuilt in memory from the
fixed seed each time the scoring path needs it, so there is no holdout
file for agent-generated code to read, glob, or join against - see
ADR-005. Only `build_train_artifact` writes to disk, and only the train
rows.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR

SEED = 42
HOLDOUT_FRACTION = 0.2
TARGET_COLUMN = "diagnosis"
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"


def _load_source() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df[TARGET_COLUMN] = data.target_names[data.target]
    return df.drop(columns=["target"])


def _split_indices(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    return train_test_split(
        df.index,
        test_size=HOLDOUT_FRACTION,
        random_state=SEED,
        stratify=df[TARGET_COLUMN],
    )


def build_train_artifact(path=TRAIN_PATH) -> None:
    """Write the agent-facing train split to disk. Run once at setup, never from agent code."""
    df = _load_source()
    train_idx, _ = _split_indices(df)
    train_df = df.loc[train_idx].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(path, index=False)


def get_holdout() -> pd.DataFrame:
    """Rebuild the holdout split in memory. Scoring-side only - never call from agent-generated code."""
    df = _load_source()
    _, holdout_idx = _split_indices(df)
    return df.loc[holdout_idx].reset_index(drop=True)


if __name__ == "__main__":
    build_train_artifact()
