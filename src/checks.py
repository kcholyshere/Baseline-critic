"""Deterministic static checks on the modeller agent's generated training
script - no LLM call, sitting beside profiling.py as another "tool the
model never has to guess at" and the critic never has to notice by eye.

These are heuristics over the script's text, not a proof of correctness -
each one targets a specific, real pattern this project has actually
observed in generated code (see references/local-model-benchmarks.md and
data/processed/runs/*.json), not a hypothetical. Keep it dumb and simple
first, the same principle the sandbox itself was built on (ADR-003).
"""

import re
from pathlib import Path

VAL_HOLDOUT_GAP_THRESHOLD = 0.08

_PUBLIC_DATASET_PATTERN = re.compile(r"\b(fetch_openml|load_breast_cancer|load_iris|load_diabetes|load_wine)\b|from\s+sklearn\.datasets\s+import")
_FILE_READ_PATTERN = re.compile(r"(?:read_csv|read_json|read_parquet)\(\s*['\"]([^'\"]+)['\"]")
# Ordering-dependent heuristic: assumes the reported accuracy is always the
# LAST decimal on the line, i.e. any threshold value (e.g. "0.5") is printed
# before it. True for every real output observed so far, but not guaranteed
# by the modeller's instruction - a model that ever prints accuracy before
# a threshold on the same line would silently misfire this extraction.
_DECIMAL_PATTERN = re.compile(r"\d\.\d+")


def _check_column_order_instability(code: str) -> str | None:
    if ".columns.difference(" in code or re.search(r"\bsorted\(\s*(?:list\()?\s*\w*\.?columns", code):
        return "Feature list may be built in a non-original column order (.columns.difference()/sorted() over columns) - LightGBM's Booster.predict() matches columns by position, not name, so this can silently misalign predictions without raising an error."
    return None


def _check_missing_stratify(code: str) -> str | None:
    if "train_test_split(" in code and "stratify" not in code:
        return "train_test_split() is used without stratify= - risks a degenerate internal validation split on an imbalanced target."
    return None


def _check_missing_seed(code: str) -> str | None:
    trains_with_lightgbm = re.search(r"\b(?:lgb|lightgbm)\.train\(", code)
    if trains_with_lightgbm and "seed" not in code and "random_state" not in code:
        return "lightgbm.train() is called with no seed/random_state anywhere in the script - unseeded randomness, named explicitly in the proposal as a defect to catch."
    return None


def _check_foreign_file_path(code: str, train_path: Path) -> str | None:
    expected = str(train_path)
    for match in _FILE_READ_PATTERN.finditer(code):
        if match.group(1) != expected:
            return f"Script reads a file path other than the given training CSV ({match.group(1)!r}) - possible train/test contamination."
    return None


def _check_public_dataset_import(code: str) -> str | None:
    if _PUBLIC_DATASET_PATTERN.search(code):
        return "Script imports a public dataset loader (e.g. sklearn.datasets) - this is the exact route by which a generated script could load the full source dataset, including the withheld holdout rows (ADR-005)."
    return None


def _check_target_column_referenced(code: str, target_column: str) -> str | None:
    if target_column not in code:
        return f"The target column name '{target_column}' never appears in the generated code - it may have hardcoded a different column name instead, which would be silently wrong on any dataset other than the one it was tested against."
    return None


def _check_target_excluded_from_features(code: str, target_column: str) -> str | None:
    """Unlike every other check here, this only ever confirms, never flags -
    ADR-010's and ADR-012's live evaluation runs both reproduced the critic
    rejecting a clean script by claiming the target column wasn't excluded
    from the feature set when it plainly was (e.g. `X =
    df.drop(columns=['diagnosis'])`, still present as features in the same
    dataframe used to train). When the code matches one of the real
    exclusion patterns actually observed in generated code and fixtures
    (drop(columns=...), a "not in [...]" filter list building a feature
    list, .columns.difference(...)), state that explicitly as a fact for the
    critic to trust over its own reading. When none of these match, stay
    silent rather than raise a new suspicion - the model may have excluded
    the target some other valid way this check doesn't recognise, and a
    static check should not manufacture a false lead."""
    escaped = re.escape(target_column)
    patterns = [
        rf"drop\(\s*columns\s*=\s*\[[^\]]*{escaped}[^\]]*\]",
        rf"drop\(\s*\[[^\]]*{escaped}[^\]]*\]\s*,\s*axis\s*=\s*1",
        rf"not\s+in\s*\[[^\]]*{escaped}[^\]]*\]",
        rf"columns\.difference\(\s*\[[^\]]*{escaped}[^\]]*\]",
    ]
    if any(re.search(pattern, code) for pattern in patterns):
        return (
            f"Target column '{target_column}' IS excluded from the feature set "
            "(matched a recognised exclusion pattern in the code) - do not reject "
            "on a claim that it is present in the features without re-reading the "
            "code yourself first."
        )
    return None


def _check_val_holdout_gap(stdout: str, holdout_accuracy: float) -> str | None:
    accuracy_lines = [line for line in stdout.splitlines() if "accuracy" in line.lower()]
    if not accuracy_lines:
        return None
    # A line reporting accuracy may also mention an unrelated number first
    # (e.g. "Validation accuracy at 0.5 threshold: 0.3516") - the actual
    # reported value is reliably the last decimal on the line, not the first.
    decimals = _DECIMAL_PATTERN.findall(accuracy_lines[-1])
    if not decimals:
        return None
    val_accuracy = float(decimals[-1])
    gap = abs(val_accuracy - holdout_accuracy)
    if gap > VAL_HOLDOUT_GAP_THRESHOLD:
        return (
            f"Internal validation accuracy ({val_accuracy:.4f}) and real holdout accuracy "
            f"({holdout_accuracy:.4f}) differ by {gap:.4f}, over the {VAL_HOLDOUT_GAP_THRESHOLD} "
            "threshold - the reported validation score may not be a reliable estimate of the "
            "real result, in either direction."
        )
    return None


def run_static_checks(
    generated_code: str,
    train_path: Path,
    target_column: str,
    stdout: str,
    holdout_accuracy: float,
) -> list[str]:
    """Runs every static check and returns the findings that actually fired."""
    checks = [
        _check_column_order_instability(generated_code),
        _check_missing_stratify(generated_code),
        _check_missing_seed(generated_code),
        _check_foreign_file_path(generated_code, train_path),
        _check_public_dataset_import(generated_code),
        _check_target_column_referenced(generated_code, target_column),
        _check_target_excluded_from_features(generated_code, target_column),
        _check_val_holdout_gap(stdout, holdout_accuracy),
    ]
    return [finding for finding in checks if finding is not None]
