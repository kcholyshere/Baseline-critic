"""Deterministic static checks on the modeller agent's generated training
script - no LLM call, sitting beside profiling.py as another "tool the
model never has to guess at" and the critic never has to notice by eye.

Most of these are heuristics over the script's text, not a proof of
correctness - each one targets a specific, real pattern this project has
actually observed in generated code (see references/local-model-benchmarks.md
and data/processed/runs/*.json), not a hypothetical. Keep it dumb and simple
first, the same principle the sandbox itself was built on (ADR-003).

One check (_check_feature_target_correlation, ADR-014) reads the actual
training data rather than the code text - the eval harness's target_leakage
fixture uses code byte-identical to the clean fixture, only the CSV
differs, so no code-level check can ever distinguish them (ADR-010).

Each flagging check (not the confirmation-only ones) is tagged with the
defect_category it is evidence for. src/critic.py's reject-gate (ADR-016)
reads each StaticFinding's category to refuse an LLM reject verdict that
names a category none of these checks actually found - closing the general
hallucination class (a reject with no static backing at all) rather than
patching one hallucinated pattern at a time. temporal_leakage is
deliberately untagged by any check (ADR-014's threshold search failed) and
stays exempt from that gate, not silently caught by it.

Callers get the findings from one call to run_checks() and derive display
text and evidence categories from that same list themselves, rather than
calling two separate functions that would each independently re-run every
check (including the correlation check's CSV read).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VAL_HOLDOUT_GAP_THRESHOLD = 0.08
LEAKAGE_CORRELATION_THRESHOLD = 0.97

_PUBLIC_DATASET_PATTERN = re.compile(r"\b(fetch_openml|load_breast_cancer|load_iris|load_diabetes|load_wine)\b|from\s+sklearn\.datasets\s+import")
_FILE_READ_PATTERN = re.compile(r"(?:read_csv|read_json|read_parquet)\(\s*['\"]([^'\"]+)['\"]")
# Ordering-dependent heuristic: assumes the reported accuracy is always the
# LAST decimal on the line, i.e. any threshold value (e.g. "0.5") is printed
# before it. True for every real output observed so far, but not guaranteed
# by the modeller's instruction - a model that ever prints accuracy before
# a threshold on the same line would silently misfire this extraction.
_DECIMAL_PATTERN = re.compile(r"\d\.\d+")


@dataclass
class StaticFinding:
    """One static-check result. `category` is set only when the finding is
    positive evidence that a specific named defect is real - confirmation-only
    findings and checks with no matching DefectCategory (e.g.
    _check_target_column_referenced) leave it None, so they can never satisfy
    src/critic.py's reject-gate."""

    text: str
    category: str | None = None


def _check_column_order_instability(code: str) -> StaticFinding | None:
    if ".columns.difference(" in code or re.search(r"\bsorted\(\s*(?:list\()?\s*\w*\.?columns", code):
        return StaticFinding(
            "Feature list may be built in a non-original column order (.columns.difference()/sorted() over columns) - LightGBM's Booster.predict() matches columns by position, not name, so this can silently misalign predictions without raising an error.",
            category="score_mismatch",
        )
    return None


def _check_missing_stratify(code: str) -> StaticFinding | None:
    if "train_test_split(" in code and "stratify" not in code:
        return StaticFinding(
            "train_test_split() is used without stratify= - risks a degenerate internal validation split on an imbalanced target.",
            category="degenerate_split",
        )
    return None


def _check_missing_imbalance_correction(code: str, is_imbalanced: bool) -> StaticFinding | None:
    """Flags code that ignores a target imbalance the profile already told it
    about (src/profiling.py's IMBALANCE_THRESHOLD). category=None: like
    _check_target_column_referenced, this is a real flag with no matching
    DefectCategory in src/critic.py, so it stays informational context for
    the critic rather than a gated reject reason (Krzysztof scoped this to
    flag-plus-check only, not a new critic-gated category)."""
    if not is_imbalanced:
        return None
    if any(marker in code for marker in ("class_weight", "scale_pos_weight", "is_unbalance")):
        return None
    return StaticFinding(
        "The dataset profile flagged the target as imbalanced, but the script sets none of "
        "class_weight, scale_pos_weight, or is_unbalance - it appears to train on the raw class "
        "counts without correcting for the imbalance."
    )


def _check_missing_seed(code: str) -> StaticFinding | None:
    trains_with_lightgbm = re.search(r"\b(?:lgb|lightgbm)\.train\(", code)
    if trains_with_lightgbm and "seed" not in code and "random_state" not in code:
        return StaticFinding(
            "lightgbm.train() is called with no seed/random_state anywhere in the script - unseeded randomness, named explicitly in the proposal as a defect to catch.",
            category="unseeded_randomness",
        )
    return None


def _check_foreign_file_path(code: str, train_path: Path) -> StaticFinding | None:
    expected = str(train_path)
    for match in _FILE_READ_PATTERN.finditer(code):
        if match.group(1) != expected:
            return StaticFinding(
                f"Script reads a file path other than the given training CSV ({match.group(1)!r}) - possible train/test contamination.",
                category="train_test_contamination",
            )
    return None


def _check_public_dataset_import(code: str) -> StaticFinding | None:
    if _PUBLIC_DATASET_PATTERN.search(code):
        return StaticFinding(
            "Script imports a public dataset loader (e.g. sklearn.datasets) - this is the exact route by which a generated script could load the full source dataset, including the withheld holdout rows (ADR-005).",
            category="train_test_contamination",
        )
    return None


def _check_target_column_referenced(code: str, target_column: str) -> StaticFinding | None:
    if target_column not in code:
        return StaticFinding(
            f"The target column name '{target_column}' never appears in the generated code - it may have hardcoded a different column name instead, which would be silently wrong on any dataset other than the one it was tested against."
        )
    return None


def _check_target_excluded_from_features(code: str, target_column: str) -> StaticFinding | None:
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
        return StaticFinding(
            f"Target column '{target_column}' IS excluded from the feature set "
            "(matched a recognised exclusion pattern in the code) - do not reject "
            "on a claim that it is present in the features without re-reading the "
            "code yourself first."
        )
    return None


def _check_scored_on_training_rows(code: str) -> StaticFinding | None:
    """Flags a script whose final .predict() call scores the exact same
    feature variable it just built the training lgb.Dataset from, instead
    of the held-back validation split - a genuine claim-vs-reality mismatch
    (the eval harness's score_mismatch fixture, agent_docs/decisions.md),
    independent of any accuracy number.

    A magnitude-based check on the resulting accuracy gap was tried first
    and rejected: on real data this defect produces a gap of only ~0.035,
    too close to a genuinely clean run's ordinary sampling variance
    (~0.002) to set a safe threshold between them - the same reasoning that
    already ruled out a correlation threshold for temporal_leakage. This
    structural check needs no threshold at all."""
    train_match = re.search(r"\.Dataset\(\s*(\w+)\s*,\s*label\s*=\s*(\w+)\s*\)", code)
    if train_match is None:
        return None
    train_features_var = train_match.group(1)
    predict_calls = re.findall(r"\.predict\(\s*(\w+)\s*\)", code)
    if predict_calls and predict_calls[-1] == train_features_var:
        return StaticFinding(
            f"The script's final .predict() call scores '{train_features_var}' - the same "
            "features it just trained on - rather than a held-back validation split, so the "
            "reported accuracy measures fit to training data, not generalisation "
            "(defect_category: score_mismatch).",
            category="score_mismatch",
        )
    return None


def _check_val_holdout_gap(stdout: str, holdout_accuracy: float, scored_on_training_rows: bool) -> StaticFinding | None:
    """Two-sided, unlike the checks above it: a small gap is stated as a
    confirmed fact, not left silent, because the eval harness (Phase 5)
    found the critic treating any difference at all - even a ~0.002 gap
    identical to one seen on a clean run - as evidence of a defect. stdout's
    number and holdout_accuracy are never the same measurement: stdout is
    the script's own internal validation fold (a slice of the training
    data), holdout_accuracy is the harness scoring the saved model against
    rows the script never saw. A small gap between two different samples is
    normal variance, not a defect signal - only a gap past
    VAL_HOLDOUT_GAP_THRESHOLD is.

    scored_on_training_rows suppresses the "normal, don't reject" half:
    when _check_scored_on_training_rows has already found the script
    scoring itself on training rows, the gap is not ordinary sampling
    variance, and confirming it as harmless would contradict that other
    finding in the same prompt - the critic is told to trust static
    findings over its own reading, so two that disagree is worse than one
    that's silent."""
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
        return StaticFinding(
            f"Internal validation accuracy ({val_accuracy:.4f}) and real holdout accuracy "
            f"({holdout_accuracy:.4f}) differ by {gap:.4f}, over the {VAL_HOLDOUT_GAP_THRESHOLD} "
            "threshold - the reported validation score may not be a reliable estimate of the "
            "real result, in either direction.",
            category="score_mismatch",
        )
    if scored_on_training_rows:
        return None
    return StaticFinding(
        f"Internal validation accuracy ({val_accuracy:.4f}) and real holdout accuracy "
        f"({holdout_accuracy:.4f}) differ by {gap:.4f} - within the normal range for two "
        "different samples of similar size. They are not the same measurement and are not "
        "expected to match exactly; do not reject on this difference alone."
    )


def _check_feature_target_correlation(train_path: Path, target_column: str) -> StaticFinding | None:
    """Flags a feature whose correlation with the (binarised) target exceeds
    LEAKAGE_CORRELATION_THRESHOLD - the classic "the answer got left in a
    feature" pattern. Threshold picked with a wide, measured safety margin:
    on the real Breast Cancer Wisconsin data, the strongest legitimate
    predictor ("worst concave points") correlates at 0.786; the eval
    harness's injected target_leakage column ("diagnosis_score" - label
    plus small noise) correlates at 0.9998. 0.97 sits far above the former
    and comfortably below the latter, so this only catches a near-duplicate
    of the label, not a merely strong feature.

    Deliberately does not attempt to catch temporal_leakage (ADR-010's
    injected target-rate encoding correlates at only 0.830 - too close to
    the legitimate 0.786 ceiling to set a safe threshold for; a check with
    a threshold low enough to catch it would risk flagging a genuinely
    strong predictor on a different dataset). That category stays
    critic-only, by design, not by oversight."""
    try:
        df = pd.read_csv(train_path)
    except Exception:
        return None
    if target_column not in df.columns:
        return None
    values = df[target_column].dropna().unique()
    if len(values) != 2:
        return None
    label = (df[target_column] == values[0]).astype(int)
    numeric_features = df.drop(columns=[target_column]).select_dtypes(include="number")
    if numeric_features.empty:
        return None
    correlations = numeric_features.corrwith(label).abs()
    offenders = correlations[correlations > LEAKAGE_CORRELATION_THRESHOLD]
    if offenders.empty:
        return None
    worst_feature = offenders.idxmax()
    return StaticFinding(
        f"Feature '{worst_feature}' correlates with the target at {offenders[worst_feature]:.4f} "
        f"(threshold {LEAKAGE_CORRELATION_THRESHOLD}) - near-perfect correlation with the label is "
        "the classic target-leakage signature (a copy of the answer left in the features), well "
        "beyond what a legitimately predictive feature reaches on real data.",
        category="target_leakage",
    )


def _run_all_checks(
    generated_code: str,
    train_path: Path,
    target_column: str,
    stdout: str,
    holdout_accuracy: float,
    is_imbalanced: bool,
) -> list[StaticFinding]:
    """Runs every static check once and returns the findings that fired -
    shared by run_static_checks (display text) and evidence_categories
    (src/critic.py's reject-gate) so both read off the same computation."""
    scored_on_training_rows_finding = _check_scored_on_training_rows(generated_code)
    checks = [
        _check_column_order_instability(generated_code),
        _check_missing_stratify(generated_code),
        _check_missing_seed(generated_code),
        _check_missing_imbalance_correction(generated_code, is_imbalanced),
        _check_foreign_file_path(generated_code, train_path),
        _check_public_dataset_import(generated_code),
        _check_target_column_referenced(generated_code, target_column),
        _check_target_excluded_from_features(generated_code, target_column),
        _check_feature_target_correlation(train_path, target_column),
        scored_on_training_rows_finding,
        _check_val_holdout_gap(stdout, holdout_accuracy, scored_on_training_rows_finding is not None),
    ]
    return [finding for finding in checks if finding is not None]


def run_checks(
    generated_code: str,
    train_path: Path,
    target_column: str,
    stdout: str,
    holdout_accuracy: float,
    is_imbalanced: bool = False,
) -> list[StaticFinding]:
    """Runs every static check once and returns the raw findings. Callers
    that need display text, evidence categories, or both (src/critic.py's
    reject-gate, src/agent.py, src/evaluation.py) should derive them from
    this one list rather than calling separate functions that would each
    re-run every check independently."""
    return _run_all_checks(generated_code, train_path, target_column, stdout, holdout_accuracy, is_imbalanced)
