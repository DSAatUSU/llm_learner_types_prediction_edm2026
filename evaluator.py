"""Evaluation metrics for model predictions."""

import pandas as pd
import re
import os
import glob
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Canonical label set
# -----------------------------
CANONICAL_LABELS = [
    "incremental_builder",
    "plan_then_burst_implementer",
    "tinkerer_explorer",
    "debugger_centric_fixer",
    "strategy_shifter",
    "ai_guided_integrator",
    "low_validation_completer",
]

# -----------------------------
# Mapping: Actual (human-readable) -> Canonical (prediction)
# -----------------------------
TRUE_LABEL_MAP = {
    "Incremental Builder": "incremental_builder",
    "Plan-Then-Burst Implementer": "plan_then_burst_implementer",
    "Tinkerer / Explorer": "tinkerer_explorer",
    "Debugger-Centric Fixer": "debugger_centric_fixer",
    "Strategy Shifter": "strategy_shifter",
    "AI-Guided Integrator": "ai_guided_integrator",
    "Low-Validation Completer": "low_validation_completer",
}

# -----------------------------
# Test set (user_ids)
# -----------------------------
TEST_STUDENTS = ['Student_10', 'Student_8', 'Student_13', 'Student_11']

def clean_text(s: str) -> str:
    """Light cleanup only: lowercase + strip."""
    if pd.isna(s):
        return ""
    return str(s).strip()

def to_canonical_true(label: str) -> str:
    """Map ground-truth label to canonical. Returns '' if unmapped."""
    label = clean_text(label)
    return TRUE_LABEL_MAP.get(label, "")

def to_canonical_pred(label: str) -> str:
    """Normalize prediction labels to canonical set."""
    label = clean_text(label).lower()
    
    # Quick exact match
    if label in CANONICAL_LABELS:
        return label
    
    # Conservative repair attempts
    repaired = re.sub(r"[^a-z0-9_]+", "_", label)
    repaired = re.sub(r"_+", "_", repaired).strip("_")
    
    if repaired in CANONICAL_LABELS:
        return repaired
    
    # Special-case repairs
    special = {
        "aiguided_integrator": "ai_guided_integrator",
        "debuggercentric_fixer": "debugger_centric_fixer",
        "planthenburst_implementer": "plan_then_burst_implementer",
        "tinkerer__explorer": "tinkerer_explorer",
    }
    if repaired in special:
        return special[repaired]
    
    return ""  # unmapped

def evaluate_predictions(pred_csv: str = "./output/llm_profile_predictions.csv",
                         true_csv: str = "./data/final_labels.csv",
                         test_students=None,
                         model_name: str = None):
    """Evaluate model predictions against ground truth."""
    
    # Load data
    pred_df = pd.read_csv(pred_csv)
    true_df = pd.read_csv(true_csv)

    # Restrict to test students only (by user_id)
    if test_students is None:
        test_students = TEST_STUDENTS
    if test_students:
        pred_df = pred_df[pred_df["user_id"].isin(test_students)].copy()
        true_df = true_df[true_df["user_id"].isin(test_students)].copy()
        print("Filtering to test students:", test_students)
        print("Rows after test filter:", len(true_df), "(true),", len(pred_df), "(pred)")
    
    # Merge
    df = true_df.merge(
        pred_df,
        on=["user_id", "exercise_id"],
        how="inner",
        suffixes=("_true", "_pred")
    )
    
    if model_name:
        print(f"Model: {model_name}")
    print("Total matched rows:", len(df))
    
    # Map both sides to canonical labels
    df["y_true"] = df["label_true"].apply(to_canonical_true)
    df["y_pred"] = df["label_pred"].apply(to_canonical_pred)
    
    # Drop unmapped rows
    bad_true = df[df["y_true"] == ""]["label_true"].value_counts()
    bad_pred = df[df["y_pred"] == ""]["label_pred"].value_counts()
    
    if len(bad_true) > 0:
        print("\n⚠️ Unmapped TRUE labels (fix TRUE_LABEL_MAP):")
        print(bad_true)
    
    if len(bad_pred) > 0:
        print("\n⚠️ Unmapped PRED labels (fix to_canonical_pred):")
        print(bad_pred)
    
    df_mapped = df[(df["y_true"] != "") & (df["y_pred"] != "")].copy()
    print("\nRows kept after label mapping:", len(df_mapped), " / ", len(df))
    
    y_true = df_mapped["y_true"]
    y_pred = df_mapped["y_pred"]
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", labels=CANONICAL_LABELS, zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=CANONICAL_LABELS, zero_division=0)
    
    print("\n===== Overall Metrics =====")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Weighted P/R/F1 : {p_w:.4f} / {r_w:.4f} / {f1_w:.4f}")
    print(f"Macro    P/R/F1 : {p_m:.4f} / {r_m:.4f} / {f1_m:.4f}")
    
    print("\n===== Per-Class Report =====")
    print(classification_report(
        y_true, y_pred,
        labels=CANONICAL_LABELS,
        digits=4,
        zero_division=0
    ))
    
    print("\n===== Confusion Matrix =====")
    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_LABELS)
    cm_df = pd.DataFrame(cm, index=CANONICAL_LABELS, columns=CANONICAL_LABELS)
    print(cm_df)
    
    # Diagnostics
    print("\nTrue-only rows (no prediction):", len(true_df) - len(df))
    print("Pred-only rows (no true label):", len(pred_df) - len(df))
    
    return df_mapped


def _infer_model_name_from_path(pred_csv: str) -> str:
    """Infer a model name from prediction filename."""
    base = os.path.basename(pred_csv)
    if base == "llm_profile_predictions.csv":
        return "default"
    if base.startswith("llm_profile_predictions_") and base.endswith(".csv"):
        return base[len("llm_profile_predictions_"):-4]
    return os.path.splitext(base)[0]


def evaluate_all_predictions(output_dir: str = "./output",
                             true_csv: str = "./data/final_labels.csv",
                             test_students=None):
    """Evaluate all prediction CSVs in output_dir against ground truth."""
    pattern = os.path.join(output_dir, "llm_profile_predictions*.csv")
    pred_files = sorted(glob.glob(pattern))

    if not pred_files:
        print(f"No prediction files found in {output_dir}")
        return {}

    results = {}
    for pred_csv in pred_files:
        model_name = _infer_model_name_from_path(pred_csv)
        print("\n" + "=" * 80)
        print(f"Evaluating: {pred_csv}")
        print("=" * 80)
        df_mapped = evaluate_predictions(
            pred_csv=pred_csv,
            true_csv=true_csv,
            test_students=test_students,
            model_name=model_name
        )
        results[model_name] = df_mapped

    return results


if __name__ == "__main__":
    evaluate_all_predictions()
