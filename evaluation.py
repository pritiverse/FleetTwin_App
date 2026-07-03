"""
evaluation.py
FleetTwin (local app edition) — performs the critical three-way model assessment.

Run after train.py:
    python evaluation.py

This script evaluates the trained fault classifier on three distinct datasets:
1. The full test set (real + synthetic data).
2. The real-only portion of the test set.
3. The frozen real-data-only holdout set.

This provides the headline evidence for model performance and generalization.
"""

import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Add src to path to import feature engineering functions
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")
sys.path.insert(0, SRC)

from features import (  # noqa: E402
    add_log_gas_features,
    add_duval_ratios,
    add_iec_zone_codes,
    add_tdcg,
    add_temperature_diagnostic_features,
)

# --- Configuration ---
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
CHECKPOINTS_DIR = os.path.join(BASE, "checkpoints")
PLOTS_DIR = os.path.join(BASE, "outputs", "plots")

FAULT_NAMES = {
    "Normal": "Normal", "PD": "Partial Discharge", "D1": "Low-energy Discharge",
    "D2": "High-energy Discharge", "T1": "Low-temp Overheating",
    "T2": "Mid-temp Overheating", "T3": "High-temp Overheating",
}
GAS_COLS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


def staged_predict(X, clf1, le1, clf2a, le2a, clf2b, le2b):
    """Performs prediction using the two-stage classifier."""
    # Stage 1 prediction
    stage1_pred_idx = clf1.predict(X)
    stage1_pred_labels = le1.inverse_transform(stage1_pred_idx)

    # Initialize final predictions with Stage 1 results
    final_preds = np.copy(stage1_pred_labels)

    # Stage 2a: Discharge
    discharge_mask = final_preds == "Discharge"
    if np.any(discharge_mask):
        X_discharge = X[discharge_mask]
        s2a_preds_idx = clf2a.predict(X_discharge)
        s2a_preds_labels = le2a.inverse_transform(s2a_preds_idx)
        final_preds[discharge_mask] = s2a_preds_labels

    # Stage 2b: Overheating
    overheating_mask = final_preds == "Overheating"
    if np.any(overheating_mask):
        X_overheating = X[overheating_mask]
        s2b_preds_idx = clf2b.predict(X_overheating)
        s2b_preds_labels = le2b.inverse_transform(s2b_preds_idx)
        final_preds[overheating_mask] = s2b_preds_labels

    return final_preds


def plot_confusion_matrix(y_true, y_pred, le: LabelEncoder, title: str):
    """Generates and saves a labeled confusion matrix plot."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    labels = le.classes_
    full_names = [FAULT_NAMES.get(l, l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=full_names, columns=full_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {title}")
    plt.ylabel("Actual Fault")
    plt.xlabel("Predicted Fault")
    plt.tight_layout()

    filename = f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Saved confusion matrix -> {save_path}")
    plt.close()


def run_evaluation(X: np.ndarray, y: np.ndarray, final_le: LabelEncoder, models: dict, title: str):
    """Runs a full evaluation on a given dataset and prints/plots results."""
    print("-" * 50)
    print(f"EVALUATING: {title} ({len(X)} samples)")

    if len(X) == 0:
        print("No data to evaluate. Skipping.")
        return

    # Use the staged prediction logic
    y_pred = staged_predict(X, **models)
    y_enc = final_le.transform(y)
    y_pred_enc = final_le.transform(y_pred)

    acc = accuracy_score(y_enc, y_pred_enc)
    f1 = f1_score(y_enc, y_pred_enc, average="macro")

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")

    # To better understand performance on imbalanced data, print a detailed
    # classification report with per-class precision, recall, and F1-score.
    class_names = [FAULT_NAMES.get(l, l) for l in final_le.classes_]
    report = classification_report(y, y_pred, labels=final_le.classes_, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(y, y_pred, final_le, title)


def build_features_for_holdout(X_raw: np.ndarray, feature_cols_target: list) -> pd.DataFrame:
    """Re-engineers features for a raw gas dataset (like the holdout set)."""
    df = pd.DataFrame(X_raw, columns=GAS_COLS)
    df = add_log_gas_features(df)
    df = add_duval_ratios(df)
    df = add_iec_zone_codes(df)
    df = add_tdcg(df)
    df = add_temperature_diagnostic_features(df)
    for col in feature_cols_target:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols_target].fillna(0)


def main():
    """Main function to perform the three-way model evaluation."""
    print("Starting FleetTwin Model Evaluation...")

    clf_dir = os.path.join(CHECKPOINTS_DIR, "fault_classifier")
    try:
        # Load all models from the staged training process
        models = {
            "clf1": joblib.load(os.path.join(clf_dir, "stage1_calibrated.joblib")),
            "le1": joblib.load(os.path.join(clf_dir, "stage1_le.joblib")),
            "clf2a": joblib.load(os.path.join(clf_dir, "stage2a_discharge.joblib")),
            "le2a": joblib.load(os.path.join(clf_dir, "stage2a_le.joblib")),
            "clf2b": joblib.load(os.path.join(clf_dir, "stage2b_overheating.joblib")),
            "le2b": joblib.load(os.path.join(clf_dir, "stage2b_le.joblib")),
        }
        with open(os.path.join(clf_dir, "feature_cols.json")) as f:
            clf_feature_cols = json.load(f)

    except FileNotFoundError as e:
        print(f"Error: Model artifact not found. Did you run train.py first? ({e})")
        sys.exit(1)

    print(f"Loaded staged classifier trained on {len(clf_feature_cols)} features.")

    test_npz = np.load(os.path.join(PROCESSED_DIR, "test.npz"), allow_pickle=True)
    X_test, y_test, is_synthetic = test_npz["X"], test_npz["y"], test_npz["is_synthetic"]
    holdout_npz = np.load(os.path.join(PROCESSED_DIR, "real_holdout.npz"), allow_pickle=True)

    # Create a final label encoder that covers all possible fault classes for evaluation
    all_labels = np.unique(np.concatenate([test_npz['y'], holdout_npz['y']]))
    final_le = LabelEncoder().fit(all_labels)

    run_evaluation(X_test, y_test, final_le, models, "Full Test Set")

    real_indices = np.where(is_synthetic == False)[0]
    run_evaluation(X_test[real_indices], y_test[real_indices], final_le, models, "Real-Only Test Set")

    X_holdout_raw, y_holdout = holdout_npz["X"], holdout_npz["y"]
    X_holdout_featured = build_features_for_holdout(X_holdout_raw, clf_feature_cols)
    run_evaluation(X_holdout_featured.to_numpy(), y_holdout, final_le, models, "Frozen Holdout Set")

    print("-" * 50)
    print("\nEvaluation complete. Plots saved in outputs/plots/")


if __name__ == "__main__":
    main()