"""
train_staged.py
FleetTwin - trains the two-stage fault classifier.

Run after build_data.py:
    python train_staged.py
"""

import json
import os

import joblib
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE, "checkpoints")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "fault_classifier")

DISCHARGE = {"PD", "D1", "D2"}
OVERHEATING = {"T1", "T2", "T3"}
SEED = 42


def group_label(label: str) -> str:
    """Map detailed fault labels into the Stage 1 taxonomy."""
    if label == "Normal":
        return "Normal"
    if label in DISCHARGE:
        return "Discharge"
    return "Overheating"


def _xgb_classifier(max_depth: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=max_depth,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=10,
    )


def _fit_calibrator(prefit_model, X_cal, y_cal):
    """
    Calibrate an already-fitted model using the 'prefit' strategy.
    """
    # The `cv="prefit"` option assumes the estimator is already fitted
    # and uses the data provided (X_cal, y_cal) for calibration.
    calibrator = CalibratedClassifierCV(
        estimator=prefit_model, method="isotonic", cv="prefit"
    )
    calibrator.fit(X_cal, y_cal)
    return calibrator


def train_fault_classifier() -> None:
    """Train and save the staged fault-classifier artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    train_npz = np.load(os.path.join(PROCESSED_DIR, "train.npz"), allow_pickle=True)
    val_npz = np.load(os.path.join(PROCESSED_DIR, "val.npz"), allow_pickle=True)
    X_tr, y_tr = train_npz["X"], train_npz["y"]
    X_val, y_val = val_npz["X"], val_npz["y"]
    is_synth_tr = train_npz["is_synthetic"]
    is_synth_val = val_npz["is_synthetic"]
    feature_cols = list(train_npz["feature_cols"])

    print("Training Stage 1 classifier...")
    y_tr_g = np.array([group_label(y) for y in y_tr])
    y_val_g = np.array([group_label(y) for y in y_val])
    le1 = LabelEncoder()
    clf1 = _xgb_classifier(max_depth=5)
    sw = np.where(is_synth_tr, 1.0, 75.0)
    clf1.fit(
        X_tr,
        le1.fit_transform(y_tr_g),
        sample_weight=sw,
        eval_set=[(X_val, le1.transform(y_val_g))],
        verbose=False,
    )
    joblib.dump(clf1, os.path.join(MODEL_DIR, "stage1.joblib"))
    joblib.dump(le1, os.path.join(MODEL_DIR, "stage1_le.joblib"))
    print("Stage 1 classifier saved.")

    print("Calibrating Stage 1 classifier...")
    real_val_mask = is_synth_val == False
    if np.any(real_val_mask):
        X_val_real = X_val[real_val_mask]
        y_val_g_real_enc = le1.transform(y_val_g[real_val_mask])
        calibrated_clf1 = _fit_calibrator(clf1, X_val_real, y_val_g_real_enc)
        joblib.dump(calibrated_clf1, os.path.join(MODEL_DIR, "stage1_calibrated.joblib"))
        print("Calibrated Stage 1 classifier saved.")
    else:
        joblib.dump(clf1, os.path.join(MODEL_DIR, "stage1_calibrated.joblib"))
        print("No real validation rows found; saved uncalibrated Stage 1 as fallback.")

    print("Training Stage 2a (Discharge) sub-classifier...")
    mask_d_tr = np.array([y in DISCHARGE for y in y_tr])
    mask_d_val = np.array([y in DISCHARGE for y in y_val])
    le2a = LabelEncoder()
    clf2a = _xgb_classifier(max_depth=4)
    sw_d = np.where(is_synth_tr[mask_d_tr], 1.0, 100.0)
    clf2a.fit(
        X_tr[mask_d_tr],
        le2a.fit_transform(y_tr[mask_d_tr]),
        sample_weight=sw_d,
        eval_set=[(X_val[mask_d_val], le2a.transform(y_val[mask_d_val]))],
        verbose=False,
    )
    joblib.dump(clf2a, os.path.join(MODEL_DIR, "stage2a_discharge.joblib"))
    joblib.dump(le2a, os.path.join(MODEL_DIR, "stage2a_le.joblib"))
    print("Stage 2a classifier saved.")

    print("Training Stage 2b (Overheating) sub-classifier...")
    mask_t_tr = np.array([y in OVERHEATING for y in y_tr])
    mask_t_val = np.array([y in OVERHEATING for y in y_val])
    le2b = LabelEncoder()
    clf2b = _xgb_classifier(max_depth=4)
    sw_t = np.where(is_synth_tr[mask_t_tr], 1.0, 100.0)
    clf2b.fit(
        X_tr[mask_t_tr],
        le2b.fit_transform(y_tr[mask_t_tr]),
        sample_weight=sw_t,
        eval_set=[(X_val[mask_t_val], le2b.transform(y_val[mask_t_val]))],
        verbose=False,
    )
    joblib.dump(clf2b, os.path.join(MODEL_DIR, "stage2b_overheating.joblib"))
    joblib.dump(le2b, os.path.join(MODEL_DIR, "stage2b_le.joblib"))
    print("Stage 2b classifier saved.")

    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    print("\nStaged classifier training complete.")


if __name__ == "__main__":
    train_fault_classifier()