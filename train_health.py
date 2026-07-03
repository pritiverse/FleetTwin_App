"""
train_health.py
FleetTwin - trains the health/RUL regressor.

Run after build_data.py:
    python train_health.py
"""

import importlib.util
import json
import os
import sys

if importlib.util.find_spec("optuna_integration") is None:
    print("ERROR: The 'optuna-integration' package is not installed.", file=sys.stderr)
    print("Install it with: pip install optuna-integration[lightgbm]", file=sys.stderr)
    sys.exit(1)

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from optuna_integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")
sys.path.insert(0, SRC)

from features import add_duval_ratios, add_tdcg  # noqa: E402

SYNTH_DIR = os.path.join(BASE, "data", "synthetic")
CHECKPOINTS_DIR = os.path.join(BASE, "checkpoints")
SEED = 42
REGRESSOR_TRIALS = int(os.getenv("FLEETTWIN_REGRESSOR_TRIALS", "30"))
REGRESSOR_TIMEOUT = int(os.getenv("FLEETTWIN_REGRESSOR_TIMEOUT", "300"))
REGRESSOR_ESTIMATORS = int(os.getenv("FLEETTWIN_REGRESSOR_ESTIMATORS", "500"))


def train_health_regressor() -> None:
    """Train the LightGBM health regressor on synthetic fleet telemetry."""
    model_dir = os.path.join(CHECKPOINTS_DIR, "health_regressor")
    os.makedirs(model_dir, exist_ok=True)

    fleet_path = os.path.join(SYNTH_DIR, "fleet_v1.parquet")
    if not os.path.exists(fleet_path):
        raise FileNotFoundError("Missing data/synthetic/fleet_v1.parquet. Run python build_data.py first.")

    fleet_df = pd.read_parquet(fleet_path)
    fleet_df = add_tdcg(add_duval_ratios(fleet_df))

    feature_cols = [
        "H2", "CH4", "C2H6", "C2H4", "C2H2",
        "ratio_C2H2_C2H4", "ratio_CH4_H2", "ratio_C2H4_C2H6", "TDCG",
        "age_years", "load_frac", "ambient_temp_c", "hotspot_temp_c", "capacity_kva",
    ]
    X = fleet_df[feature_cols].fillna(0)
    y = fleet_df["health_score"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "l1",
            "random_state": SEED,
            "n_estimators": REGRESSOR_ESTIMATORS,
            "verbose": -1,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
        reg = lgb.LGBMRegressor(**params)
        reg.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(15, verbose=False),
                LightGBMPruningCallback(trial, "l1", valid_name="valid_0"),
            ],
        )
        return mean_absolute_error(y_val, reg.predict(X_val))

    print("\n[health regressor] Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=REGRESSOR_TRIALS, timeout=REGRESSOR_TIMEOUT)

    print(f"\n[health regressor] Optuna study complete. Best trial MAE: {study.best_value:.4f}")
    print(f"[health regressor] Best params: {study.best_params}")
    reg = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        random_state=SEED,
        n_estimators=REGRESSOR_ESTIMATORS,
        verbose=-1,
        n_jobs=-1,
        **study.best_params,
    )
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])

    y_pred = reg.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"[health regressor] Final model MAE={mae:.3f} R2={r2:.4f}")

    joblib.dump(reg, os.path.join(model_dir, "best.joblib"))
    with open(os.path.join(model_dir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(model_dir, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"mae": mae, "r2": r2, "feature_cols": feature_cols, "best_hyperparameters": study.best_params},
            f,
            indent=2,
        )
    print(f"Saved health regressor -> {model_dir}")


if __name__ == "__main__":
    train_health_regressor()
    print("\nHealth regressor training complete.")
