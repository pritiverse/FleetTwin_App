"""
app/main.py
FleetTwin local app — FastAPI service connecting the trained fault classifier
and health regressor to a simple web dashboard + JSON API.

Run with:
    uvicorn app.main:app --reload --port 8000
Then open http://localhost:8000
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
sys.path.insert(0, SRC)

from features import (  # noqa: E402
    add_log_gas_features,
    add_duval_ratios,
    add_iec_zone_codes,
    add_tdcg,
    add_temperature_diagnostic_features,
)
from twin_models import hotspot_temperature, aging_rate  # noqa: E402

CHECKPOINTS_DIR = os.path.join(BASE, "checkpoints")
SYNTH_DIR = os.path.join(BASE, "data", "synthetic")

FAULT_NAMES = {
    "Normal": "Normal (no fault)", "PD": "Partial Discharge",
    "D1": "Low-energy Discharge", "D2": "High-energy Discharge",
    "T1": "Low-temperature Overheating", "T2": "Mid-temperature Overheating",
    "T3": "High-temperature Overheating",
}

app = FastAPI(title="FleetTwin", description="Distribution transformer fleet health monitoring")
templates = Jinja2Templates(directory=os.path.join(BASE, "app", "templates"))
STATIC_DIR = os.path.join(BASE, "app", "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ModelStore:
    """Loads all trained artifacts once at startup and keeps them in memory."""

    def __init__(self):
        clf_dir = os.path.join(CHECKPOINTS_DIR, "fault_classifier")
        reg_dir = os.path.join(CHECKPOINTS_DIR, "health_regressor")

        # Load staged classifiers and their label encoders
        self.classifier_s1 = joblib.load(os.path.join(clf_dir, "stage1_calibrated.joblib"))
        self.le_s1 = joblib.load(os.path.join(clf_dir, "stage1_le.joblib"))
        # Load uncalibrated model for SHAP TreeExplainer compatibility
        clf1_uncalibrated = joblib.load(os.path.join(clf_dir, "stage1.joblib"))

        self.classifier_s2a = joblib.load(os.path.join(clf_dir, "stage2a_discharge.joblib"))
        self.le_s2a = joblib.load(os.path.join(clf_dir, "stage2a_le.joblib"))
        self.classifier_s2b = joblib.load(os.path.join(clf_dir, "stage2b_overheating.joblib"))
        self.le_s2b = joblib.load(os.path.join(clf_dir, "stage2b_le.joblib"))

        with open(os.path.join(clf_dir, "feature_cols.json")) as f:
            self.clf_feature_cols = json.load(f)

        self.health_regressor = joblib.load(os.path.join(reg_dir, "best.joblib"))
        with open(os.path.join(reg_dir, "feature_cols.json")) as f:
            self.reg_feature_cols = json.load(f)
        with open(os.path.join(reg_dir, "best_model.json")) as f:
            self.reg_meta = json.load(f)
        # The old clf_meta is no longer valid for the staged model
        self.clf_meta = {"architecture": "two-stage"}

        self.fleet_df = pd.read_parquet(os.path.join(SYNTH_DIR, "fleet_v1.parquet"))

        # Create SHAP explainers for each model in the cascade
        self.shap_explainer_s1 = shap.Explainer(
            clf1_uncalibrated, feature_names=self.clf_feature_cols)
        self.shap_explainer_s2a = shap.Explainer(
            self.classifier_s2a, feature_names=self.clf_feature_cols)
        self.shap_explainer_s2b = shap.Explainer(
            self.classifier_s2b, feature_names=self.clf_feature_cols)

    def latest_row(self, transformer_id: str) -> pd.Series:
        rows = self.fleet_df[self.fleet_df["transformer_id"] == transformer_id]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"Unknown transformer_id '{transformer_id}'")
        return rows.sort_values("timestamp").iloc[-1]

    def build_clf_features(self, row: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame([row.to_dict()])
        df = add_log_gas_features(df)
        df = add_duval_ratios(df)
        df = add_iec_zone_codes(df)
        df = add_tdcg(df)
        df = add_temperature_diagnostic_features(df)
        for col in self.clf_feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.clf_feature_cols].fillna(0)

    def build_reg_features(self, row: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame([row.to_dict()])
        df = add_duval_ratios(df)
        df = add_tdcg(df)
        return df[self.reg_feature_cols].fillna(0)


store: ModelStore | None = None


def get_store() -> ModelStore:
    if store is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")
    return store


@app.on_event("startup")
def load_models():
    global store
    store = ModelStore()
    print(f"Loaded {len(store.fleet_df['transformer_id'].unique())} transformers, regressor R2={store.reg_meta.get('r2'):.4f}")


# ---------------------------------------------------------------------------
# Web Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Serve the FleetTwin web dashboard."""
    return templates.TemplateResponse(request, "dashboard.html", {"request": request})


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.get("/api/transformers")
def list_transformers():
    model_store = get_store()
    latest = model_store.fleet_df.sort_values("timestamp").groupby("transformer_id").tail(1)
    latest = latest.sort_values("transformer_id")
    return [
        {
            "transformer_id": r["transformer_id"],
            "region": r["region"],
            "age_years": round(float(r["age_years"]), 1),
            "capacity_kva": int(r["capacity_kva"]),
            "health_score": round(float(r["health_score"]), 1),
            "fault_class": r["fault_class"],
        }
        for _, r in latest.iterrows()
    ]


@app.get("/api/transformers/{transformer_id}/status")
def transformer_status(transformer_id: str):
    model_store = get_store()
    row = model_store.latest_row(transformer_id)
    return {
        "transformer_id": transformer_id,
        "timestamp": str(row["timestamp"]),
        "region": row["region"],
        "age_years": round(float(row["age_years"]), 2),
        "capacity_kva": int(row["capacity_kva"]),
        "load_frac": round(float(row["load_frac"]), 3),
        "ambient_temp_c": round(float(row["ambient_temp_c"]), 1),
        "hotspot_temp_c": round(float(row["hotspot_temp_c"]), 1),
        "health_score": round(float(row["health_score"]), 1),
        "gas_readings": {g: round(float(row[g]), 3) for g in ["H2", "CH4", "C2H6", "C2H4", "C2H2"]},
    }


@app.get("/api/transformers/{transformer_id}/predict")
def predict_fault(transformer_id: str):
    model_store = get_store()
    row = model_store.latest_row(transformer_id)
    X = model_store.build_clf_features(row)

    # --- Staged Prediction Logic ---
    stage1_pred_idx = model_store.classifier_s1.predict(X)[0]
    stage1_pred_label = model_store.le_s1.inverse_transform([stage1_pred_idx])[0]

    if stage1_pred_label == "Normal":
        pred_label = "Normal"
        shap_vals = model_store.shap_explainer_s1(X)
        class_idx = list(model_store.le_s1.classes_).index("Normal")
        contribs = shap_vals.values[0, :, class_idx]

    elif stage1_pred_label == "Discharge":
        pred_idx = model_store.classifier_s2a.predict(X)[0]
        pred_label = model_store.le_s2a.inverse_transform([pred_idx])[0]
        shap_vals = model_store.shap_explainer_s2a(X)
        contribs = shap_vals.values[0, :, pred_idx]

    else:  # Overheating
        pred_idx = model_store.classifier_s2b.predict(X)[0]
        pred_label = model_store.le_s2b.inverse_transform([pred_idx])[0]
        shap_vals = model_store.shap_explainer_s2b(X)
        contribs = shap_vals.values[0, :, pred_idx]

    # For simplicity, class probabilities are not returned from the staged model.
    proba_map = {}

    top_idx = np.argsort(-np.abs(contribs))[:5]
    top_features = [
        {"feature": model_store.clf_feature_cols[i], "shap_value": round(float(contribs[i]), 4)}
        for i in top_idx
    ]

    return {
        "transformer_id": transformer_id,
        "predicted_class": str(pred_label),
        "predicted_class_name": FAULT_NAMES.get(str(pred_label), str(pred_label)),
        "class_probabilities": proba_map,
        "top_shap_features": top_features,
    }


@app.get("/api/transformers/{transformer_id}/health")
def predict_health(transformer_id: str):
    model_store = get_store()
    row = model_store.latest_row(transformer_id)
    X = model_store.build_reg_features(row)
    pred = float(model_store.health_regressor.predict(X)[0])
    return {
        "transformer_id": transformer_id,
        "predicted_health_score": round(pred, 1),
        "actual_health_score": round(float(row["health_score"]), 1),
    }


class SimulateRequest(BaseModel):
    load_increase_pct: float = 10.0


@app.post("/api/transformers/{transformer_id}/simulate")
def simulate_load_increase(transformer_id: str, body: SimulateRequest):
    model_store = get_store()
    row = model_store.latest_row(transformer_id)
    new_load = float(row["load_frac"]) * (1 + body.load_increase_pct / 100.0)
    ambient = float(row["ambient_temp_c"])

    old_hotspot = float(hotspot_temperature(np.array([row["load_frac"]]), np.array([ambient]))[0])
    new_hotspot = float(hotspot_temperature(np.array([new_load]), np.array([ambient]))[0])
    old_rate = float(aging_rate(np.array([old_hotspot]))[0])
    new_rate = float(aging_rate(np.array([new_hotspot]))[0])

    return {
        "transformer_id": transformer_id,
        "load_increase_pct": body.load_increase_pct,
        "old_load_frac": round(float(row["load_frac"]), 3),
        "new_load_frac": round(new_load, 3),
        "old_hotspot_c": round(old_hotspot, 1),
        "new_hotspot_c": round(new_hotspot, 1),
        "relative_aging_rate_multiplier": round(new_rate / max(old_rate, 1e-9), 2),
    }


@app.get("/api/model_info")
def model_info():
    model_store = get_store()
    return {
        "fault_classifier": model_store.clf_meta,
        "health_regressor": model_store.reg_meta,
    }