"""
agent_tools.py
FleetTwin — tool functions for the single-agent LangGraph diagnostic demo.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import shap

from features import (
    add_log_gas_features, add_duval_ratios, add_iec_zone_codes,
    add_tdcg, add_temperature_diagnostic_features
)

FAULT_NAMES = {
    "Normal": "Normal (no fault)",
    "PD": "Partial Discharge",
    "D1": "Low-energy Discharge",
    "D2": "High-energy Discharge",
    "T1": "Low-temperature Overheating",
    "T2": "Mid-temperature Overheating",
    "T3": "High-temperature Overheating",
}

GAS_COLS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


class FleetTwinToolkit:
    """
    Holds references to the loaded fleet dataframe + trained models.
    """

    def __init__(self, fleet_df: pd.DataFrame, models: dict, feature_cols: list):
        self.fleet_df = fleet_df
        self.models = models
        self.feature_cols = feature_cols

    def _build_features(self, row: pd.Series) -> pd.DataFrame:
        """Builds all necessary features for a given data row."""
        df = pd.DataFrame([row.to_dict()])
        df = add_log_gas_features(df)
        df = add_duval_ratios(df)
        df = add_iec_zone_codes(df)
        df = add_tdcg(df)
        df = add_temperature_diagnostic_features(df)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_cols].fillna(0)

    def _staged_predict_with_shap(self, X: pd.DataFrame) -> dict:
        """Implements the full staged prediction logic with SHAP explanations."""
        s1_pred_idx = self.models["s1"].predict(X)[0]
        s1_pred_label = self.models["le1"].inverse_transform([s1_pred_idx])[0]

        if s1_pred_label == "Normal":
            return {"label": "Normal", "explainer": self.models["shap1"], "class_idx": int(s1_pred_idx)}
        elif s1_pred_label == "Discharge":
            s2_pred_idx = self.models["s2a"].predict(X)[0]
            return {"label": self.models["le2a"].inverse_transform([s2_pred_idx])[0], "explainer": self.models["shap2a"], "class_idx": int(s2_pred_idx)}
        else:  # Overheating
            s2_pred_idx = self.models["s2b"].predict(X)[0]
            return {"label": self.models["le2b"].inverse_transform([s2_pred_idx])[0], "explainer": self.models["shap2b"], "class_idx": int(s2_pred_idx)}

    def get_transformer_status(self, transformer_id: str) -> dict:
        """Tool 1: latest known status snapshot for a transformer."""
        rows = self.fleet_df[self.fleet_df["transformer_id"] == transformer_id]
        if rows.empty:
            return {"error": f"No records found for transformer {transformer_id}"}
        latest = rows.sort_values("timestamp").iloc[-1]
        return {
            "transformer_id": transformer_id,
            "timestamp": str(latest["timestamp"]),
            "health_score": float(latest.get("health_score", np.nan)),
            "load_frac": float(latest.get("load_frac", np.nan)),
            "age_years": float(latest.get("age_years", np.nan)),
            "gas_readings": {g: float(latest[g]) for g in GAS_COLS if g in latest},
        }

    def run_health_check(self, transformer_id: str) -> dict:
        """Tool 2: trend-based health check over the available history."""
        rows = self.fleet_df[self.fleet_df["transformer_id"] == transformer_id].sort_values("timestamp")
        if rows.empty:
            return {"error": f"No records found for transformer {transformer_id}"}
        health = rows["health_score"]
        trend = "declining" if health.iloc[-1] < health.iloc[0] else "stable/improving"
        return {
            "transformer_id": transformer_id,
            "current_health": float(health.iloc[-1]),
            "min_health_observed": float(health.min()),
            "trend": trend,
            "n_records": int(len(rows)),
        }

    def predict_fault(self, transformer_id: str) -> dict:
        """Tool 3: run the trained classifier on the transformer's latest features."""
        rows = self.fleet_df[self.fleet_df["transformer_id"] == transformer_id].sort_values("timestamp")
        if rows.empty:
            return {"error": f"No records found for transformer {transformer_id}"}
        latest = rows.iloc[[-1]]
        X = self._build_features(latest.iloc[0])

        result = self._staged_predict_with_shap(X)
        pred_label = result["label"]

        shap_vals = result["explainer"](X)
        contribs = shap_vals.values[0, :, result["class_idx"]]
        top_idx = np.argsort(-np.abs(contribs))[:3]
        explanation = [{"feature": self.feature_cols[i], "shap_value": float(contribs[i])} for i in top_idx]

        return {
            "transformer_id": transformer_id,
            "predicted_class": str(pred_label),
            "predicted_class_name": FAULT_NAMES.get(str(pred_label), str(pred_label)),
            "class_probabilities": {},  # Probs are complex in staged model, omitted for tool simplicity
            "top_shap_features": explanation,
        }

    def simulate_load_increase(self, transformer_id: str, load_increase_pct: float) -> dict:
        """Tool 4: what-if — bump load and re-run the physics aging model + classifier."""
        from twin_models import hotspot_temperature, aging_rate

        rows = self.fleet_df[self.fleet_df["transformer_id"] == transformer_id].sort_values("timestamp")
        if rows.empty:
            return {"error": f"No records found for transformer {transformer_id}"}
        latest = rows.iloc[-1].copy()

        new_load = float(latest["load_frac"]) * (1 + load_increase_pct / 100.0)
        ambient = float(latest.get("ambient_temp_c", 30.0))
        old_hotspot = hotspot_temperature(np.array([latest["load_frac"]]), np.array([ambient]))[0]
        new_hotspot = hotspot_temperature(np.array([new_load]), np.array([ambient]))[0]
        old_rate = aging_rate(np.array([old_hotspot]))[0]
        new_rate = aging_rate(np.array([new_hotspot]))[0]

        return {
            "transformer_id": transformer_id,
            "load_increase_pct": load_increase_pct,
            "old_load_frac": round(float(latest["load_frac"]), 3),
            "new_load_frac": round(new_load, 3),
            "old_hotspot_c": round(float(old_hotspot), 1),
            "new_hotspot_c": round(float(new_hotspot), 1),
            "relative_aging_rate_multiplier": round(float(new_rate / max(old_rate, 1e-9)), 2),
            "note": "Gas readings are not re-simulated here; this estimates only the "
                    "thermal/aging impact of the proposed load change.",
        }


def load_toolkit(fleet_parquet_path: str, classifier_dir: str) -> "FleetTwinToolkit":
    """Convenience loader for agent demos."""
    fleet_df = pd.read_parquet(fleet_parquet_path)

    models = {
        "s1": joblib.load(os.path.join(classifier_dir, "stage1_calibrated.joblib")),
        "le1": joblib.load(os.path.join(classifier_dir, "stage1_le.joblib")),
        "s2a": joblib.load(os.path.join(classifier_dir, "stage2a_discharge.joblib")),
        "le2a": joblib.load(os.path.join(classifier_dir, "stage2a_le.joblib")),
        "s2b": joblib.load(os.path.join(classifier_dir, "stage2b_overheating.joblib")),
        "le2b": joblib.load(os.path.join(classifier_dir, "stage2b_le.joblib")),
    }

    with open(os.path.join(classifier_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    # Load uncalibrated models for SHAP compatibility
    s1_uncalib = joblib.load(os.path.join(classifier_dir, "stage1.joblib"))

    # Create SHAP explainers
    models["shap1"] = shap.Explainer(s1_uncalib, feature_names=feature_cols)
    models["shap2a"] = shap.Explainer(models["s2a"], feature_names=feature_cols)
    models["shap2b"] = shap.Explainer(models["s2b"], feature_names=feature_cols)

    return FleetTwinToolkit(fleet_df, models, feature_cols)