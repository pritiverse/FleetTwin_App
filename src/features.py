"""
features.py
FleetTwin — DGA (dissolved gas analysis) feature engineering.

All functions are pure / vectorized over a DataFrame with columns
['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2'] at minimum.
"""

import numpy as np
import pandas as pd

GAS_COLS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


def _safe_ratio(num: pd.Series, den: pd.Series, eps: float = 1e-6):
    return num / (den.replace(0, eps) + eps)


def add_log_gas_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform core gas features to handle wide dynamic range."""
    df = df.copy()
    for g in GAS_COLS:
        # Use log1p to handle zero values gracefully. Clip to ensure non-negative.
        df[f"log_{g}"] = np.log1p(df[g].clip(lower=0))
    return df


def add_duval_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Classic Duval-triangle / IEC three-ratio inputs."""
    df = df.copy()
    df["ratio_C2H2_C2H4"] = _safe_ratio(df["C2H2"], df["C2H4"])   # R1 (IEC)
    df["ratio_CH4_H2"] = _safe_ratio(df["CH4"], df["H2"])         # R2 (IEC)
    df["ratio_C2H4_C2H6"] = _safe_ratio(df["C2H4"], df["C2H6"])   # R5 (IEC)
    return df

def add_iec_zone_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    IEC 60599 Table 1 — three-ratio zone encoding.
    Each ratio maps to code 0, 1, or 2 based on standard thresholds.
    """
    df = df.copy()
    r1 = _safe_ratio(df["C2H2"], df["C2H4"])
    r2 = _safe_ratio(df["CH4"], df["H2"])
    r5 = _safe_ratio(df["C2H4"], df["C2H6"])

    # R1: C2H2/C2H4
    df["iec_code_r1"] = np.select([r1 < 0.1, r1 < 3.0], [0, 1], default=2)
    # R2: CH4/H2
    df["iec_code_r2"] = np.select([r2 < 0.1, r2 < 1.0], [0, 1], default=2)
    # R5: C2H4/C2H6
    df["iec_code_r5"] = np.select([r5 < 1.0, r5 < 3.0], [0, 1], default=2)
    return df


def add_temperature_diagnostic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features known to help separate thermal fault subtypes."""
    df = df.copy()
    eps = 1e-6

    # Methane-to-ethane ratio: key T1/T2/T3 separator
    # T1: CH4 dominates, T3: C2H4 and C2H6 dominate
    df["ratio_CH4_C2H6"]  = df["CH4"]  / (df["C2H6"] + eps)
    df["ratio_C2H6_C2H4"] = df["C2H6"] / (df["C2H4"] + eps)

    # Ethylene fraction: helps separate T2 from T1/T3
    denom = df["C2H4"] + df["C2H6"] + eps
    df["ethylene_fraction"] = df["C2H4"] / denom

    # Total hydrocarbon gas (THG): magnitude feature, separates D2 from D1
    df["THG"] = df["CH4"] + df["C2H6"] + df["C2H4"] + df["C2H2"]

    # C2H2 presence flag: almost zero in T-faults, nonzero in D-faults
    df["c2h2_present"] = (df["C2H2"] > 0.5).astype(float)

    return df


def add_tdcg(df: pd.DataFrame) -> pd.DataFrame:
    """Total Dissolved Combustible Gas (IEEE C57.104): sum of all 5 gases here
    (H2, CH4, C2H6, C2H4, C2H2 — CO/CO2 not present in this dataset)."""
    df = df.copy()
    df["TDCG"] = df[GAS_COLS].sum(axis=1)
    return df


def add_rolling_load_features(df: pd.DataFrame, group_col: str = "transformer_id",
                               time_col: str = "timestamp", load_col: str = "load_frac",
                               windows_minutes=(60, 1440), step_minutes: int = 5) -> pd.DataFrame:
    """
    Rolling mean/std of load for synthetic (time-series) rows only. Real
    rows have no time axis and will simply get NaN here (filled later with
    the is_synthetic flag so the model can condition on data source).
    """
    df = df.sort_values([group_col, time_col]).copy()
    for w_min in windows_minutes:
        w = max(1, w_min // step_minutes)
        grp = df.groupby(group_col)[load_col]
        df[f"load_roll_mean_{w_min}m"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"load_roll_std_{w_min}m"] = grp.transform(lambda s: s.rolling(w, min_periods=1).std())
    return df


def add_health_lag_features(df: pd.DataFrame, group_col: str = "transformer_id",
                             time_col: str = "timestamp", health_col: str = "health_score",
                             lags=(1, 12, 288)) -> pd.DataFrame:
    """Lagged health-score features (synthetic rows only)."""
    df = df.sort_values([group_col, time_col]).copy()
    for lag in lags:
        df[f"{health_col}_lag_{lag}"] = df.groupby(group_col)[health_col].shift(lag)
    return df


def build_feature_table(df: pd.DataFrame, is_synthetic: bool) -> pd.DataFrame:
    """
    Full feature pipeline applied identically to real and synthetic rows.
    `is_synthetic` flags which branch of optional (time/age/health) features
    can actually be populated; NaNs are left for the model / imputer to
    handle and reported as the explicit is_synthetic column.
    """
    df = add_log_gas_features(df)
    df = add_duval_ratios(df)
    df = add_iec_zone_codes(df)
    df = add_tdcg(df)
    df = add_temperature_diagnostic_features(df)
    df["is_synthetic"] = is_synthetic
    return df