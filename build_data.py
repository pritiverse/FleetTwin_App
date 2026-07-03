"""
build_data.py
FleetTwin local data pipeline.

Run:
    python build_data.py --n-transformers 15 --n-days 14
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")
sys.path.insert(0, SRC)

from features import build_feature_table, add_health_lag_features, add_rolling_load_features  # noqa: E402
from twin_models import (  # noqa: E402
    aging_rate,
    choose_fault_class_by_health,
    daily_load_profile,
    health_from_cumulative_aging,
    hotspot_temperature,
    inject_ood_corruption,
    sample_gas_from_anchor,
)

RAW_DIR = os.path.join(BASE, "data", "raw")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
SYNTH_DIR = os.path.join(BASE, "data", "synthetic")
REPORTS_DIR = os.path.join(BASE, "outputs", "reports")
CONFIGS_DIR = os.path.join(BASE, "configs")

GAS_COLS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
LABEL_MAP = {
    "正常": "Normal",
    "局部放电": "PD",
    "低能放电": "D1",
    "高能放电": "D2",
    "低温过热": "T1",
    "中温过热": "T2",
    "高温过热": "T3",
}
SEED = 42
URLS = {
    "dataset_589.xlsx": "https://github.com/alan-456/transformer-fault-dataset/raw/main/dataset_(589).xlsx",
    "data.xlsx": "https://github.com/alan-456/transformer-fault-dataset/raw/main/data.xlsx",
}


def ensure_dirs() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, SYNTH_DIR, REPORTS_DIR, CONFIGS_DIR]:
        os.makedirs(directory, exist_ok=True)


def step_download() -> None:
    for fname, url in URLS.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            print(f"[cache hit] {fname}")
            continue
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, path)


def step_dedup_and_holdout(force: bool = False) -> pd.DataFrame:
    clean_path = os.path.join(PROCESSED_DIR, "real_clean_deduped.npz")
    holdout_path = os.path.join(PROCESSED_DIR, "real_holdout.npz")
    if not force and os.path.exists(clean_path) and os.path.exists(holdout_path):
        print("[cache hit] real_clean_deduped.npz / real_holdout.npz")
        data = np.load(clean_path, allow_pickle=True)
        return pd.DataFrame(data["X"], columns=GAS_COLS).assign(fault_class=data["y"])

    frames = [pd.read_excel(os.path.join(RAW_DIR, "dataset_589.xlsx")), pd.read_excel(os.path.join(RAW_DIR, "data.xlsx"))]
    for frame in frames:
        label_col = "故障类型"
        if label_col not in frame.columns:
            label_col = frame.columns[-1]
        frame["label"] = frame[label_col].map(LABEL_MAP)
        if frame["label"].isna().any():
            raise ValueError("Could not map all fault labels in raw Excel files.")

    merged = pd.concat([frame[GAS_COLS + ["label"]] for frame in frames], ignore_index=True)
    real_clean = merged.drop_duplicates(subset=GAS_COLS + ["label"]).rename(columns={"label": "fault_class"})
    real_clean = real_clean.reset_index(drop=True)
    np.savez(clean_path, X=real_clean[GAS_COLS].to_numpy(), y=real_clean["fault_class"].to_numpy(), columns=np.array(GAS_COLS))

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        real_clean[GAS_COLS].to_numpy(),
        real_clean["fault_class"].to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=real_clean["fault_class"].to_numpy(),
    )
    np.savez(holdout_path, X=X_holdout, y=y_holdout, columns=np.array(GAS_COLS))
    return pd.DataFrame(X_train, columns=GAS_COLS).assign(fault_class=y_train)


def step_anchor_stats(real_train_pool: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "anchor_stats.csv")
    if not force and os.path.exists(path):
        print("[cache hit] anchor_stats.csv")
        return pd.read_csv(path, index_col=0)

    rows = []
    for fault_class, group in real_train_pool.groupby("fault_class"):
        row = {"fault_class": fault_class, "n_real_train_rows": len(group)}
        for gas in GAS_COLS:
            row[f"{gas}_mean"] = group[gas].mean()
            row[f"{gas}_std"] = group[gas].std()
            row[f"{gas}_min"] = group[gas].min()
            row[f"{gas}_max"] = group[gas].max()
        rows.append(row)
    stats = pd.DataFrame(rows).set_index("fault_class")
    stats.to_csv(path)
    print(f"Saved anchor_stats.csv -> {path}")
    return stats


def simulate_transformer(meta_row, rng, anchor_stats, fault_classes, base_probs, n_days, steps_per_day, step_minutes):
    n_steps = n_days * steps_per_day
    timestamps = pd.date_range("2025-01-01", periods=n_steps, freq=f"{step_minutes}min")
    ambient_c = 30 + 8 * np.sin(2 * np.pi * (timestamps.dayofyear.values / 365.0)) + rng.normal(0, 2.0, n_steps)
    load_series = np.concatenate([daily_load_profile(steps_per_day, meta_row["base_load_frac"], rng) for _ in range(n_days)])
    age0 = meta_row["age_years"]
    elapsed_years = np.arange(n_steps) * step_minutes / (60 * 24 * 365.0)
    hotspot_c = hotspot_temperature(load_series, ambient_c)
    equivalent_years = np.cumsum(aging_rate(hotspot_c) * np.diff(np.concatenate([[0], elapsed_years])))
    health_scores = np.array([health_from_cumulative_aging(age0 + years) for years in equivalent_years])

    n_faults = max(1, int(0.25 * n_steps))
    health_weight = 101 - health_scores
    fault_timesteps = set(rng.choice(n_steps, size=n_faults, replace=False, p=health_weight / health_weight.sum()))

    rows = []
    for i in range(n_steps):
        fault_class = choose_fault_class_by_health(health_scores[i], fault_classes, base_probs, rng) if i in fault_timesteps else "Normal"
        anchor = anchor_stats.loc[fault_class] if fault_class in anchor_stats.index else anchor_stats.iloc[0]
        reading, was_corrupted = inject_ood_corruption(
            sample_gas_from_anchor(anchor, rng, age0 + elapsed_years[i], load_series[i]),
            rng,
            rate=0.02,
        )
        rows.append({
            "transformer_id": meta_row["transformer_id"],
            "timestamp": timestamps[i],
            "age_years": age0 + elapsed_years[i],
            "capacity_kva": meta_row["capacity_kva"],
            "region": meta_row["region"],
            "ambient_temp_c": ambient_c[i],
            "load_frac": load_series[i],
            "hotspot_temp_c": hotspot_c[i],
            "health_score": health_scores[i],
            "fault_class": fault_class,
            "is_fault_injected": i in fault_timesteps,
            "ood_corrupted": was_corrupted,
            **reading,
        })
    return pd.DataFrame(rows)


def step_synthesize_fleet(anchor_stats, n_transformers: int, n_days: int, force: bool = False) -> pd.DataFrame:
    path = os.path.join(SYNTH_DIR, "fleet_v1.parquet")
    if not force and os.path.exists(path):
        print("[cache hit] fleet_v1.parquet")
        return pd.read_parquet(path)

    rng = np.random.default_rng(SEED)
    fault_classes = anchor_stats.index.tolist()
    fault_only = [cls for cls in fault_classes if cls != "Normal"]
    base_probs = {cls: 1.0 / len(fault_only) for cls in fault_only}
    base_probs["Normal"] = 0.0
    fleet_meta = pd.DataFrame({
        "transformer_id": [f"T{idx:03d}" for idx in range(n_transformers)],
        "age_years": rng.integers(1, 41, n_transformers),
        "capacity_kva": rng.choice([100, 160, 200, 250, 315, 400, 500], n_transformers),
        "region": rng.choice(["North", "South", "East", "West", "Central"], n_transformers),
        "base_load_frac": rng.uniform(0.35, 0.85, n_transformers),
    })
    fleet = pd.concat(
        [simulate_transformer(row, rng, anchor_stats, fault_classes, base_probs, n_days, 288, 5) for _, row in fleet_meta.iterrows()],
        ignore_index=True,
    )
    fleet.to_parquet(path, index=False)
    print(f"Saved fleet_v1.parquet -> {path} ({fleet.shape})")
    return fleet


def step_preprocess(real_train_pool: pd.DataFrame, fleet_df: pd.DataFrame, force: bool = False) -> None:
    train_path = os.path.join(PROCESSED_DIR, "train.npz")
    if not force and os.path.exists(train_path):
        print("[cache hit] train/val/test.npz")
        return

    real_train_pool = real_train_pool.copy()
    real_train_pool["transformer_id"] = "REAL"
    real_feat = build_feature_table(real_train_pool, is_synthetic=False)
    fleet_feat = add_health_lag_features(add_rolling_load_features(build_feature_table(fleet_df, is_synthetic=True)))
    combined = pd.concat(
        [real_feat.reindex(columns=fleet_feat.columns.union(real_feat.columns)), fleet_feat.reindex(columns=fleet_feat.columns.union(real_feat.columns))],
        ignore_index=True,
    )
    feature_cols = [
        "H2", "CH4", "C2H6", "C2H4", "C2H2",
        "log_H2", "log_CH4", "log_C2H6", "log_C2H4", "log_C2H2",
        "TDCG", "ratio_C2H2_C2H4", "ratio_CH4_H2", "ratio_C2H4_C2H6",
        "iec_code_r1", "iec_code_r2", "iec_code_r5",
        "ratio_CH4_C2H6", "ratio_C2H6_C2H4", "ethylene_fraction", "THG", "c2h2_present",
    ]

    idx_train, idx_temp = train_test_split(combined.index.values, test_size=0.30, random_state=SEED, stratify=combined["fault_class"])
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=SEED, stratify=combined.loc[idx_temp, "fault_class"])

    def save_split(path, frame):
        np.savez(
            path,
            X=frame[feature_cols].fillna(0).to_numpy(),
            y=frame["fault_class"].to_numpy(),
            feature_cols=np.array(feature_cols),
            is_synthetic=frame["is_synthetic"].to_numpy(),
        )

    save_split(train_path, combined.loc[idx_train])
    save_split(os.path.join(PROCESSED_DIR, "val.npz"), combined.loc[idx_val])
    save_split(os.path.join(PROCESSED_DIR, "test.npz"), combined.loc[idx_test])
    with open(os.path.join(CONFIGS_DIR, "config_v1.yaml"), "w", encoding="utf-8") as f:
        yaml.dump({"seed": SEED, "feature_cols": feature_cols}, f, sort_keys=False)
    print("Saved train.npz / val.npz / test.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-transformers", type=int, default=15)
    parser.add_argument("--n-days", type=int, default=14)
    args = parser.parse_args()

    ensure_dirs()
    step_download()
    real_train_pool = step_dedup_and_holdout(force=args.force)
    anchor_stats = step_anchor_stats(real_train_pool, force=args.force)
    fleet_df = step_synthesize_fleet(anchor_stats, args.n_transformers, args.n_days, force=args.force)
    step_preprocess(real_train_pool, fleet_df, force=args.force)
    print("\nData pipeline complete.")


if __name__ == "__main__":
    main()
