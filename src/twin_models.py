"""
twin_models.py
FleetTwin - physics-based transformer aging model + synthetic fleet generator.
"""

import numpy as np

REFERENCE_HOTSPOT_C = 110.0
ACTIVATION_ENERGY_K = 15000.0
TOP_OIL_RISE_RATED_C = 55.0
HOTSPOT_GRADIENT_RATED_C = 15.0


def hotspot_temperature(load_pct: np.ndarray, ambient_c: np.ndarray, n: float = 0.8, m: float = 0.8):
    """Approximate winding hotspot temperature from load fraction and ambient temperature."""
    top_oil_rise = TOP_OIL_RISE_RATED_C * (load_pct ** n)
    hotspot_gradient = HOTSPOT_GRADIENT_RATED_C * (load_pct ** m)
    return ambient_c + top_oil_rise + hotspot_gradient


def aging_rate(hotspot_c: np.ndarray):
    """Arrhenius relative aging rate versus the 110C reference hotspot temperature."""
    return np.exp(
        (ACTIVATION_ENERGY_K / (REFERENCE_HOTSPOT_C + 273.0))
        - (ACTIVATION_ENERGY_K / (hotspot_c + 273.0))
    )


def health_from_cumulative_aging(cumulative_relative_age_years: float, rated_life_years: float = 30.0):
    """Map equivalent consumed life to a 0-100 health score."""
    frac_used = cumulative_relative_age_years / rated_life_years
    return float(np.clip(100.0 * (1.0 - frac_used), 0.0, 100.0))


def daily_load_profile(
    n_steps_per_day: int,
    base_load: float,
    rng: np.random.Generator,
    morning_peak_hr: float = 9.0,
    evening_peak_hr: float = 19.0,
    noise_std: float = 0.05,
):
    """Generate a two-peak daily load profile."""
    hours = np.linspace(0, 24, n_steps_per_day, endpoint=False)
    morning = np.exp(-0.5 * ((hours - morning_peak_hr) / 2.5) ** 2)
    evening = np.exp(-0.5 * ((hours - evening_peak_hr) / 2.5) ** 2)
    shape = 0.45 + 0.35 * morning + 0.45 * evening
    load = base_load * shape / shape.max()
    load = load + rng.normal(0, noise_std, size=n_steps_per_day)
    return np.clip(load, 0.02, 1.4)


def sample_gas_from_anchor(anchor_row, rng: np.random.Generator, age_years: float, load_frac: float,
                           gas_cols=("H2", "CH4", "C2H6", "C2H4", "C2H2")):
    """Sample a gas reading from per-class anchor statistics."""
    age_factor = 1.0 + 0.01 * age_years
    load_factor = 1.0 + 0.15 * max(load_frac - 0.8, 0)
    noise_scale = age_factor * load_factor

    out = {}
    for gas in gas_cols:
        std_eff = max(anchor_row[f"{gas}_std"], 1e-6) * noise_scale
        val = rng.normal(anchor_row[f"{gas}_mean"], std_eff)
        val = float(np.clip(val, anchor_row[f"{gas}_min"] * 0.5, anchor_row[f"{gas}_max"] * 1.5))
        out[gas] = max(val, 0.0)
    return out


def choose_fault_class_by_health(health_score: float, classes, base_probs: dict, rng: np.random.Generator):
    """Sample a fault class, biasing more severe classes as health declines."""
    probs = np.array([base_probs.get(c, 1e-6) for c in classes], dtype=float)
    probs = probs / probs.sum()

    if health_score < 70:
        severity = (70 - health_score) / 70.0
        boost = np.zeros_like(probs)
        for i, cls in enumerate(classes):
            if cls in ("D2", "T3"):
                boost[i] = severity * 3.0
            elif cls == "Normal":
                boost[i] = -severity * 2.0
        probs = probs * np.exp(boost)
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum()

    return rng.choice(classes, p=probs)


def inject_ood_corruption(reading: dict, rng: np.random.Generator, rate: float = 0.02):
    """Randomly corrupt a reading to simulate sensor dropout or spikes."""
    if rng.random() >= rate:
        return reading, False

    corrupted = dict(reading)
    gas = rng.choice(list(reading.keys()))
    if rng.choice(["dropout", "spike"]) == "dropout":
        corrupted[gas] = 0.0
    else:
        corrupted[gas] = corrupted[gas] * rng.uniform(5, 15)
    return corrupted, True
