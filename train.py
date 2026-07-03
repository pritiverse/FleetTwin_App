"""
train.py
FleetTwin - trains all model artifacts used by the local app.

Run after build_data.py:
    python train.py
"""

from train_health import train_health_regressor
from train_staged import train_fault_classifier


def main() -> None:
    train_fault_classifier()
    train_health_regressor()
    print("\nAll FleetTwin models trained.")


if __name__ == "__main__":
    main()
