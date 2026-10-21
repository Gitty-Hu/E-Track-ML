import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PASSENGER COMFORT & PRACTICALITY INDEX
# ============================================================

INPUT_FILE = Path(
    "data/processed/ml_training_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/comfort_training_dataset.csv"
)

print("=" * 70)
print("PASSENGER COMFORT & PRACTICALITY INDEX")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

# ------------------------------------------------------------
# SAFE NORMALIZATION
# ------------------------------------------------------------

def minmax(series, low=None, high=None):

    s = series.copy()

    if low is None:
        low = s.min()

    if high is None:
        high = s.max()

    if high == low:
        return pd.Series(
            0.5,
            index=s.index
        )

    return (
        (s - low) /
        (high - low)
    ).clip(0, 1)


# ------------------------------------------------------------
# 1. SEAT CAPACITY
# ------------------------------------------------------------

seat_score = minmax(
    df["seats"],
    4,
    7
)


# ------------------------------------------------------------
# 2. WHEELBASE
# ------------------------------------------------------------

wheelbase_score = minmax(
    df["wheelbase_mm"],
    2400,
    2850
)


# ------------------------------------------------------------
# 3. LUGGAGE CAPACITY
# ------------------------------------------------------------

luggage_capacity = (
    df["luggage_volume_l"] * 0.25
)

luggage_score = (
    luggage_capacity /
    (df["luggage_kg_per_trip"] + 1e-6)
).clip(
    0,
    2
) / 2


# ------------------------------------------------------------
# 4. SEAT UTILIZATION
# ------------------------------------------------------------

seat_utilization = (
    df["passengers_per_trip_2035"] /
    df["seats"]
).clip(
    0,
    1
)

# Moderate utilization is desirable.
# Very high utilization indicates crowding.

utilization_score = np.where(
    seat_utilization <= 0.75,

    seat_utilization / 0.75,

    np.maximum(
        0,
        1 -
        (
            seat_utilization - 0.75
        ) / 0.25
    )
)


# ------------------------------------------------------------
# 5. VEHICLE MASS
# ------------------------------------------------------------

# Lower mass generally helps efficiency and maneuverability.
mass_score = 1 - minmax(
    df["kerb_weight_kg"],
    900,
    2200
)


# ------------------------------------------------------------
# 6. POWER PER PASSENGER
# ------------------------------------------------------------

power_per_passenger = (
    df["power_kw"] /
    df["passengers_per_trip_2035"].clip(
        lower=1
    )
)

power_score = minmax(
    power_per_passenger,
    15,
    70
)


# ------------------------------------------------------------
# 7. TRIP DURATION
# ------------------------------------------------------------

# Longer trips require more comfort/practicality.
trip_duration_score = minmax(
    df["trip_distance_km"],
    5,
    250
)


# ------------------------------------------------------------
# WEIGHTED INDEX
# ------------------------------------------------------------

comfort_index = (

    0.20 * seat_score +

    0.20 * wheelbase_score +

    0.20 * luggage_score +

    0.15 * utilization_score +

    0.10 * mass_score +

    0.10 * power_score +

    0.05 * trip_duration_score

) * 100


# ------------------------------------------------------------
# STORE COMPONENTS
# ------------------------------------------------------------

df["seat_score"] = seat_score * 100
df["wheelbase_score"] = wheelbase_score * 100
df["luggage_score"] = luggage_score * 100
df["utilization_score"] = utilization_score * 100
df["mass_score"] = mass_score * 100
df["power_score"] = power_score * 100
df["trip_duration_score"] = (
    trip_duration_score * 100
)

df["comfort_index"] = comfort_index


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\nComfort index statistics:")

print(
    df["comfort_index"].describe()
)


print("\nMinimum comfort index:")

min_row = df.loc[
    df["comfort_index"].idxmin()
]

print(
    min_row[
        [
            "vehicle_id",
            "model",
            "comfort_index"
        ]
    ].to_dict()
)


print("\nMaximum comfort index:")

max_row = df.loc[
    df["comfort_index"].idxmax()
]

print(
    max_row[
        [
            "vehicle_id",
            "model",
            "comfort_index"
        ]
    ].to_dict()
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("COMFORT INDEX COMPLETE")
print("=" * 70)

print(
    f"\nRows: {len(df)}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print(
    "\nComfort index range: 0–100"
)

print(
    "\nIMPORTANT:"
)

print(
    "This is an engineering practicality proxy, "
    "not a subjective survey-based comfort rating."
)

print("=" * 70)