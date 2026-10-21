import pandas as pd
from pathlib import Path

# ============================================================
# ML DATASET PREPARATION
# ============================================================

INPUT_FILE = Path("data/processed/fleet_economics_2035.csv")
OUTPUT_FILE = Path("data/processed/ml_training_dataset.csv")

print("=" * 70)
print("ML DATASET PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal rows: {len(df)}")
print(f"Original columns: {len(df.columns)}")

# ------------------------------------------------------------
# SELECT VARIABLES
# ------------------------------------------------------------

vehicle_features = [
    "vehicle_id",
    "manufacturer",
    "model",
    "powertrain",
    "vehicle_type",
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "battery_capacity_kwh",
    "certified_range_km",
    "power_kw",
    "purchase_price_inr",
]

duty_cycle_features = [
    "city",
    "scenario_id",
    "scenario_name",
    "trip_distance_km",
    "trips_per_day_2035",
    "passengers_per_trip_2035",
    "luggage_kg_per_trip",
    "daily_distance_km_2035",
    "daily_operating_hours_2035",
]

target_columns = [
    "energy_quantity_per_km",
    "energy_cost_inr_per_km_2035",
    "annual_profit_inr",
    "profit_margin_percent",
    "profit_per_vehicle_km_inr",
    "profit_per_passenger_km_inr",
    "annual_co2_kg",
    "annual_co2_tonnes",
    "seat_utilization_percent",
    "luggage_capacity_proxy_kg",
    "range_margin_km",
    "range_feasible",
    "luggage_feasible",
]

keep_columns = vehicle_features + duty_cycle_features + target_columns

# Keep only columns that actually exist
keep_columns = [c for c in keep_columns if c in df.columns]

ml_df = df[keep_columns].copy()

# ------------------------------------------------------------
# REMOVE COMPLETELY MISSING FEATURES
# ------------------------------------------------------------

remove_features = []

for col in ml_df.columns:

    if col in target_columns:
        continue

    if ml_df[col].isna().all():
        remove_features.append(col)

if remove_features:

    print("\nRemoving completely missing features:")

    for col in remove_features:
        print(f"  - {col}")

    ml_df.drop(columns=remove_features, inplace=True)

# ------------------------------------------------------------
# DATA QUALITY REPORT
# ------------------------------------------------------------

print("\nRemaining missing values:")

missing = ml_df.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("  None")

else:
    print(missing)

# ------------------------------------------------------------
# UNIQUE VEHICLES
# ------------------------------------------------------------

print("\nUnique vehicles:")
print(ml_df["vehicle_id"].nunique())

print("\nRows per vehicle:")
print(
    ml_df.groupby("vehicle_id")
    .size()
    .describe()
)

# ------------------------------------------------------------
# TARGET DISTRIBUTIONS
# ------------------------------------------------------------

print("\nTARGET SUMMARY")
print("-" * 70)

for target in target_columns:

    if target not in ml_df.columns:
        continue

    print(f"\n{target}")

    if pd.api.types.is_numeric_dtype(ml_df[target]):

        print(
            f"  min  = {ml_df[target].min():.4f}"
        )

        print(
            f"  max  = {ml_df[target].max():.4f}"
        )

        print(
            f"  mean = {ml_df[target].mean():.4f}"
        )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

ml_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("ML DATASET READY")
print("=" * 70)

print(f"\nRows saved    : {len(ml_df)}")
print(f"Columns saved : {len(ml_df.columns)}")
print(f"Output        : {OUTPUT_FILE}")

print("\nNext step:")
print("Train grouped ML surrogate models.")