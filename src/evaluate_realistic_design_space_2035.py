import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("=" * 70)
print("INDIA 2035 REALISTIC DESIGN-SPACE ML EVALUATION")
print("=" * 70)

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT = BASE_DIR / "data" / "processed" / "design_space_realistic_2035.csv"
OUTPUT = BASE_DIR / "data" / "processed" / "design_space_realistic_evaluated_2035.csv"

ENERGY_MODEL = BASE_DIR / "models" / "energy_consumption_model.pkl"
COMFORT_MODEL = BASE_DIR / "models" / "comfort_index_model.pkl"

df = pd.read_csv(INPUT)

print(f"Input realistic designs: {len(df):,}")

# ============================================================
# 2035 CENTRAL COMMERCIAL DUTY CYCLE
# ============================================================

TRIPS_PER_DAY = 10.351807
PASSENGERS_PER_TRIP = 2.968
LUGGAGE_PER_TRIP_KG = 20.0
DAILY_DISTANCE_KM = 313.121343
ANNUAL_WORKING_DAYS = 330

TRIP_DISTANCE_KM = DAILY_DISTANCE_KM / TRIPS_PER_DAY
DAILY_OPERATING_HOURS = TRIPS_PER_DAY * 45 / 60

df["trip_distance_km"] = TRIP_DISTANCE_KM
df["trips_per_day_2035"] = TRIPS_PER_DAY
df["passengers_per_trip_2035"] = PASSENGERS_PER_TRIP
df["luggage_kg_per_trip"] = LUGGAGE_PER_TRIP_KG
df["daily_distance_km_2035"] = DAILY_DISTANCE_KM
df["annual_distance_km_2035"] = (
    DAILY_DISTANCE_KM * ANNUAL_WORKING_DAYS
)
df["daily_operating_hours_2035"] = DAILY_OPERATING_HOURS

df["scenario_name"] = "mixed_commercial"
df["vehicle_type"] = df["vehicle_type"].fillna("Passenger Vehicle")

# ============================================================
# LOAD ML MODELS
# ============================================================

energy_model = joblib.load(ENERGY_MODEL)
comfort_model = joblib.load(COMFORT_MODEL)

print("Energy ML model loaded.")
print("Comfort ML model loaded.")

# ============================================================
# ENERGY ML PREDICTION
# ============================================================

energy_numeric = [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "battery_capacity_kwh",
    "certified_range_km",
    "power_kw",
    "purchase_price_inr",
    "trip_distance_km",
    "trips_per_day_2035",
    "passengers_per_trip_2035",
    "luggage_kg_per_trip",
    "daily_distance_km_2035",
    "daily_operating_hours_2035"
]

energy_categorical = [
    "powertrain",
    "vehicle_type",
    "scenario_name"
]

X_energy = df[energy_numeric + energy_categorical].copy()

df["predicted_energy_quantity_per_km"] = (
    energy_model.predict(X_energy)
)

# ============================================================
# COMFORT ML PREDICTION
# ============================================================

comfort_numeric = [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
    "trip_distance_km",
    "trips_per_day_2035",
    "passengers_per_trip_2035",
    "luggage_kg_per_trip",
    "daily_distance_km_2035",
    "daily_operating_hours_2035"
]

comfort_categorical = [
    "powertrain",
    "vehicle_type",
    "scenario_name"
]

X_comfort = df[comfort_numeric + comfort_categorical].copy()

df["predicted_comfort_index"] = (
    comfort_model.predict(X_comfort)
)

# ============================================================
# ENERGY PRICES
# ============================================================

energy_price = {
    "BEV": 4.50,
    "CNG": 97.00,
    "Hybrid": 111.68,
    "Petrol": 111.68
}

df["energy_price_2035"] = df["powertrain"].map(energy_price)

df["energy_cost_inr_per_km_2035"] = (
    df["predicted_energy_quantity_per_km"]
    * df["energy_price_2035"]
)

# ============================================================
# REVENUE
# ============================================================

FARE_PER_KM = 30.0

df["annual_revenue_inr"] = (
    df["annual_distance_km_2035"]
    * FARE_PER_KM
)

# ============================================================
# COST MODEL
# ============================================================

# Energy
df["annual_energy_cost_inr"] = (
    df["annual_distance_km_2035"]
    * df["energy_cost_inr_per_km_2035"]
)

# Maintenance
maintenance_rate = {
    "BEV": 0.025,
    "CNG": 0.035,
    "Petrol": 0.040,
    "Hybrid": 0.035
}

df["annual_maintenance_cost_inr"] = (
    df["purchase_price_inr"]
    * df["powertrain"].map(maintenance_rate)
)

# Insurance
df["annual_insurance_cost_inr"] = (
    df["purchase_price_inr"] * 0.025
)

# Depreciation
RESIDUAL_VALUE = 0.20
LIFE_YEARS = 8

df["annual_depreciation_cost_inr"] = (
    df["purchase_price_inr"]
    * (1 - RESIDUAL_VALUE)
    / LIFE_YEARS
)

# Driver
DRIVER_COST_PER_DAY = 900

df["annual_driver_cost_inr"] = (
    DRIVER_COST_PER_DAY * ANNUAL_WORKING_DAYS
)

# Platform commission
PLATFORM_COMMISSION = 0.20

df["annual_platform_cost_inr"] = (
    df["annual_revenue_inr"]
    * PLATFORM_COMMISSION
)

# Total cost
df["annual_operating_cost_inr"] = (
    df["annual_energy_cost_inr"]
    + df["annual_maintenance_cost_inr"]
    + df["annual_insurance_cost_inr"]
    + df["annual_depreciation_cost_inr"]
    + df["annual_driver_cost_inr"]
    + df["annual_platform_cost_inr"]
)

# Profit
df["annual_profit_inr"] = (
    df["annual_revenue_inr"]
    - df["annual_operating_cost_inr"]
)

df["profit_margin_percent"] = (
    df["annual_profit_inr"]
    / df["annual_revenue_inr"]
    * 100
)

# ============================================================
# CO2
# ============================================================

co2_factor = {
    "BEV": 0.70,
    "CNG": 2.75,
    "Hybrid": 2.31,
    "Petrol": 2.31
}

df["annual_co2_kg"] = (
    df["annual_distance_km_2035"]
    * df["predicted_energy_quantity_per_km"]
    * df["powertrain"].map(co2_factor)
)

df["annual_co2_tonnes"] = (
    df["annual_co2_kg"] / 1000
)

# ============================================================
# BEV RANGE FEASIBILITY
# ============================================================

df["usable_range_km"] = (
    df["certified_range_km"] * 0.80
)

df["range_margin_km"] = (
    df["usable_range_km"]
    - DAILY_DISTANCE_KM
)

df["range_feasible"] = np.where(
    df["powertrain"] == "BEV",
    df["usable_range_km"] >= DAILY_DISTANCE_KM,
    True
)

# ============================================================
# LUGGAGE FEASIBILITY
# ============================================================

df["luggage_capacity_proxy_kg"] = (
    df["luggage_volume_l"] * 0.25
)

df["luggage_feasible"] = (
    df["luggage_capacity_proxy_kg"]
    >= LUGGAGE_PER_TRIP_KG
)

# ============================================================
# POWER-TO-WEIGHT
# ============================================================

df["power_to_weight_kw_per_tonne"] = (
    df["power_kw"]
    / df["kerb_weight_kg"]
    * 1000
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ML EVALUATION RESULTS")
print("=" * 70)

print(f"Evaluated designs : {len(df):,}")

print()
print("Predicted energy:")
print(
    df.groupby("powertrain")
    ["predicted_energy_quantity_per_km"]
    .mean()
)

print()
print("Comfort:")
print(
    df.groupby("powertrain")
    ["predicted_comfort_index"]
    .mean()
)

print()
print("Energy cost/km:")
print(
    df.groupby("powertrain")
    ["energy_cost_inr_per_km_2035"]
    .mean()
)

print()
print("Annual profit:")
print(
    df.groupby("powertrain")
    ["annual_profit_inr"]
    .mean()
)

print()
print("Annual CO2:")
print(
    df.groupby("powertrain")
    ["annual_co2_tonnes"]
    .mean()
)

print()
print("Range feasibility:")
print(
    df.groupby("powertrain")
    ["range_feasible"]
    .mean()
)

# ============================================================
# RANGES
# ============================================================

print()
print("-" * 70)
print("OVERALL DESIGN RANGES")
print("-" * 70)

for col in [
    "predicted_energy_quantity_per_km",
    "energy_cost_inr_per_km_2035",
    "predicted_comfort_index",
    "annual_profit_inr",
    "annual_co2_tonnes",
    "purchase_price_inr"
]:
    print(
        f"{col:40s}: "
        f"{df[col].min():,.3f} – {df[col].max():,.3f}"
    )

# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT, index=False)

print()
print("=" * 70)
print("REALISTIC DESIGN-SPACE EVALUATION COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(f"  {OUTPUT}")

print()
print("Next step:")
print("Run the final NSGA-II optimization on this evaluated")
print("realistic 2035 design space.")