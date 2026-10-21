import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# INDIA 2035 - HYPOTHETICAL VEHICLE DESIGN EVALUATION
# ============================================================
# Purpose:
# Evaluate hypothetical 2035 vehicle designs using the
# previously trained ML surrogate models.
#
# Models used:
#   1. Energy consumption ML model
#   2. Passenger comfort & practicality ML model
#
# Additional engineering calculations:
#   - Range feasibility
#   - Energy cost
#   - CO2 emissions
#   - Daily / annual commercial activity
#
# Output:
#   data/processed/design_space_evaluated.csv
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DESIGN_FILE = BASE_DIR / "data" / "processed" / "design_space_2035.csv"

ENERGY_MODEL_FILE = BASE_DIR / "models" / "energy_consumption_model.pkl"
COMFORT_MODEL_FILE = BASE_DIR / "models" / "comfort_index_model.pkl"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "design_space_evaluated.csv"


# ============================================================
# SETTINGS
# ============================================================

# 2035 central commercial-duty-cycle assumptions
TRIPS_PER_DAY = 10.351807
PASSENGERS_PER_TRIP = 2.968
LUGGAGE_PER_TRIP_KG = 20.0

DAILY_DISTANCE_KM = 313.121343
ANNUAL_WORKING_DAYS = 330

# 2035 energy-price assumption
# Based on the central 2035 scenario used previously.
#
# These are modelling assumptions, not forecasts.
ENERGY_PRICES = {
    "BEV": 4.50 * 1.00,       # Rs/kWh
    "CNG": 97.00 * 1.00,      # Rs/kg
    "Petrol": 111.68 * 1.00,  # Rs/L
    "Hybrid": 111.68 * 1.00  # Rs/L
}

# Electricity grid emission factor
ELECTRICITY_CO2_KG_PER_KWH = 0.70

# Fuel emission factors
PETROL_CO2_KG_PER_L = 2.31
CNG_CO2_KG_PER_KG = 2.75

# Approximate usable fraction of certified BEV range.
# This accounts for operating reserve rather than assuming
# the entire certified range is practically usable.
BEV_USABLE_RANGE_FACTOR = 0.80

# Minimum desired practical range margin.
# We require the vehicle to cover the full daily duty cycle
# with some operational reserve.
MIN_RANGE_MARGIN_KM = 0.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_columns(df, required_columns, name):
    """Check that required columns exist."""

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"\n{name} is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def safe_numeric(series):
    """Convert a pandas series to numeric safely."""
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INDIA 2035 HYPOTHETICAL VEHICLE DESIGN EVALUATION")
print("=" * 70)

print("\nLoading design space...")

designs = pd.read_csv(DESIGN_FILE)

print(f"Design records loaded: {len(designs):,}")


# ============================================================
# LOAD ML MODELS
# ============================================================

print("\nLoading ML surrogate models...")

energy_model = joblib.load(ENERGY_MODEL_FILE)
comfort_model = joblib.load(COMFORT_MODEL_FILE)

print("Energy model loaded successfully.")
print("Comfort model loaded successfully.")


# ============================================================
# CHECK DESIGN DATA
# ============================================================

required_design_columns = [
    "design_id",
    "powertrain",
    "vehicle_type",
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
]

check_columns(
    designs,
    required_design_columns,
    "design_space_2035.csv"
)


# ============================================================
# NUMERIC CLEANUP
# ============================================================

numeric_columns = [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
    "battery_capacity_kwh",
    "certified_range_km",
]

for column in numeric_columns:

    if column in designs.columns:
        designs[column] = safe_numeric(designs[column])


# ============================================================
# CREATE MODEL INPUT DATA
# ============================================================

print("\nPreparing ML input features...")


# ------------------------------------------------------------
# ENERGY MODEL FEATURES
# ------------------------------------------------------------
#
# These correspond to the features used during training.
# ------------------------------------------------------------

energy_numeric_features = [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "battery_capacity_kwh",
    "certified_range_km",
    "power_kw",
    "purchase_price_inr",
]

energy_categorical_features = [
    "powertrain",
    "vehicle_type",
    "scenario_name",
]


# ------------------------------------------------------------
# COMFORT MODEL FEATURES
# ------------------------------------------------------------

comfort_numeric_features = [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
]

comfort_categorical_features = [
    "powertrain",
    "vehicle_type",
    "scenario_name",
]


# ============================================================
# ADD 2035 DUTY-CYCLE FEATURES
# ============================================================

# The trained models expect these variables because the original
# training data contained commercial operating conditions.

designs["trip_distance_km"] = DAILY_DISTANCE_KM / TRIPS_PER_DAY

designs["trips_per_day_2035"] = TRIPS_PER_DAY

designs["passengers_per_trip_2035"] = PASSENGERS_PER_TRIP

designs["luggage_kg_per_trip"] = LUGGAGE_PER_TRIP_KG

designs["daily_distance_km_2035"] = DAILY_DISTANCE_KM

designs["daily_operating_hours_2035"] = (
    TRIPS_PER_DAY * 45.0 / 60.0
)


# ============================================================
# SCENARIO NAME
# ============================================================

# The ML models were trained using scenario_name.
# Use a representative central commercial scenario.

designs["scenario_name"] = "mixed_commercial"


# ============================================================
# CHECK ALL MODEL FEATURES
# ============================================================

energy_features = (
    energy_numeric_features
    + [
        "trip_distance_km",
        "trips_per_day_2035",
        "passengers_per_trip_2035",
        "luggage_kg_per_trip",
        "daily_distance_km_2035",
        "daily_operating_hours_2035",
    ]
    + energy_categorical_features
)

comfort_features = (
    comfort_numeric_features
    + [
        "trip_distance_km",
        "trips_per_day_2035",
        "passengers_per_trip_2035",
        "luggage_kg_per_trip",
        "daily_distance_km_2035",
        "daily_operating_hours_2035",
    ]
    + comfort_categorical_features
)

check_columns(
    designs,
    energy_features,
    "Energy model input"
)

check_columns(
    designs,
    comfort_features,
    "Comfort model input"
)


# ============================================================
# HANDLE BEV-ONLY BATTERY/RANGE VARIABLES
# ============================================================
#
# The hypothetical design space contains NaN battery/range
# values for non-BEV vehicles.
#
# The trained sklearn pipeline cannot always accept missing
# values, so use neutral placeholders only for ML prediction.
#
# IMPORTANT:
# These placeholders are NOT interpreted as actual vehicle
# specifications.
# ============================================================

energy_input = designs[energy_features].copy()

energy_input["battery_capacity_kwh"] = (
    energy_input["battery_capacity_kwh"].fillna(0.0)
)

energy_input["certified_range_km"] = (
    energy_input["certified_range_km"].fillna(0.0)
)


# ============================================================
# ML PREDICTION - ENERGY
# ============================================================

print("\nPredicting energy consumption...")

designs["predicted_energy_quantity_per_km"] = (
    energy_model.predict(energy_input)
)

# Avoid physically impossible negative predictions.
designs["predicted_energy_quantity_per_km"] = (
    designs["predicted_energy_quantity_per_km"].clip(lower=0)
)


# ============================================================
# ML PREDICTION - COMFORT
# ============================================================

print("Predicting passenger comfort & practicality...")

comfort_input = designs[comfort_features].copy()

designs["predicted_comfort_index"] = (
    comfort_model.predict(comfort_input)
)

# Keep comfort score within the engineering index scale.
designs["predicted_comfort_index"] = (
    designs["predicted_comfort_index"].clip(0, 100)
)


# ============================================================
# ENERGY CONVERSION
# ============================================================
#
# Native ML output:
#
# BEV     -> kWh/km
# CNG     -> kg/km
# Petrol  -> L/km
# Hybrid  -> L/km
#
# For comparison and emissions calculations we create:
#
# energy_kwh_equivalent_per_km
#
# Fuel lower heating values / approximate conversion factors:
#
# Petrol: 1 L ≈ 9.0 kWh
# CNG:    1 kg ≈ 13.9 kWh
#
# These are engineering approximations used only to place
# different energy carriers on a common energy scale.
# ============================================================

def convert_to_kwh_equivalent(row):

    powertrain = row["powertrain"]
    energy = row["predicted_energy_quantity_per_km"]

    if powertrain == "BEV":
        return energy

    elif powertrain == "CNG":
        return energy * 13.9

    elif powertrain in ["Petrol", "Hybrid"]:
        return energy * 9.0

    else:
        return np.nan


designs["energy_kwh_equivalent_per_km"] = (
    designs.apply(convert_to_kwh_equivalent, axis=1)
)


# ============================================================
# ENERGY COST
# ============================================================

print("Calculating energy costs...")

def calculate_energy_cost(row):

    powertrain = row["powertrain"]
    energy = row["predicted_energy_quantity_per_km"]

    if powertrain not in ENERGY_PRICES:
        return np.nan

    return energy * ENERGY_PRICES[powertrain]


designs["energy_cost_inr_per_km_2035"] = (
    designs.apply(calculate_energy_cost, axis=1)
)


# ============================================================
# DAILY / ANNUAL ENERGY COST
# ============================================================

designs["daily_energy_cost_inr"] = (
    designs["energy_cost_inr_per_km_2035"]
    * designs["daily_distance_km_2035"]
)

designs["annual_energy_cost_inr"] = (
    designs["daily_energy_cost_inr"]
    * ANNUAL_WORKING_DAYS
)


# ============================================================
# ANNUAL DISTANCE
# ============================================================

designs["annual_distance_km_2035"] = (
    designs["daily_distance_km_2035"]
    * ANNUAL_WORKING_DAYS
)


# ============================================================
# ANNUAL PASSENGER-KM
# ============================================================

designs["annual_passenger_km_2035"] = (
    designs["annual_distance_km_2035"]
    * designs["passengers_per_trip_2035"]
)


# ============================================================
# RANGE FEASIBILITY
# ============================================================
#
# BEV:
#   usable range = certified range × 0.80
#
# For this central-duty-cycle evaluation, the vehicle should
# cover one full operating day.
#
# Non-BEV vehicles are treated as range-feasible because their
# energy replenishment infrastructure allows refuelling.
# ============================================================

designs["usable_range_km"] = np.nan

bev_mask = designs["powertrain"] == "BEV"

designs.loc[bev_mask, "usable_range_km"] = (
    designs.loc[bev_mask, "certified_range_km"]
    * BEV_USABLE_RANGE_FACTOR
)

designs["range_margin_km"] = (
    designs["usable_range_km"]
    - designs["daily_distance_km_2035"]
)


designs["range_feasible"] = 1

designs.loc[bev_mask, "range_feasible"] = (
    designs.loc[bev_mask, "range_margin_km"]
    >= MIN_RANGE_MARGIN_KM
).astype(int)


# ============================================================
# RANGE RESERVE RATIO
# ============================================================

designs["range_reserve_ratio"] = np.nan

designs.loc[bev_mask, "range_reserve_ratio"] = (
    designs.loc[bev_mask, "usable_range_km"]
    / designs.loc[bev_mask, "daily_distance_km_2035"]
)


# ============================================================
# LUGGAGE FEASIBILITY
# ============================================================
#
# Approximate luggage capacity requirement:
# 20 kg per passenger/trip.
#
# We use 1 litre ≈ 0.25 kg as a conservative luggage-density
# proxy, consistent with the comfort-index construction.
# ============================================================

designs["luggage_capacity_proxy_kg"] = (
    designs["luggage_volume_l"] * 0.25
)

designs["required_luggage_capacity_kg"] = (
    PASSENGERS_PER_TRIP * LUGGAGE_PER_TRIP_KG
)

designs["luggage_feasible"] = (
    designs["luggage_capacity_proxy_kg"]
    >= designs["required_luggage_capacity_kg"]
).astype(int)


# ============================================================
# SEAT FEASIBILITY
# ============================================================

designs["seat_feasible"] = (
    designs["seats"] >= PASSENGERS_PER_TRIP
).astype(int)


# ============================================================
# POWER-TO-WEIGHT
# ============================================================

designs["power_to_weight_kw_per_tonne"] = (
    designs["power_kw"]
    / (designs["kerb_weight_kg"] / 1000.0)
)


# ============================================================
# POWER-TO-WEIGHT FEASIBILITY
# ============================================================

designs["power_to_weight_feasible"] = (
    (designs["power_to_weight_kw_per_tonne"] >= 40)
    & (designs["power_to_weight_kw_per_tonne"] <= 120)
).astype(int)


# ============================================================
# CO2 EMISSIONS
# ============================================================

print("Calculating annual CO2 emissions...")

def calculate_co2_per_km(row):

    powertrain = row["powertrain"]
    energy = row["predicted_energy_quantity_per_km"]

    if powertrain == "BEV":
        return energy * ELECTRICITY_CO2_KG_PER_KWH

    elif powertrain == "CNG":
        return energy * CNG_CO2_KG_PER_KG

    elif powertrain in ["Petrol", "Hybrid"]:
        return energy * PETROL_CO2_KG_PER_L

    else:
        return np.nan


designs["co2_kg_per_km"] = (
    designs.apply(calculate_co2_per_km, axis=1)
)

designs["annual_co2_kg"] = (
    designs["co2_kg_per_km"]
    * designs["annual_distance_km_2035"]
)

designs["annual_co2_tonnes"] = (
    designs["annual_co2_kg"] / 1000.0
)


# ============================================================
# COMMERCIAL REVENUE
# ============================================================
#
# We use the central fare assumptions from the existing
# commercial model:
#
# Base average fare = Rs 30/km
#
# This is deliberately kept as a transparent modelling input.
# It is NOT presented as an official 2035 fare forecast.
# ============================================================

CENTRAL_FARE_PER_KM = 30.0

designs["fare_per_km_2035"] = CENTRAL_FARE_PER_KM

designs["annual_revenue_inr"] = (
    designs["annual_distance_km_2035"]
    * designs["fare_per_km_2035"]
)


# ============================================================
# BASIC COMMERCIAL PROFIT PROXY
# ============================================================
#
# This is NOT the final fleet economics model.
#
# It is included so that the design-space evaluation already
# contains a comparable commercial metric.
#
# Cost components:
#   Energy
#   Maintenance
#   Insurance
#   Depreciation
#   Driver
#   Platform commission
# ============================================================

maintenance_rate = {
    "BEV": 0.025,
    "CNG": 0.035,
    "Petrol": 0.040,
    "Hybrid": 0.035
}

designs["annual_maintenance_cost_inr"] = (
    designs.apply(
        lambda row:
            row["purchase_price_inr"]
            * maintenance_rate.get(row["powertrain"], 0.04),
        axis=1
    )
)

designs["annual_insurance_cost_inr"] = (
    designs["purchase_price_inr"] * 0.025
)


# Straight-line depreciation:
# 8-year life, 20% residual value

designs["annual_depreciation_cost_inr"] = (
    designs["purchase_price_inr"]
    * (1.0 - 0.20)
    / 8.0
)


# Driver cost
DRIVER_COST_PER_DAY = 900.0

designs["annual_driver_cost_inr"] = (
    DRIVER_COST_PER_DAY * ANNUAL_WORKING_DAYS
)


# Platform commission
PLATFORM_COMMISSION = 0.20

designs["annual_platform_cost_inr"] = (
    designs["annual_revenue_inr"]
    * PLATFORM_COMMISSION
)


# Total operating cost

designs["annual_operating_cost_inr"] = (
    designs["annual_energy_cost_inr"]
    + designs["annual_maintenance_cost_inr"]
    + designs["annual_insurance_cost_inr"]
    + designs["annual_depreciation_cost_inr"]
    + designs["annual_driver_cost_inr"]
    + designs["annual_platform_cost_inr"]
)


# ============================================================
# ANNUAL PROFIT
# ============================================================

designs["annual_profit_inr"] = (
    designs["annual_revenue_inr"]
    - designs["annual_operating_cost_inr"]
)


# ============================================================
# PROFIT MARGIN
# ============================================================

designs["profit_margin_percent"] = np.where(
    designs["annual_revenue_inr"] > 0,
    designs["annual_profit_inr"]
    / designs["annual_revenue_inr"]
    * 100.0,
    np.nan
)


# ============================================================
# PROFIT PER VEHICLE-KM
# ============================================================

designs["profit_per_vehicle_km_inr"] = (
    designs["annual_profit_inr"]
    / designs["annual_distance_km_2035"]
)


# ============================================================
# PROFIT PER PASSENGER-KM
# ============================================================

designs["profit_per_passenger_km_inr"] = (
    designs["annual_profit_inr"]
    / designs["annual_passenger_km_2035"]
)


# ============================================================
# OVERALL ENGINEERING FEASIBILITY
# ============================================================

designs["engineering_feasible"] = (
    (designs["range_feasible"] == 1)
    & (designs["luggage_feasible"] == 1)
    & (designs["seat_feasible"] == 1)
    & (designs["power_to_weight_feasible"] == 1)
).astype(int)


# ============================================================
# REMOVE INFINITE / INVALID VALUES
# ============================================================

designs = designs.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SAVE
# ============================================================

designs.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DESIGN SPACE EVALUATION COMPLETE")
print("=" * 70)

print(f"\nInput designs      : {len(designs):,}")
print(f"Output file        : {OUTPUT_FILE}")

print("\nEngineering feasibility:")
print(
    designs["engineering_feasible"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Not feasible",
        1: "Feasible"
    })
)

print("\nRange feasibility:")
print(
    designs.groupby("powertrain")["range_feasible"]
    .mean()
    .mul(100)
    .round(2)
    .astype(str)
    + " %"
)

print("\n------------------------------------------------------------")
print("PREDICTED ENERGY")
print("------------------------------------------------------------")

print(
    designs.groupby("powertrain")[
        "predicted_energy_quantity_per_km"
    ].mean().round(5)
)


print("\n------------------------------------------------------------")
print("ENERGY COST")
print("------------------------------------------------------------")

print(
    designs.groupby("powertrain")[
        "energy_cost_inr_per_km_2035"
    ].mean().round(3)
)


print("\n------------------------------------------------------------")
print("COMFORT INDEX")
print("------------------------------------------------------------")

print(
    designs.groupby("powertrain")[
        "predicted_comfort_index"
    ].mean().round(2)
)


print("\n------------------------------------------------------------")
print("ANNUAL PROFIT")
print("------------------------------------------------------------")

print(
    designs.groupby("powertrain")[
        "annual_profit_inr"
    ].mean().round(0)
)


print("\n------------------------------------------------------------")
print("ANNUAL CO2")
print("------------------------------------------------------------")

print(
    designs.groupby("powertrain")[
        "annual_co2_tonnes"
    ].mean().round(2)
)


print("\n------------------------------------------------------------")
print("DESIGN RANGES")
print("------------------------------------------------------------")

summary_columns = [
    "predicted_energy_quantity_per_km",
    "energy_kwh_equivalent_per_km",
    "energy_cost_inr_per_km_2035",
    "predicted_comfort_index",
    "annual_profit_inr",
    "annual_co2_tonnes",
    "power_to_weight_kw_per_tonne"
]

for column in summary_columns:

    print(
        f"{column:40s}: "
        f"{designs[column].min():.4f}"
        f" – "
        f"{designs[column].max():.4f}"
    )


# ============================================================
# TOP DESIGNS FOR INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 DESIGNS BY ANNUAL PROFIT")
print("=" * 70)

top_profit = (
    designs
    .sort_values("annual_profit_inr", ascending=False)
    .head(10)
)

print(
    top_profit[
        [
            "design_id",
            "powertrain",
            "seats",
            "kerb_weight_kg",
            "wheelbase_mm",
            "luggage_volume_l",
            "power_kw",
            "purchase_price_inr",
            "predicted_comfort_index",
            "energy_cost_inr_per_km_2035",
            "annual_profit_inr",
            "annual_co2_tonnes",
            "range_feasible"
        ]
    ].to_string(index=False)
)


# ============================================================
# TOP DESIGNS BY COMFORT
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 DESIGNS BY PREDICTED COMFORT")
print("=" * 70)

top_comfort = (
    designs
    .sort_values("predicted_comfort_index", ascending=False)
    .head(10)
)

print(
    top_comfort[
        [
            "design_id",
            "powertrain",
            "seats",
            "kerb_weight_kg",
            "wheelbase_mm",
            "luggage_volume_l",
            "power_kw",
            "predicted_comfort_index",
            "annual_profit_inr",
            "annual_co2_tonnes",
            "range_feasible"
        ]
    ].to_string(index=False)
)


# ============================================================
# TOP DESIGNS BY LOW CO2
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 DESIGNS BY LOWEST ANNUAL CO2")
print("=" * 70)

top_co2 = (
    designs
    .sort_values("annual_co2_tonnes", ascending=True)
    .head(10)
)

print(
    top_co2[
        [
            "design_id",
            "powertrain",
            "seats",
            "kerb_weight_kg",
            "battery_capacity_kwh",
            "certified_range_km",
            "energy_kwh_equivalent_per_km",
            "annual_co2_tonnes",
            "predicted_comfort_index",
            "annual_profit_inr",
            "range_feasible"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("NEXT STEP: INSPECT RESULTS BEFORE NSGA-II OPTIMIZATION")
print("=" * 70)