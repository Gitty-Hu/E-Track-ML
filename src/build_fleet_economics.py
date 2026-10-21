import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FLEET ECONOMICS MODEL
# E-Track - Ideal Ride Service Vehicle for India 2035
# ============================================================

print("=" * 70)
print("FLEET ECONOMICS MODEL")
print("=" * 70)


# ============================================================
# FILES
# ============================================================

SIMULATION_FILE = Path(
    "data/processed/vehicle_scenario_dataset.csv"
)

VEHICLE_FILE = Path(
    "data/processed/vehicles_final.csv"
)

OUTPUT_SCENARIO = Path(
    "data/processed/fleet_economics_scenario.csv"
)

OUTPUT_SUMMARY = Path(
    "data/processed/fleet_economics_summary.csv"
)


# ============================================================
# BASELINE ASSUMPTIONS
# ============================================================

VEHICLE_LIFE_YEARS = 8

OPERATING_DAYS_PER_YEAR = 330

PLATFORM_COMMISSION = 0.20

DRIVER_COST_PER_DAY = 900

DEADHEADING_RATE = 0.10

INSURANCE_RATE = 0.025

RESIDUAL_VALUE_RATE = 0.20


MAINTENANCE_RATE = {
    "BEV": 0.025,
    "CNG": 0.035,
    "Petrol": 0.040,
    "Diesel": 0.040,
    "Hybrid": 0.035,
}


# ============================================================
# ENERGY / EMISSION PARAMETERS
# ============================================================

ELECTRICITY_PRICE = 4.50
CNG_PRICE = 97.00
PETROL_PRICE = 111.68
DIESEL_PRICE = 99.19


CO2_KG_PER_KWH_ELECTRICITY = 0.70
CO2_KG_PER_KG_CNG = 2.75
CO2_KG_PER_LITRE_PETROL = 2.31
CO2_KG_PER_LITRE_DIESEL = 2.68


# ============================================================
# LOAD FILES
# ============================================================

if not SIMULATION_FILE.exists():

    raise FileNotFoundError(
        f"Simulation file not found: {SIMULATION_FILE}"
    )


if not VEHICLE_FILE.exists():

    raise FileNotFoundError(
        f"Vehicle file not found: {VEHICLE_FILE}"
    )


df = pd.read_csv(
    SIMULATION_FILE
)

vehicles = pd.read_csv(
    VEHICLE_FILE
)


print(
    f"\nSimulation records: {len(df)}"
)

print(
    f"Vehicle records: {len(vehicles)}"
)


# ============================================================
# STATIC VEHICLE DATA
# ============================================================
#
# All vehicle specifications come from vehicles_final.csv.
#
# This prevents the simulation file and vehicle master file
# from getting out of sync.
# ============================================================

static_columns = [
    "vehicle_id",
    "manufacturer",
    "model",
    "model_year",
    "powertrain",
    "vehicle_type",
    "seats",
    "kerb_weight_kg",
    "payload_kg",
    "wheelbase_mm",
    "ground_clearance_mm",
    "luggage_volume_l",
    "battery_capacity_kwh",
    "fuel_tank_l",
    "certified_range_km",
    "power_kw",
    "charging_power_kw",
    "charging_time_min",
    "purchase_price_inr",
]


missing_static = [
    c
    for c in static_columns
    if c not in vehicles.columns
]


if missing_static:

    print("\nMissing vehicle master columns:")

    for c in missing_static:

        print(
            "  -",
            c
        )

    raise ValueError(
        "vehicles_final.csv is missing required "
        "vehicle specification columns."
    )


vehicle_master = (

    vehicles[
        static_columns
    ]
    .drop_duplicates(
        "vehicle_id"
    )

)


# ============================================================
# MERGE MASTER DATA
# ============================================================

# Remove any static columns accidentally present in
# simulation data before merging.

columns_to_remove = [
    c
    for c in static_columns
    if c != "vehicle_id"
    and c in df.columns
]


df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)


df = df.merge(
    vehicle_master,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)


# ============================================================
# VALIDATION
# ============================================================

missing_vehicle_ids = (
    df["manufacturer"].isna()
).sum()


if missing_vehicle_ids:

    raise ValueError(
        f"{missing_vehicle_ids} simulation rows "
        "could not be matched to vehicles_final.csv."
    )


missing_prices = (
    df["purchase_price_inr"].isna()
).sum()


print(
    f"\nMissing purchase prices: {missing_prices}"
)


if missing_prices:

    raise ValueError(
        "Purchase prices are missing."
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [

    "purchase_price_inr",

    "energy_cost_inr_per_km",

    "annual_distance_km",

    "annual_revenue_inr",

    "trip_distance_km",

    "trips_per_day",

    "typical_passengers",

    "seats",

    "kerb_weight_kg",

    "payload_kg",

    "luggage_volume_l",

]


for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 1. DEADHEADING
# ============================================================

df["deadhead_distance_km"] = (

    df["annual_distance_km"]
    * DEADHEADING_RATE

)


df["total_vehicle_distance_km"] = (

    df["annual_distance_km"]
    + df["deadhead_distance_km"]

)


# ============================================================
# 2. ENERGY COST
# ============================================================

df["annual_energy_cost_inr_adjusted"] = (

    df["total_vehicle_distance_km"]
    * df["energy_cost_inr_per_km"]

)


# ============================================================
# 3. MAINTENANCE
# ============================================================

df["maintenance_rate"] = (

    df["powertrain"]
    .map(MAINTENANCE_RATE)
    .fillna(0.035)

)


df["annual_maintenance_cost_inr"] = (

    df["purchase_price_inr"]
    * df["maintenance_rate"]

)


# ============================================================
# 4. INSURANCE
# ============================================================

df["annual_insurance_cost_inr"] = (

    df["purchase_price_inr"]
    * INSURANCE_RATE

)


# ============================================================
# 5. DEPRECIATION
# ============================================================

depreciable_amount = (

    df["purchase_price_inr"]
    * (1 - RESIDUAL_VALUE_RATE)

)


df["annual_depreciation_cost_inr"] = (

    depreciable_amount
    / VEHICLE_LIFE_YEARS

)


# ============================================================
# 6. DRIVER COST
# ============================================================

df["annual_driver_cost_inr"] = (

    DRIVER_COST_PER_DAY
    * OPERATING_DAYS_PER_YEAR

)


# ============================================================
# 7. PLATFORM COMMISSION
# ============================================================

df["annual_platform_cost_inr"] = (

    df["annual_revenue_inr"]
    * PLATFORM_COMMISSION

)


# ============================================================
# 8. TOTAL OPERATING COST
# ============================================================

df["annual_operating_cost_inr"] = (

    df["annual_energy_cost_inr_adjusted"]

    + df["annual_maintenance_cost_inr"]

    + df["annual_insurance_cost_inr"]

    + df["annual_depreciation_cost_inr"]

    + df["annual_driver_cost_inr"]

    + df["annual_platform_cost_inr"]

)


# ============================================================
# 9. ANNUAL PROFIT
# ============================================================

df["annual_profit_inr"] = (

    df["annual_revenue_inr"]
    - df["annual_operating_cost_inr"]

)


# ============================================================
# 10. PROFIT / KM
# ============================================================

df["profit_per_vehicle_km_inr"] = np.where(

    df["total_vehicle_distance_km"] > 0,

    df["annual_profit_inr"]
    / df["total_vehicle_distance_km"],

    np.nan

)


# ============================================================
# 11. PROFIT MARGIN
# ============================================================

df["profit_margin_percent"] = np.where(

    df["annual_revenue_inr"] > 0,

    100
    * df["annual_profit_inr"]
    / df["annual_revenue_inr"],

    np.nan

)


# ============================================================
# 12. REVENUE / KM
# ============================================================

df["revenue_per_vehicle_km_inr"] = (

    df["annual_revenue_inr"]
    / df["total_vehicle_distance_km"]

)


# ============================================================
# 13. BREAK-EVEN REVENUE / KM
# ============================================================

df["break_even_revenue_per_km_inr"] = (

    df["annual_operating_cost_inr"]
    / df["total_vehicle_distance_km"]

)


# ============================================================
# 14. PASSENGER-KM
# ============================================================

if "annual_passenger_km" in df.columns:

    df["annual_passenger_km_adjusted"] = (

        df["annual_passenger_km"]

    )

else:

    df["annual_passenger_km_adjusted"] = (

        df["annual_distance_km"]
        * df["typical_passengers"]

    )


# ============================================================
# 15. PROFIT / PASSENGER-KM
# ============================================================

df["profit_per_passenger_km_inr"] = np.where(

    df["annual_passenger_km_adjusted"] > 0,

    df["annual_profit_inr"]
    / df["annual_passenger_km_adjusted"],

    np.nan

)


# ============================================================
# 16. CO2
# ============================================================

def calculate_co2(row):

    powertrain = str(
        row["powertrain"]
    )

    distance = (
        row["total_vehicle_distance_km"]
    )

    energy_cost = (
        row["energy_cost_inr_per_km"]
    )


    if powertrain == "BEV":

        kwh_per_km = (

            energy_cost
            / ELECTRICITY_PRICE

        )

        return (

            distance
            * kwh_per_km
            * CO2_KG_PER_KWH_ELECTRICITY

        )


    elif powertrain == "CNG":

        kg_per_km = (

            energy_cost
            / CNG_PRICE

        )

        return (

            distance
            * kg_per_km
            * CO2_KG_PER_KG_CNG

        )


    elif powertrain == "Diesel":

        litres_per_km = (

            energy_cost
            / DIESEL_PRICE

        )

        return (

            distance
            * litres_per_km
            * CO2_KG_PER_LITRE_DIESEL

        )


    else:

        litres_per_km = (

            energy_cost
            / PETROL_PRICE

        )

        return (

            distance
            * litres_per_km
            * CO2_KG_PER_LITRE_PETROL

        )


df["annual_co2_kg"] = (

    df.apply(
        calculate_co2,
        axis=1
    )

)


df["annual_co2_tonnes"] = (

    df["annual_co2_kg"]
    / 1000

)


# ============================================================
# 17. SEAT UTILIZATION
# ============================================================

df["seat_utilization_percent"] = np.where(

    df["seats"] > 0,

    100
    * df["typical_passengers"]
    / df["seats"],

    np.nan

)


# ============================================================
# 18. DAILY DISTANCE
# ============================================================

df["annual_operating_days"] = (

    OPERATING_DAYS_PER_YEAR

)


df["average_daily_distance_km"] = (

    df["total_vehicle_distance_km"]
    / OPERATING_DAYS_PER_YEAR

)


# ============================================================
# 19. RANGE SAFETY MARGIN
# ============================================================

if "range_margin_km" in df.columns:

    df["range_safety_margin_km"] = (

        df["range_margin_km"]

    )

else:

    df["range_safety_margin_km"] = np.nan


# ============================================================
# 20. COMMERCIAL EFFICIENCY INDEX
# ============================================================
#
# Diagnostic only.
# NOT the final optimization objective.
# ============================================================

def normalize(series):

    minimum = series.min()

    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index
        )

    return (

        (series - minimum)
        / (maximum - minimum)

    )


profit_norm = normalize(
    df["annual_profit_inr"]
)


energy_norm = normalize(
    df["energy_cost_inr_per_km"]
)


co2_norm = normalize(
    df["annual_co2_kg"]
)


df["commercial_efficiency_index"] = (

    0.60 * profit_norm

    + 0.20 * (1 - energy_norm)

    + 0.20 * (1 - co2_norm)

)


# ============================================================
# SAVE SCENARIO DATA
# ============================================================

df.to_csv(

    OUTPUT_SCENARIO,
    index=False

)


# ============================================================
# VEHICLE SUMMARY
# ============================================================

summary = (

    df
    .groupby(
        [
            "vehicle_id",
            "manufacturer",
            "model",
            "powertrain"
        ],
        as_index=False
    )
    .agg({

        "model_year": "first",

        "vehicle_type": "first",

        "seats": "first",

        "kerb_weight_kg": "first",

        "payload_kg": "first",

        "wheelbase_mm": "first",

        "ground_clearance_mm": "first",

        "luggage_volume_l": "first",

        "battery_capacity_kwh": "first",

        "fuel_tank_l": "first",

        "certified_range_km": "first",

        "power_kw": "first",

        "charging_power_kw": "first",

        "charging_time_min": "first",

        "purchase_price_inr": "first",

        "annual_distance_km": "mean",

        "total_vehicle_distance_km": "mean",

        "annual_revenue_inr": "mean",

        "annual_energy_cost_inr_adjusted": "mean",

        "annual_maintenance_cost_inr": "mean",

        "annual_insurance_cost_inr": "mean",

        "annual_depreciation_cost_inr": "mean",

        "annual_driver_cost_inr": "mean",

        "annual_platform_cost_inr": "mean",

        "annual_operating_cost_inr": "mean",

        "annual_profit_inr": "mean",

        "profit_margin_percent": "mean",

        "profit_per_vehicle_km_inr": "mean",

        "profit_per_passenger_km_inr": "mean",

        "break_even_revenue_per_km_inr": "mean",

        "annual_co2_kg": "mean",

        "annual_co2_tonnes": "mean",

        "seat_utilization_percent": "mean",

        "commercial_efficiency_index": "mean",

    })

)


# ============================================================
# SORT
# ============================================================

summary = summary.sort_values(

    "annual_profit_inr",
    ascending=False

)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(

    OUTPUT_SUMMARY,
    index=False

)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("FLEET ECONOMICS COMPLETE")
print("=" * 70)


print(
    f"\nScenario records: {len(df)}"
)


print(
    f"Vehicle records : {len(summary)}"
)


print("\nAverage annual revenue:")

print(
    f"Rs {summary['annual_revenue_inr'].mean():,.0f}"
)


print("\nAverage annual operating cost:")

print(
    f"Rs {summary['annual_operating_cost_inr'].mean():,.0f}"
)


print("\nAverage annual profit:")

print(
    f"Rs {summary['annual_profit_inr'].mean():,.0f}"
)


print("\nAverage profit margin:")

print(
    f"{summary['profit_margin_percent'].mean():.2f}%"
)


print("\n" + "=" * 70)
print("TOP VEHICLES BY MODELED ANNUAL PROFIT")
print("=" * 70)


print(

    summary[
        [
            "vehicle_id",
            "model",
            "powertrain",
            "purchase_price_inr",
            "annual_profit_inr",
            "profit_margin_percent",
            "annual_co2_tonnes"
        ]
    ]
    .head(15)
    .to_string(index=False)

)


print("\n" + "=" * 70)
print("POWERTRAIN SUMMARY")
print("=" * 70)


powertrain_summary = (

    summary
    .groupby("powertrain")
    .agg(

        vehicles=(
            "vehicle_id",
            "count"
        ),

        avg_purchase_price=(
            "purchase_price_inr",
            "mean"
        ),

        avg_profit=(
            "annual_profit_inr",
            "mean"
        ),

        avg_margin=(
            "profit_margin_percent",
            "mean"
        ),

        avg_co2_tonnes=(
            "annual_co2_tonnes",
            "mean"
        ),

        avg_profit_per_km=(
            "profit_per_vehicle_km_inr",
            "mean"
        )

    )
    .sort_values(
        "avg_profit",
        ascending=False
    )

)


print(
    powertrain_summary.to_string()
)


print("\nFiles created:")

print(
    f"  {OUTPUT_SCENARIO}"
)

print(
    f"  {OUTPUT_SUMMARY}"
)


print("\nMODEL NOTES:")

print(
    "Vehicle specifications and purchase prices come "
    "from vehicles_final.csv."
)

print(
    "Fleet economics are baseline modeled estimates."
)

print(
    "Current energy prices and existing fare references "
    "are baseline inputs."
)

print(
    "2035 scenarios will be introduced separately."
)

print(
    "The commercial efficiency index is diagnostic only."
)

print("=" * 70)