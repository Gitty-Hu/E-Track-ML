import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# INDIA 2035 RIDE-SERVICE DEMAND MODEL
# E-Track ML Project
# ============================================================
#
# Purpose:
# Convert the current engineering duty-cycle scenarios into
# transparent 2035 commercial scenarios.
#
# IMPORTANT:
# The multipliers below are engineering scenario assumptions.
# They are NOT presented as official NITI Aayog forecasts.
#
# NITI Aayog 2026 transport work is used as the macro anchor:
# - passenger mobility demand rises with economic development
# - road remains a major passenger mode
# - public/shared road mobility becomes more important
# - transport demand differs between policy pathways
#
# ============================================================


print("=" * 70)
print("INDIA 2035 RIDE-SERVICE DEMAND MODEL")
print("=" * 70)


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "data/processed/vehicle_scenario_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/demand_2035_scenarios.csv"
)

SUMMARY_FILE = Path(
    "data/processed/demand_2035_summary.csv"
)


# ============================================================
# MODEL YEARS
# ============================================================

BASE_YEAR = 2026

TARGET_YEAR = 2035

YEARS_TO_2035 = TARGET_YEAR - BASE_YEAR


# ============================================================
# OPERATING ASSUMPTIONS
# ============================================================

OPERATING_DAYS_PER_YEAR = 330


# ============================================================
# 2035 DEMAND SCENARIOS
# ============================================================
#
# These are engineering scenarios, not official forecasts.
#
# Conservative:
# Slower growth in ride-service utilization.
#
# Central:
# Moderate growth in shared commercial mobility.
#
# High shared mobility:
# Stronger utilization of taxis / ride-service vehicles.
#


DEMAND_SCENARIOS = {

    "conservative": {

        "demand_multiplier": 1.15,

        "passenger_multiplier": 1.03,

        "utilization_multiplier": 1.08,

        "description":
            "Slower growth in commercial ride-service demand."

    },


    "central": {

        "demand_multiplier": 1.30,

        "passenger_multiplier": 1.06,

        "utilization_multiplier": 1.15,

        "description":
            "Moderate growth in shared commercial mobility."

    },


    "high_shared_mobility": {

        "demand_multiplier": 1.50,

        "passenger_multiplier": 1.10,

        "utilization_multiplier": 1.25,

        "description":
            "Higher growth in shared/taxi utilization."

    }

}


# ============================================================
# SCENARIO-SPECIFIC CHANGES
# ============================================================
#
# Different trip types should not all grow identically.
#
# Urban mobility is affected by metro/public transport.
# Airport and inter-city services remain important.
# Long-distance road trips may grow more slowly because of
# rail substitution.
#


TRIP_TYPE_FACTORS = {

    "urban_short": {

        "conservative": 1.00,
        "central": 1.02,
        "high_shared_mobility": 1.05

    },

    "urban_commute": {

        "conservative": 1.00,
        "central": 1.03,
        "high_shared_mobility": 1.08

    },

    "urban_mixed": {

        "conservative": 1.02,
        "central": 1.05,
        "high_shared_mobility": 1.10

    },

    "airport": {

        "conservative": 1.05,
        "central": 1.12,
        "high_shared_mobility": 1.20

    },

    "inter_city": {

        "conservative": 0.98,
        "central": 1.02,
        "high_shared_mobility": 1.08

    },

    "long_distance": {

        "conservative": 0.95,
        "central": 1.00,
        "high_shared_mobility": 1.05

    },

    "mixed_commercial": {

        "conservative": 1.05,
        "central": 1.12,
        "high_shared_mobility": 1.20

    }

}


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)


print(
    f"\nInput records: {len(df)}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [

    "vehicle_id",
    "manufacturer",
    "model",
    "powertrain",
    "scenario_id",
    "scenario_name",
    "city",
    "trip_distance_km",
    "trip_time_min",
    "typical_passengers",
    "typical_luggage_kg",
    "trips_per_day",
    "daily_distance_km"

]


missing = [

    c for c in required_columns
    if c not in df.columns

]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CREATE 2035 SCENARIOS
# ============================================================

records = []


for scenario_name, scenario_params in DEMAND_SCENARIOS.items():

    demand_multiplier = (
        scenario_params["demand_multiplier"]
    )

    passenger_multiplier = (
        scenario_params["passenger_multiplier"]
    )

    utilization_multiplier = (
        scenario_params["utilization_multiplier"]
    )


    for _, row in df.iterrows():

        trip_type = row["scenario_name"]


        if trip_type not in TRIP_TYPE_FACTORS:

            raise ValueError(
                f"Unknown trip type: {trip_type}"
            )


        trip_factor = (

            TRIP_TYPE_FACTORS[
                trip_type
            ][
                scenario_name
            ]

        )


        # ----------------------------------------------------
        # 2035 TRIPS / DAY
        # ----------------------------------------------------

        trips_2035 = (

            row["trips_per_day"]

            * demand_multiplier

            * trip_factor

            * utilization_multiplier

        )


        # ----------------------------------------------------
        # 2035 PASSENGERS
        # ----------------------------------------------------

        passengers_2035 = (

            row["typical_passengers"]

            * passenger_multiplier

        )


        # Commercial vehicle capacity cannot be exceeded.
        #
        # For vehicles with a valid seat count, cap passenger
        # utilization at the available seating capacity.

        if pd.notna(row.get("seats", np.nan)):

            passengers_2035 = min(

                passengers_2035,
                row["seats"]

            )


        # ----------------------------------------------------
        # DISTANCE PER TRIP
        # ----------------------------------------------------

        distance_per_trip = (

            row["trip_distance_km"]

        )


        # ----------------------------------------------------
        # DAILY DISTANCE
        # ----------------------------------------------------

        daily_distance_2035 = (

            trips_2035
            * distance_per_trip

        )


        # ----------------------------------------------------
        # ANNUAL DISTANCE
        # ----------------------------------------------------

        annual_distance_2035 = (

            daily_distance_2035
            * OPERATING_DAYS_PER_YEAR

        )


        # ----------------------------------------------------
        # PASSENGER-KM
        # ----------------------------------------------------

        passenger_km_per_trip = (

            distance_per_trip
            * passengers_2035

        )


        daily_passenger_km = (

            passenger_km_per_trip
            * trips_2035

        )


        annual_passenger_km = (

            daily_passenger_km
            * OPERATING_DAYS_PER_YEAR

        )


        # ----------------------------------------------------
        # LUGGAGE
        # ----------------------------------------------------

        luggage_per_trip = (

            row["typical_luggage_kg"]

        )


        # ----------------------------------------------------
        # OPERATING HOURS
        # ----------------------------------------------------

        daily_operating_minutes = (

            trips_2035
            * row["trip_time_min"]

        )


        daily_operating_hours = (

            daily_operating_minutes
            / 60

        )


        # ----------------------------------------------------
        # VEHICLE UTILIZATION
        # ----------------------------------------------------

        utilization_hours_per_day = (

            daily_operating_hours

        )


        # Approximate vehicle-km utilization relative to
        # the original commercial scenario.

        utilization_km_per_day = (

            daily_distance_2035

        )


        records.append({

            "vehicle_id":
                row["vehicle_id"],

            "manufacturer":
                row["manufacturer"],

            "model":
                row["model"],

            "powertrain":
                row["powertrain"],

            "vehicle_type":
                row.get("vehicle_type", np.nan),

            "seats":
                row.get("seats", np.nan),

            "city":
                row["city"],

            "scenario_id":
                row["scenario_id"],

            "scenario_name":
                trip_type,

            "demand_scenario":
                scenario_name,

            "scenario_description":
                scenario_params["description"],

            "base_trips_per_day":
                row["trips_per_day"],

            "demand_multiplier":
                demand_multiplier,

            "trip_type_factor":
                trip_factor,

            "utilization_multiplier":
                utilization_multiplier,

            "passenger_multiplier":
                passenger_multiplier,

            "trips_per_day_2035":
                trips_2035,

            "trip_distance_km":
                distance_per_trip,

            "trip_time_min":
                row["trip_time_min"],

            "passengers_per_trip_2035":
                passengers_2035,

            "luggage_kg_per_trip":
                luggage_per_trip,

            "daily_distance_km_2035":
                daily_distance_2035,

            "annual_distance_km_2035":
                annual_distance_2035,

            "daily_passenger_km_2035":
                daily_passenger_km,

            "annual_passenger_km_2035":
                annual_passenger_km,

            "daily_operating_hours_2035":
                utilization_hours_per_day,

            "daily_operating_minutes_2035":
                daily_operating_minutes,

            "operating_days_per_year":
                OPERATING_DAYS_PER_YEAR

        })


# ============================================================
# DATAFRAME
# ============================================================

result = pd.DataFrame(
    records
)


# ============================================================
# SANITY CHECK
# ============================================================

expected_records = (

    len(df)
    * len(DEMAND_SCENARIOS)

)


print(
    f"\nExpected records: {expected_records}"
)

print(
    f"Actual records  : {len(result)}"
)


if len(result) != expected_records:

    raise ValueError(
        "Record count mismatch."
    )


# ============================================================
# SAVE DETAILED DATASET
# ============================================================

result.to_csv(

    OUTPUT_FILE,
    index=False

)


# ============================================================
# SUMMARY BY DEMAND SCENARIO
# ============================================================

summary = (

    result
    .groupby(
        "demand_scenario",
        as_index=False
    )
    .agg({

        "trips_per_day_2035":
            "mean",

        "daily_distance_km_2035":
            "mean",

        "annual_distance_km_2035":
            "mean",

        "daily_passenger_km_2035":
            "mean",

        "annual_passenger_km_2035":
            "mean",

        "daily_operating_hours_2035":
            "mean",

        "passengers_per_trip_2035":
            "mean"

    })

)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(

    SUMMARY_FILE,
    index=False

)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("2035 DEMAND MODEL COMPLETE")
print("=" * 70)


print("\nDemand scenario summary:")


display_columns = [

    "demand_scenario",
    "trips_per_day_2035",
    "daily_distance_km_2035",
    "annual_distance_km_2035",
    "daily_passenger_km_2035",
    "annual_passenger_km_2035",
    "daily_operating_hours_2035",
    "passengers_per_trip_2035"

]


print(

    summary[
        display_columns
    ].to_string(
        index=False
    )

)


# ============================================================
# TRIP TYPE SUMMARY
# ============================================================

trip_summary = (

    result
    .groupby(
        [
            "demand_scenario",
            "scenario_name"
        ],
        as_index=False
    )
    .agg({

        "trips_per_day_2035":
            "mean",

        "daily_distance_km_2035":
            "mean",

        "annual_distance_km_2035":
            "mean",

        "annual_passenger_km_2035":
            "mean"

    })

)


print("\n" + "=" * 70)
print("2035 DEMAND BY TRIP TYPE")
print("=" * 70)


print(

    trip_summary.to_string(
        index=False
    )

)


# ============================================================
# TOTAL SYSTEM ACTIVITY
# ============================================================

system_summary = (

    result
    .groupby(
        "demand_scenario"
    )
    .agg(

        total_vehicle_km_2035=(
            "annual_distance_km_2035",
            "sum"
        ),

        total_passenger_km_2035=(
            "annual_passenger_km_2035",
            "sum"
        ),

        average_trips_per_day=(
            "trips_per_day_2035",
            "mean"
        )

    )

)


print("\n" + "=" * 70)
print("TOTAL MODELED SYSTEM ACTIVITY")
print("=" * 70)


print(

    system_summary.to_string()

)


print("\nFiles created:")

print(
    f"  {OUTPUT_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)


print("\nMODEL NOTES:")

print(
    "2035 multipliers are transparent engineering scenarios."
)

print(
    "They are not official NITI Aayog forecasts."
)

print(
    "NITI Aayog transport scenarios provide the macro "
    "direction for increasing mobility demand and "
    "greater public/shared transport relevance."
)

print(
    "The existing seven duty cycles are retained."
)

print(
    "Trip-specific factors account for differences between "
    "urban, airport, inter-city and long-distance mobility."
)

print(
    "This dataset will be used for 2035 fleet economics "
    "and subsequent optimization."
)


print("=" * 70)