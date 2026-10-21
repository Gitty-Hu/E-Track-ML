import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# INDIA 2035 FLEET ECONOMICS MODEL
# E-Track ML Project
# ============================================================

print("=" * 70)
print("INDIA 2035 FLEET ECONOMICS MODEL")
print("=" * 70)


# ============================================================
# FILES
# ============================================================

DEMAND_FILE = Path(
    "data/processed/demand_2035_scenarios.csv"
)

VEHICLE_FILE = Path(
    "data/processed/vehicles_final.csv"
)

OUTPUT_FILE = Path(
    "data/processed/fleet_economics_2035.csv"
)

SUMMARY_FILE = Path(
    "data/processed/fleet_economics_2035_summary.csv"
)


# ============================================================
# GENERAL ASSUMPTIONS
# ============================================================

OPERATING_DAYS = 330

MAX_VEHICLE_OPERATING_HOURS_PER_DAY = 16

DEADHEADING_RATE = 0.10

PLATFORM_COMMISSION = 0.20

DRIVER_COST_PER_DAY = 900

INSURANCE_RATE = 0.025

VEHICLE_LIFE_YEARS = 8

RESIDUAL_VALUE_RATE = 0.20


MAINTENANCE_RATE = {

    "BEV": 0.025,
    "CNG": 0.035,
    "Petrol": 0.040,
    "Diesel": 0.040,
    "Hybrid": 0.035

}


# ============================================================
# BASE ENERGY PRICES
# ============================================================

BASE_ENERGY_PRICE = {

    "BEV": 4.50,
    "CNG": 97.00,
    "Petrol": 111.68,
    "Diesel": 99.19,
    "Hybrid": 111.68

}


# ============================================================
# 2035 ENERGY PRICE SCENARIOS
# ============================================================

ENERGY_SCENARIOS = {

    "low_energy_cost": 1.00,

    "central_energy_cost": 1.25,

    "high_energy_cost": 1.50

}


# ============================================================
# 2035 FARE SCENARIOS
# ============================================================

FARE_SCENARIOS = {

    "low_fare_growth": 1.25,

    "central_fare_growth": 1.40,

    "high_fare_growth": 1.60

}


# ============================================================
# LOAD DATA
# ============================================================

if not DEMAND_FILE.exists():

    raise FileNotFoundError(
        f"Demand file not found: {DEMAND_FILE}"
    )


if not VEHICLE_FILE.exists():

    raise FileNotFoundError(
        f"Vehicle file not found: {VEHICLE_FILE}"
    )


demand = pd.read_csv(
    DEMAND_FILE
)

vehicles = pd.read_csv(
    VEHICLE_FILE
)


print(
    f"\nDemand records: {len(demand)}"
)

print(
    f"Vehicle records: {len(vehicles)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_demand = [

    "vehicle_id",
    "scenario_id",
    "scenario_name",
    "demand_scenario",
    "trip_distance_km",
    "trip_time_min",
    "trips_per_day_2035",
    "passengers_per_trip_2035",
    "luggage_kg_per_trip",
    "daily_distance_km_2035",
    "annual_distance_km_2035",
    "daily_passenger_km_2035",
    "annual_passenger_km_2035",
    "daily_operating_hours_2035"

]


missing_demand = [

    c for c in required_demand
    if c not in demand.columns

]


if missing_demand:

    raise ValueError(
        f"Missing demand columns: {missing_demand}"
    )


required_vehicle = [

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
    "purchase_price_inr"

]


missing_vehicle = [

    c for c in required_vehicle
    if c not in vehicles.columns

]


if missing_vehicle:

    raise ValueError(
        f"Missing vehicle columns: {missing_vehicle}"
    )


# ============================================================
# VEHICLE MASTER
# ============================================================

vehicle_master = (

    vehicles[
        required_vehicle
    ]
    .drop_duplicates(
        "vehicle_id"
    )
)


# ============================================================
# IMPORTANT:
# REMOVE VEHICLE MASTER COLUMNS ALREADY PRESENT IN DEMAND
# ============================================================
#
# The demand file may contain copies of vehicle specifications.
# We remove them first.
#
# vehicles_final.csv is the authoritative vehicle source.
# ============================================================

vehicle_columns_to_replace = [

    c for c in required_vehicle
    if c != "vehicle_id"
    and c in demand.columns

]


if vehicle_columns_to_replace:

    print(
        "\nReplacing overlapping vehicle columns:"
    )

    for c in vehicle_columns_to_replace:

        print(
            f"  - {c}"
        )


    demand = demand.drop(
        columns=vehicle_columns_to_replace
    )


# ============================================================
# MERGE VEHICLE MASTER
# ============================================================

demand = demand.merge(

    vehicle_master,

    on="vehicle_id",

    how="left",

    validate="many_to_one"

)


# ============================================================
# VALIDATE MERGE
# ============================================================

unmatched = (

    demand["manufacturer"]
    .isna()
    .sum()

)


print(
    f"\nUnmatched vehicle records: {unmatched}"
)


if unmatched > 0:

    missing_ids = (

        demand.loc[
            demand["manufacturer"].isna(),
            "vehicle_id"
        ]
        .unique()
        .tolist()

    )

    raise ValueError(
        f"Vehicle IDs could not be matched: {missing_ids}"
    )


missing_prices = (

    demand["purchase_price_inr"]
    .isna()
    .sum()

)


print(
    f"Missing purchase prices: {missing_prices}"
)


if missing_prices > 0:

    raise ValueError(
        "Purchase prices are missing."
    )


# ============================================================
# HELPER: FIND EFFICIENCY COLUMN
# ============================================================

def find_column(
    dataframe,
    candidates
):

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    return None


cng_efficiency_col = find_column(

    demand,

    [
        "cng_efficiency_km_per_kg",
        "cng_km_per_kg"
    ]

)


fuel_efficiency_col = find_column(

    demand,

    [
        "fuel_efficiency_km_per_l",
        "fuel_consumption_km_per_l",
        "petrol_efficiency_km_per_l",
        "diesel_efficiency_km_per_l"
    ]

)


# ============================================================
# LOAD ORIGINAL SIMULATION FOR ENERGY EFFICIENCY
# ============================================================
#
# This is important.
#
# We do NOT estimate BEV/CNG/petrol efficiency purely from
# price. We recover the actual efficiency relationship from
# the existing simulation dataset.
# ============================================================

SIMULATION_FILE = Path(

    "data/processed/vehicle_scenario_dataset.csv"

)


simulation = pd.read_csv(

    SIMULATION_FILE

)


simulation_columns = [

    "vehicle_id",
    "energy_cost_inr_per_km"

]


for c in simulation_columns:

    if c not in simulation.columns:

        raise ValueError(
            f"Simulation file missing: {c}"
        )


# One vehicle has the same physical energy efficiency across
# the seven duty cycles in our current model, so take the mean.

vehicle_energy = (

    simulation
    .groupby(
        "vehicle_id",
        as_index=False
    )
    .agg(

        energy_cost_inr_per_km=(
            "energy_cost_inr_per_km",
            "mean"
        )

    )

)


demand = demand.merge(

    vehicle_energy,

    on="vehicle_id",

    how="left",

    validate="many_to_one"

)


# ============================================================
# DERIVE PHYSICAL ENERGY QUANTITY
# ============================================================

def derive_energy_per_km(
    row
):

    powertrain = row["powertrain"]

    simulation_cost = (
        row["energy_cost_inr_per_km"]
    )


    if pd.isna(simulation_cost):

        return np.nan


    price = BASE_ENERGY_PRICE[
        powertrain
    ]


    if price <= 0:

        return np.nan


    return (

        simulation_cost
        / price

    )


demand["energy_quantity_per_km"] = (

    demand.apply(
        derive_energy_per_km,
        axis=1
    )

)


# ============================================================
# RECORD GENERATION
# ============================================================

records = []


for _, row in demand.iterrows():

    powertrain = row["powertrain"]


    base_energy_price = (

        BASE_ENERGY_PRICE[
            powertrain
        ]

    )


    energy_per_km = (

        row["energy_quantity_per_km"]

    )


    if pd.isna(energy_per_km):

        raise ValueError(

            f"Could not determine energy efficiency "
            f"for vehicle {row['vehicle_id']}."

        )


    # ========================================================
    # ENERGY SCENARIOS
    # ========================================================

    for energy_scenario, energy_multiplier in (

        ENERGY_SCENARIOS.items()

    ):

        energy_price_2035 = (

            base_energy_price
            * energy_multiplier

        )


        energy_cost_per_km = (

            energy_per_km
            * energy_price_2035

        )


        # ====================================================
        # FARE SCENARIOS
        # ====================================================

        for fare_scenario, fare_multiplier in (

            FARE_SCENARIOS.items()

        ):


            trip_type = row[
                "scenario_name"
            ]


            # ------------------------------------------------
            # BASE FARE
            # ------------------------------------------------
            #
            # These are modeling anchors based on the existing
            # economics dataset, not official 2035 fares.
            # ------------------------------------------------

            if trip_type in [

                "urban_short",
                "urban_commute",
                "urban_mixed"

            ]:

                base_fare = 27.0


            elif trip_type == "airport":

                base_fare = 31.5


            elif trip_type in [

                "inter_city",
                "long_distance"

            ]:

                base_fare = 31.5


            else:

                base_fare = 31.5


            fare_2035 = (

                base_fare
                * fare_multiplier

            )


            # =================================================
            # REVENUE
            # =================================================

            annual_revenue = (

                row["annual_distance_km_2035"]
                * fare_2035

            )


            # =================================================
            # DEADHEADING
            # =================================================

            deadhead_distance = (

                row["annual_distance_km_2035"]
                * DEADHEADING_RATE

            )


            total_vehicle_distance = (

                row["annual_distance_km_2035"]
                + deadhead_distance

            )


            # =================================================
            # ENERGY
            # =================================================

            annual_energy_cost = (

                total_vehicle_distance
                * energy_cost_per_km

            )


            # =================================================
            # MAINTENANCE
            # =================================================

            maintenance_rate = (

                MAINTENANCE_RATE[
                    powertrain
                ]

            )


            annual_maintenance = (

                row["purchase_price_inr"]
                * maintenance_rate

            )


            # =================================================
            # INSURANCE
            # =================================================

            annual_insurance = (

                row["purchase_price_inr"]
                * INSURANCE_RATE

            )


            # =================================================
            # DEPRECIATION
            # =================================================

            annual_depreciation = (

                row["purchase_price_inr"]
                * (1 - RESIDUAL_VALUE_RATE)
                / VEHICLE_LIFE_YEARS

            )


            # =================================================
            # DRIVER / SHIFT REQUIREMENT
            # =================================================

            operating_hours = (

                row[
                    "daily_operating_hours_2035"
                ]

            )


            vehicle_shifts_required = max(

                1,

                int(

                    np.ceil(

                        operating_hours
                        / MAX_VEHICLE_OPERATING_HOURS_PER_DAY

                    )

                )

            )


            annual_driver_cost = (

                DRIVER_COST_PER_DAY
                * OPERATING_DAYS
                * vehicle_shifts_required

            )


            # =================================================
            # PLATFORM
            # =================================================

            annual_platform_cost = (

                annual_revenue
                * PLATFORM_COMMISSION

            )


            # =================================================
            # TOTAL COST
            # =================================================

            annual_operating_cost = (

                annual_energy_cost

                + annual_maintenance

                + annual_insurance

                + annual_depreciation

                + annual_driver_cost

                + annual_platform_cost

            )


            # =================================================
            # PROFIT
            # =================================================

            annual_profit = (

                annual_revenue
                - annual_operating_cost

            )


            # =================================================
            # PROFIT METRICS
            # =================================================

            if annual_revenue > 0:

                profit_margin = (

                    annual_profit
                    / annual_revenue
                    * 100

                )

            else:

                profit_margin = np.nan


            profit_per_km = (

                annual_profit
                / total_vehicle_distance

            )


            annual_passenger_km = (

                row[
                    "annual_passenger_km_2035"
                ]

            )


            if annual_passenger_km > 0:

                profit_per_passenger_km = (

                    annual_profit
                    / annual_passenger_km

                )

            else:

                profit_per_passenger_km = np.nan


            # =================================================
            # RANGE
            # =================================================

            if (

                powertrain == "BEV"

                and pd.notna(
                    row["certified_range_km"]
                )

                and row["certified_range_km"] > 0

            ):

                usable_range = (

                    row["certified_range_km"]
                    * 0.80

                )

                range_type = (
                    "80_percent_certified_range"
                )

                range_margin = (

                    usable_range
                    - row[
                        "daily_distance_km_2035"
                    ]

                )

                range_feasible = (

                    row[
                        "daily_distance_km_2035"
                    ]
                    <= usable_range

                )

            else:

                usable_range = np.inf

                range_type = (
                    "refueling_assumed"
                )

                range_margin = np.inf

                range_feasible = True


            # =================================================
            # CO2
            # =================================================

            if powertrain == "BEV":

                co2_factor = 0.70


            elif powertrain == "CNG":

                co2_factor = 2.75


            elif powertrain == "Diesel":

                co2_factor = 2.68


            else:

                co2_factor = 2.31


            annual_energy_quantity = (

                total_vehicle_distance
                * energy_per_km

            )


            annual_co2_kg = (

                annual_energy_quantity
                * co2_factor

            )


            # =================================================
            # SEAT UTILIZATION
            # =================================================

            if (

                pd.notna(row["seats"])
                and row["seats"] > 0

            ):

                seat_utilization = (

                    row[
                        "passengers_per_trip_2035"
                    ]
                    / row["seats"]
                    * 100

                )

            else:

                seat_utilization = np.nan


            # =================================================
            # LUGGAGE
            # =================================================

            luggage_volume = (

                row["luggage_volume_l"]

            )

            luggage_requirement = (

                row[
                    "luggage_kg_per_trip"
                ]

            )


            if (

                pd.notna(luggage_volume)
                and luggage_volume > 0

            ):

                luggage_capacity_proxy = (

                    luggage_volume
                    * 0.25

                )

                luggage_feasible = (

                    luggage_capacity_proxy
                    >= luggage_requirement

                )

            else:

                luggage_capacity_proxy = np.nan

                luggage_feasible = np.nan


            # =================================================
            # RECORD
            # =================================================

            records.append({

                "vehicle_id":
                    row["vehicle_id"],

                "manufacturer":
                    row["manufacturer"],

                "model":
                    row["model"],

                "model_year":
                    row["model_year"],

                "powertrain":
                    powertrain,

                "vehicle_type":
                    row["vehicle_type"],

                "seats":
                    row["seats"],

                "kerb_weight_kg":
                    row["kerb_weight_kg"],

                "payload_kg":
                    row["payload_kg"],

                "wheelbase_mm":
                    row["wheelbase_mm"],

                "ground_clearance_mm":
                    row["ground_clearance_mm"],

                "luggage_volume_l":
                    luggage_volume,

                "battery_capacity_kwh":
                    row["battery_capacity_kwh"],

                "certified_range_km":
                    row["certified_range_km"],

                "power_kw":
                    row["power_kw"],

                "purchase_price_inr":
                    row["purchase_price_inr"],

                "city":
                    row["city"],

                "scenario_id":
                    row["scenario_id"],

                "scenario_name":
                    trip_type,

                "demand_scenario":
                    row["demand_scenario"],

                "energy_scenario":
                    energy_scenario,

                "fare_scenario":
                    fare_scenario,

                "trip_distance_km":
                    row["trip_distance_km"],

                "trips_per_day_2035":
                    row["trips_per_day_2035"],

                "passengers_per_trip_2035":
                    row[
                        "passengers_per_trip_2035"
                    ],

                "luggage_kg_per_trip":
                    luggage_requirement,

                "daily_distance_km_2035":
                    row[
                        "daily_distance_km_2035"
                    ],

                "annual_distance_km_2035":
                    row[
                        "annual_distance_km_2035"
                    ],

                "annual_passenger_km_2035":
                    annual_passenger_km,

                "daily_operating_hours_2035":
                    operating_hours,

                "vehicle_shifts_required":
                    vehicle_shifts_required,

                "base_energy_price_2026":
                    base_energy_price,

                "energy_price_2035":
                    energy_price_2035,

                "energy_quantity_per_km":
                    energy_per_km,

                "energy_cost_inr_per_km_2035":
                    energy_cost_per_km,

                "base_fare_per_km":
                    base_fare,

                "fare_per_km_2035":
                    fare_2035,

                "annual_revenue_inr":
                    annual_revenue,

                "deadhead_distance_km":
                    deadhead_distance,

                "total_vehicle_distance_km":
                    total_vehicle_distance,

                "annual_energy_cost_inr":
                    annual_energy_cost,

                "annual_maintenance_cost_inr":
                    annual_maintenance,

                "annual_insurance_cost_inr":
                    annual_insurance,

                "annual_depreciation_cost_inr":
                    annual_depreciation,

                "annual_driver_cost_inr":
                    annual_driver_cost,

                "annual_platform_cost_inr":
                    annual_platform_cost,

                "annual_operating_cost_inr":
                    annual_operating_cost,

                "annual_profit_inr":
                    annual_profit,

                "profit_margin_percent":
                    profit_margin,

                "profit_per_vehicle_km_inr":
                    profit_per_km,

                "profit_per_passenger_km_inr":
                    profit_per_passenger_km,

                "usable_range_km":
                    usable_range,

                "range_margin_km":
                    range_margin,

                "range_feasible":
                    range_feasible,

                "annual_co2_kg":
                    annual_co2_kg,

                "annual_co2_tonnes":
                    annual_co2_kg / 1000,

                "seat_utilization_percent":
                    seat_utilization,

                "luggage_capacity_proxy_kg":
                    luggage_capacity_proxy,

                "luggage_feasible":
                    luggage_feasible

            })


# ============================================================
# CREATE DATAFRAME
# ============================================================

result = pd.DataFrame(
    records
)


# ============================================================
# VALIDATE RECORD COUNT
# ============================================================

expected_records = (

    len(demand)
    * len(ENERGY_SCENARIOS)
    * len(FARE_SCENARIOS)

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
# SAVE DETAILED DATA
# ============================================================

result.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# VEHICLE SUMMARY
# ============================================================

summary = (

    result
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

        "purchase_price_inr":
            "first",

        "seats":
            "first",

        "kerb_weight_kg":
            "first",

        "payload_kg":
            "first",

        "wheelbase_mm":
            "first",

        "luggage_volume_l":
            "first",

        "battery_capacity_kwh":
            "first",

        "certified_range_km":
            "first",

        "annual_revenue_inr":
            "mean",

        "annual_operating_cost_inr":
            "mean",

        "annual_profit_inr":
            "mean",

        "profit_margin_percent":
            "mean",

        "profit_per_vehicle_km_inr":
            "mean",

        "profit_per_passenger_km_inr":
            "mean",

        "annual_co2_tonnes":
            "mean",

        "seat_utilization_percent":
            "mean",

        "range_feasible":
            "mean",

        "luggage_feasible":
            "mean",

        "vehicle_shifts_required":
            "mean"

    })

)


summary = summary.sort_values(

    "annual_profit_inr",

    ascending=False

)


summary.to_csv(

    SUMMARY_FILE,

    index=False

)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("2035 FLEET ECONOMICS COMPLETE")
print("=" * 70)


print(
    f"\nEconomic records: {len(result)}"
)

print(
    f"Vehicle records : {len(summary)}"
)


print("\nAverage annual revenue:")

print(
    f"Rs {result['annual_revenue_inr'].mean():,.0f}"
)


print("\nAverage annual operating cost:")

print(
    f"Rs {result['annual_operating_cost_inr'].mean():,.0f}"
)


print("\nAverage annual profit:")

print(
    f"Rs {result['annual_profit_inr'].mean():,.0f}"
)


print("\nAverage profit margin:")

print(
    f"{result['profit_margin_percent'].mean():.2f}%"
)


print("\nAverage annual CO2:")

print(
    f"{result['annual_co2_tonnes'].mean():.2f} tonnes"
)


# ============================================================
# DEMAND SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DEMAND SCENARIO ECONOMICS")
print("=" * 70)


demand_summary = (

    result
    .groupby(
        "demand_scenario"
    )
    .agg(

        avg_revenue=(
            "annual_revenue_inr",
            "mean"
        ),

        avg_cost=(
            "annual_operating_cost_inr",
            "mean"
        ),

        avg_profit=(
            "annual_profit_inr",
            "mean"
        ),

        avg_margin=(
            "profit_margin_percent",
            "mean"
        )

    )

)


print(
    demand_summary.to_string()
)


# ============================================================
# ENERGY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ENERGY PRICE SCENARIO ECONOMICS")
print("=" * 70)


energy_summary = (

    result
    .groupby(
        "energy_scenario"
    )
    .agg(

        avg_revenue=(
            "annual_revenue_inr",
            "mean"
        ),

        avg_energy_cost=(
            "annual_energy_cost_inr",
            "mean"
        ),

        avg_profit=(
            "annual_profit_inr",
            "mean"
        ),

        avg_margin=(
            "profit_margin_percent",
            "mean"
        )

    )

)


print(
    energy_summary.to_string()
)


# ============================================================
# FARE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FARE SCENARIO ECONOMICS")
print("=" * 70)


fare_summary = (

    result
    .groupby(
        "fare_scenario"
    )
    .agg(

        avg_revenue=(
            "annual_revenue_inr",
            "mean"
        ),

        avg_profit=(
            "annual_profit_inr",
            "mean"
        ),

        avg_margin=(
            "profit_margin_percent",
            "mean"
        )

    )

)


print(
    fare_summary.to_string()
)


# ============================================================
# POWERTRAIN SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("POWERTRAIN SUMMARY")
print("=" * 70)


powertrain_summary = (

    result
    .groupby(
        "powertrain"
    )
    .agg(

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

        avg_energy_cost=(
            "annual_energy_cost_inr",
            "mean"
        ),

        avg_purchase_price=(
            "purchase_price_inr",
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


# ============================================================
# TOP VEHICLES
# ============================================================

print("\n" + "=" * 70)
print("TOP VEHICLES BY 2035 MODELED PROFIT")
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


# ============================================================
# RANGE FEASIBILITY
# ============================================================

print("\n" + "=" * 70)
print("RANGE FEASIBILITY")
print("=" * 70)


range_result = (

    result
    .groupby(
        "powertrain"
    )[
        "range_feasible"
    ]
    .mean()
    * 100

)


print(
    range_result.to_string()
)


# ============================================================
# SHIFT REQUIREMENT
# ============================================================

print("\n" + "=" * 70)
print("VEHICLE SHIFT REQUIREMENT")
print("=" * 70)


shift_result = (

    result
    .groupby(
        "demand_scenario"
    )[
        "vehicle_shifts_required"
    ]
    .mean()

)


print(
    shift_result.to_string()
)


# ============================================================
# FILES
# ============================================================

print("\nFiles created:")

print(
    f"  {OUTPUT_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)


print("\nMODEL NOTES:")

print(
    "Vehicle specifications are sourced from "
    "vehicles_final.csv."
)

print(
    "Vehicle energy efficiency is recovered from "
    "the existing commercial simulation."
)

print(
    "2035 demand multipliers are engineering scenarios."
)

print(
    "2035 energy-price multipliers are sensitivity scenarios."
)

print(
    "2035 fare multipliers are sensitivity scenarios."
)

print(
    "NITI Aayog transport scenarios provide macro context, "
    "not specific ride-hailing fare forecasts."
)

print(
    "BEV usable range is approximated at 80 percent of "
    "certified range."
)

print(
    "Vehicle shifts are introduced when daily operating "
    "hours exceed 16 hours."
)

print(
    "This remains a scenario model; it is not the final "
    "optimized vehicle."
)


print("=" * 70)