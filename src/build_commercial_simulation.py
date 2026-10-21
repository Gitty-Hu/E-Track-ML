import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# E-TRACK
# COMMERCIAL VEHICLE SIMULATION
# ============================================================

VEHICLE_FILE = Path("data/processed/vehicles_final.csv")
MOBILITY_FILE = Path("data/raw/mobility_raw.csv")
ENERGY_FILE = Path("data/raw/energy_raw.csv")
ECONOMICS_FILE = Path("data/raw/economics_raw.csv")

OUTPUT_FILE = Path("data/processed/vehicle_scenario_dataset.csv")
SUMMARY_FILE = Path("data/processed/vehicle_summary.csv")


# ============================================================
# LOAD DATA
# ============================================================

vehicles = pd.read_csv(VEHICLE_FILE)
mobility = pd.read_csv(MOBILITY_FILE)
energy = pd.read_csv(ENERGY_FILE)
economics = pd.read_csv(ECONOMICS_FILE)

print("=" * 70)
print("COMMERCIAL VEHICLE SIMULATION")
print("=" * 70)

print()
print("Vehicles :", len(vehicles))
print("Scenarios:", len(mobility))

print()
print("Mobility dataset columns:")
print(list(mobility.columns))

print()
print("Energy dataset columns:")
print(list(energy.columns))

print()
print("Economics dataset columns:")
print(list(economics.columns))


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_column(df, possible_names, required=True):

    for name in possible_names:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            "\nCould not identify required column.\n"
            f"Possible names tried: {possible_names}\n"
            f"Actual columns: {list(df.columns)}"
        )

    return None


# ============================================================
# MOBILITY COLUMN DETECTION
# ============================================================

scenario_id_col = find_column(
    mobility,
    [
        "scenario_id",
        "record_id",
        "id"
    ]
)

city_col = find_column(
    mobility,
    [
        "city",
        "location",
        "region"
    ]
)

scenario_name_col = find_column(
    mobility,
    [
        "scenario_name",
        "scenario",
        "duty_cycle",
        "trip_type"
    ]
)

distance_col = find_column(
    mobility,
    [
        "trip_distance_km",
        "distance_km",
        "trip_length_km",
        "distance",
        "trip_distance"
    ]
)

time_col = find_column(
    mobility,
    [
        "trip_time_min",
        "time_min",
        "trip_duration_min",
        "duration_min",
        "trip_time"
    ]
)

passenger_col = find_column(
    mobility,
    [
        "typical_passengers",
        "passengers",
        "average_passengers",
        "occupancy",
        "passenger_count"
    ]
)

luggage_col = find_column(
    mobility,
    [
        "typical_luggage_kg",
        "luggage_kg",
        "average_luggage_kg",
        "luggage_weight_kg"
    ]
)

trips_col = find_column(
    mobility,
    [
        "trips_per_day",
        "daily_trips",
        "number_of_trips_per_day",
        "trips_day"
    ]
)

daily_distance_col = find_column(
    mobility,
    [
        "daily_distance_km",
        "distance_per_day_km",
        "daily_km",
        "km_per_day"
    ]
)

print()
print("Detected mobility columns:")
print("Scenario ID     :", scenario_id_col)
print("City            :", city_col)
print("Scenario name   :", scenario_name_col)
print("Trip distance   :", distance_col)
print("Trip time       :", time_col)
print("Passengers      :", passenger_col)
print("Luggage         :", luggage_col)
print("Trips/day       :", trips_col)
print("Daily distance  :", daily_distance_col)


# ============================================================
# ENERGY COLUMN DETECTION
# ============================================================

energy_type_col = find_column(
    energy,
    [
        "energy_source",
        "energy_type",
        "fuel_type",
        "fuel",
        "source"
    ]
)

energy_price_col = find_column(
    energy,
    [
        "energy_price",
        "energy_price_inr_per_unit",
        "energy_price_inr",
        "price_inr",
        "price",
        "unit_price_inr",
        "cost_inr_per_unit"
    ]
)

print()
print("Detected energy type column:")
print(energy_type_col)

print()
print("Detected energy price column:")
print(energy_price_col)


# ============================================================
# BUILD ENERGY PRICE DICTIONARY
# ============================================================

energy_price = {}

for _, row in energy.iterrows():

    fuel = str(
        row[energy_type_col]
    ).strip().lower()

    price = pd.to_numeric(
        row[energy_price_col],
        errors="coerce"
    )

    if pd.notna(price):
        energy_price[fuel] = float(price)


# ============================================================
# PRICE LOOKUP
# ============================================================

def get_price(possible_names, default):

    for name in possible_names:

        if name.lower() in energy_price:
            return energy_price[name.lower()]

    for key, value in energy_price.items():

        for name in possible_names:

            if name.lower() in key:
                return value

    return default


electricity_price = get_price(
    [
        "electricity",
        "electric",
        "ev",
        "grid electricity"
    ],
    4.50
)

cng_price = get_price(
    ["cng"],
    97.00
)

petrol_price = get_price(
    ["petrol", "gasoline"],
    111.68
)

diesel_price = get_price(
    ["diesel"],
    99.19
)


print()
print("Energy-price references:")
print(f"Electricity : Rs {electricity_price:.2f}/kWh")
print(f"CNG         : Rs {cng_price:.2f}/kg")
print(f"Petrol      : Rs {petrol_price:.2f}/L")
print(f"Diesel      : Rs {diesel_price:.2f}/L")


# ============================================================
# ECONOMICS / FARE STRUCTURE
# ============================================================

fare_defaults = {

    "Go_AC": {
        "fare_min": 18,
        "fare_max": 36,
        "minimum_fare": 75,
        "minimum_distance_km": 4
    },

    "Sedan_Premier": {
        "fare_min": 21,
        "fare_max": 42,
        "minimum_fare": 100,
        "minimum_distance_km": 4
    },

    "XL": {
        "fare_min": 24,
        "fare_max": 48,
        "minimum_fare": 120,
        "minimum_distance_km": 4
    }
}


economics_category_col = find_column(
    economics,
    [
        "vehicle_category",
        "service_category",
        "category",
        "vehicle_segment",
        "vehicle_type"
    ],
    required=False
)

fare_min_col = find_column(
    economics,
    [
        "fare_min",
        "fare_min_per_km",
        "fare_min_inr_per_km",
        "minimum_fare_per_km"
    ],
    required=False
)

fare_max_col = find_column(
    economics,
    [
        "fare_max",
        "fare_max_per_km",
        "fare_max_inr_per_km",
        "maximum_fare_per_km"
    ],
    required=False
)

minimum_fare_col = find_column(
    economics,
    [
        "minimum_fare",
        "minimum_fare_inr"
    ],
    required=False
)

minimum_distance_col = find_column(
    economics,
    [
        "minimum_fare_distance_km",
        "minimum_distance_km"
    ],
    required=False
)


# ============================================================
# UPDATE FARES FROM ECONOMICS FILE
# ============================================================

if economics_category_col is not None:

    for _, row in economics.iterrows():

        category = str(
            row[economics_category_col]
        ).strip()

        if category not in fare_defaults:
            continue

        if (
            fare_min_col is not None
            and pd.notna(row[fare_min_col])
        ):
            fare_defaults[category]["fare_min"] = float(
                row[fare_min_col]
            )

        if (
            fare_max_col is not None
            and pd.notna(row[fare_max_col])
        ):
            fare_defaults[category]["fare_max"] = float(
                row[fare_max_col]
            )

        if (
            minimum_fare_col is not None
            and pd.notna(row[minimum_fare_col])
        ):
            fare_defaults[category]["minimum_fare"] = float(
                row[minimum_fare_col]
            )

        if (
            minimum_distance_col is not None
            and pd.notna(row[minimum_distance_col])
        ):
            fare_defaults[category]["minimum_distance_km"] = float(
                row[minimum_distance_col]
            )


for category in fare_defaults:

    fare_defaults[category]["base_fare_per_km"] = (
        fare_defaults[category]["fare_min"]
        + fare_defaults[category]["fare_max"]
    ) / 2


# ============================================================
# SERVICE CATEGORY
# ============================================================

def determine_service_category(row):

    seats = pd.to_numeric(
        row["seats"],
        errors="coerce"
    )

    if pd.isna(seats):
        seats = 5

    model = str(
        row["model"]
    ).lower()

    if seats >= 6:
        return "XL"

    small_models = [
        "aura",
        "dzire",
        "amaze",
        "tiago",
        "tigor"
    ]

    if any(
        name in model
        for name in small_models
    ):
        return "Go_AC"

    return "Sedan_Premier"


vehicles["service_category"] = vehicles.apply(
    determine_service_category,
    axis=1
)


# ============================================================
# ENERGY TYPE
# ============================================================

def determine_energy_type(row):

    powertrain = str(
        row["powertrain"]
    ).strip().lower()

    if powertrain == "bev":
        return "electricity"

    if powertrain == "cng":
        return "cng"

    if powertrain == "diesel":
        return "diesel"

    return "petrol"


vehicles["energy_type"] = vehicles.apply(
    determine_energy_type,
    axis=1
)


# ============================================================
# ENERGY COST PER KM
# ============================================================

def energy_cost_per_km(row):

    powertrain = str(
        row["powertrain"]
    ).strip().lower()

    if powertrain == "bev":

        efficiency = row[
            "energy_consumption_kwh_100km"
        ]

        if pd.isna(efficiency):
            return np.nan

        return (
            efficiency / 100
        ) * electricity_price


    if powertrain == "cng":

        efficiency = row[
            "fuel_consumption_kg_100km"
        ]

        if pd.isna(efficiency):
            return np.nan

        return (
            efficiency / 100
        ) * cng_price


    efficiency = row[
        "fuel_consumption_l_100km"
    ]

    if pd.isna(efficiency):
        return np.nan

    if powertrain == "diesel":

        return (
            efficiency / 100
        ) * diesel_price

    return (
        efficiency / 100
    ) * petrol_price


vehicles["energy_cost_inr_per_km"] = vehicles.apply(
    energy_cost_per_km,
    axis=1
)


# ============================================================
# SIMULATION
# ============================================================

records = []

for _, vehicle in vehicles.iterrows():

    for _, scenario in mobility.iterrows():

        powertrain = str(
            vehicle["powertrain"]
        ).strip().lower()

        distance_km = float(
            scenario[distance_col]
        )

        trip_time_min = float(
            scenario[time_col]
        )

        passengers = float(
            scenario[passenger_col]
        )

        luggage_kg = float(
            scenario[luggage_col]
        )

        trips_per_day = float(
            scenario[trips_col]
        )

        daily_distance = float(
            scenario[daily_distance_col]
        )


        # ----------------------------------------------------
        # ENERGY COST
        # ----------------------------------------------------

        cost_km = vehicle[
            "energy_cost_inr_per_km"
        ]

        trip_energy_cost = (
            distance_km * cost_km
            if pd.notna(cost_km)
            else np.nan
        )

        daily_energy_cost = (
            daily_distance * cost_km
            if pd.notna(cost_km)
            else np.nan
        )


        # ----------------------------------------------------
        # ENERGY QUANTITY
        # ----------------------------------------------------

        if powertrain == "bev":

            consumption = vehicle[
                "energy_consumption_kwh_100km"
            ]

            energy_quantity_trip = (
                distance_km * consumption / 100
                if pd.notna(consumption)
                else np.nan
            )

            energy_quantity_day = (
                daily_distance * consumption / 100
                if pd.notna(consumption)
                else np.nan
            )

            battery = vehicle[
                "battery_capacity_kwh"
            ]

            if (
                pd.notna(battery)
                and battery > 0
            ):

                battery_fraction_trip = (
                    energy_quantity_trip / battery
                )

                battery_fraction_day = (
                    energy_quantity_day / battery
                )

            else:

                battery_fraction_trip = np.nan
                battery_fraction_day = np.nan


        elif powertrain == "cng":

            consumption = vehicle[
                "fuel_consumption_kg_100km"
            ]

            energy_quantity_trip = (
                distance_km * consumption / 100
                if pd.notna(consumption)
                else np.nan
            )

            energy_quantity_day = (
                daily_distance * consumption / 100
                if pd.notna(consumption)
                else np.nan
            )

            battery_fraction_trip = np.nan
            battery_fraction_day = np.nan


        else:

            consumption = vehicle[
                "fuel_consumption_l_100km"
            ]

            energy_quantity_trip = (
                distance_km * consumption / 100
                if pd.notna(consumption)
                else np.nan
            )

            energy_quantity_day = (
                daily_distance * consumption / 100
                if pd.notna(consumption)
                else np.nan
            )

            battery_fraction_trip = np.nan
            battery_fraction_day = np.nan


        # ----------------------------------------------------
        # FARE
        # ----------------------------------------------------

        service_category = vehicle[
            "service_category"
        ]

        fare = fare_defaults[
            service_category
        ]

        fare_per_km = fare[
            "base_fare_per_km"
        ]

        minimum_fare = fare[
            "minimum_fare"
        ]

        distance_fare = (
            distance_km * fare_per_km
        )

        trip_revenue = max(
            distance_fare,
            minimum_fare
        )


        # ----------------------------------------------------
        # REVENUE
        # ----------------------------------------------------

        daily_revenue = (
            trip_revenue * trips_per_day
        )

        annual_revenue = (
            daily_revenue * 365
        )


        # ----------------------------------------------------
        # PASSENGER-KM
        # ----------------------------------------------------

        passenger_km = (
            distance_km * passengers
        )

        daily_passenger_km = (
            daily_distance * passengers
        )

        annual_passenger_km = (
            daily_passenger_km * 365
        )


        # ----------------------------------------------------
        # UTILIZATION
        # ----------------------------------------------------

        daily_operating_minutes = (
            trips_per_day * trip_time_min
        )

        annual_distance = (
            daily_distance * 365
        )


        # ----------------------------------------------------
        # SEAT UTILIZATION
        # ----------------------------------------------------

        seats = pd.to_numeric(
            vehicle["seats"],
            errors="coerce"
        )

        seat_utilization = (
            passengers / seats
            if pd.notna(seats) and seats > 0
            else np.nan
        )


        # ----------------------------------------------------
        # ENERGY COST / PASSENGER-KM
        # ----------------------------------------------------

        if daily_passenger_km > 0:

            energy_cost_per_passenger_km = (
                daily_energy_cost
                / daily_passenger_km
            )

        else:

            energy_cost_per_passenger_km = np.nan


        # ----------------------------------------------------
        # RANGE
        # ----------------------------------------------------

        certified_range = vehicle[
            "certified_range_km"
        ]

        range_margin = np.nan

        if (
            powertrain == "bev"
            and pd.notna(certified_range)
        ):

            range_margin = (
                certified_range
                - distance_km
            )


        # ----------------------------------------------------
        # LUGGAGE
        # ----------------------------------------------------

        luggage_volume = vehicle[
            "luggage_volume_l"
        ]

        luggage_feasible = np.nan

        if pd.notna(luggage_volume):

            # Screening proxy only.
            luggage_capacity_proxy = (
                luggage_volume * 0.25
            )

            luggage_feasible = (
                luggage_capacity_proxy >= luggage_kg
            )


        # ----------------------------------------------------
        # STORE RECORD
        # ----------------------------------------------------

        records.append({

            "vehicle_id":
                vehicle["vehicle_id"],

            "manufacturer":
                vehicle["manufacturer"],

            "model":
                vehicle["model"],

            "model_year":
                vehicle["model_year"],

            "powertrain":
                vehicle["powertrain"],

            "vehicle_type":
                vehicle["vehicle_type"],

            "seats":
                vehicle["seats"],

            "kerb_weight_kg":
                vehicle["kerb_weight_kg"],

            "wheelbase_mm":
                vehicle["wheelbase_mm"],

            "ground_clearance_mm":
                vehicle["ground_clearance_mm"],

            "luggage_volume_l":
                vehicle["luggage_volume_l"],

            "battery_capacity_kwh":
                vehicle["battery_capacity_kwh"],

            "certified_range_km":
                vehicle["certified_range_km"],

            "power_kw":
                vehicle["power_kw"],

            "service_category":
                service_category,

            "energy_type":
                vehicle["energy_type"],

            "scenario_id":
                scenario[scenario_id_col],

            "city":
                scenario[city_col],

            "scenario_name":
                scenario[scenario_name_col],

            "trip_distance_km":
                distance_km,

            "trip_time_min":
                trip_time_min,

            "typical_passengers":
                passengers,

            "typical_luggage_kg":
                luggage_kg,

            "trips_per_day":
                trips_per_day,

            "daily_distance_km":
                daily_distance,

            "energy_cost_inr_per_km":
                cost_km,

            "trip_energy_quantity":
                energy_quantity_trip,

            "daily_energy_quantity":
                energy_quantity_day,

            "trip_energy_cost_inr":
                trip_energy_cost,

            "daily_energy_cost_inr":
                daily_energy_cost,

            "annual_energy_cost_inr":
                (
                    daily_energy_cost * 365
                    if pd.notna(daily_energy_cost)
                    else np.nan
                ),

            "battery_fraction_trip":
                battery_fraction_trip,

            "battery_fraction_day":
                battery_fraction_day,

            "range_margin_km":
                range_margin,

            "fare_per_km_inr":
                fare_per_km,

            "minimum_fare_inr":
                minimum_fare,

            "trip_revenue_inr":
                trip_revenue,

            "daily_revenue_inr":
                daily_revenue,

            "annual_revenue_inr":
                annual_revenue,

            "daily_operating_minutes":
                daily_operating_minutes,

            "annual_distance_km":
                annual_distance,

            "passenger_km_per_trip":
                passenger_km,

            "daily_passenger_km":
                daily_passenger_km,

            "annual_passenger_km":
                annual_passenger_km,

            "seat_utilization":
                seat_utilization,

            "energy_cost_per_passenger_km":
                energy_cost_per_passenger_km,

            "luggage_feasible":
                luggage_feasible,

            "data_quality":
                vehicle["data_quality"],

            "specification_source_type":
                vehicle["specification_source_type"]
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

simulation = pd.DataFrame(records)


# ============================================================
# VALIDATION
# ============================================================

expected_records = (
    len(vehicles) * len(mobility)
)

print()
print("=" * 70)
print("RECORD VALIDATION")
print("=" * 70)

print("Expected records :", expected_records)
print("Actual records   :", len(simulation))

if len(simulation) != expected_records:

    raise RuntimeError(
        "Record count mismatch. "
        "Simulation was not completed correctly."
    )


# ============================================================
# DERIVED METRICS
# ============================================================

simulation[
    "annual_energy_cost_share"
] = (
    simulation["annual_energy_cost_inr"]
    / simulation["annual_revenue_inr"]
)

simulation[
    "revenue_per_km"
] = (
    simulation["annual_revenue_inr"]
    / simulation["annual_distance_km"]
)

simulation[
    "energy_contribution_inr_per_day"
] = (
    simulation["daily_revenue_inr"]
    - simulation["daily_energy_cost_inr"]
)

simulation[
    "energy_contribution_inr_per_year"
] = (
    simulation["annual_revenue_inr"]
    - simulation["annual_energy_cost_inr"]
)


# ============================================================
# VEHICLE SUMMARY
# ============================================================

summary = (
    simulation
    .groupby(
        [
            "vehicle_id",
            "manufacturer",
            "model",
            "powertrain",
            "seats",
            "service_category"
        ],
        as_index=False
    )
    .agg({
        "annual_distance_km": "mean",
        "annual_passenger_km": "mean",
        "annual_revenue_inr": "mean",
        "annual_energy_cost_inr": "mean",
        "energy_cost_per_passenger_km": "mean",
        "energy_cost_inr_per_km": "mean",
        "energy_contribution_inr_per_year": "mean",
        "seat_utilization": "mean",
        "battery_fraction_day": "mean",
        "luggage_feasible": "mean",
        "kerb_weight_kg": "first",
        "wheelbase_mm": "first",
        "luggage_volume_l": "first",
        "power_kw": "first",
        "battery_capacity_kwh": "first",
        "certified_range_km": "first"
    })
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

simulation.to_csv(
    OUTPUT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)

print()
print("Vehicle x scenario records:")
print(len(simulation))

print()
print("Expected:")
print(
    len(vehicles),
    "vehicles x",
    len(mobility),
    "scenarios =",
    len(vehicles) * len(mobility)
)

print()
print("Average energy cost/km:")

print(
    simulation
    .groupby("powertrain")[
        "energy_cost_inr_per_km"
    ]
    .mean()
    .round(2)
)

print()
print("Average annual distance:")

print(
    simulation
    .groupby("scenario_name")[
        "annual_distance_km"
    ]
    .mean()
    .round(0)
)

print()
print("Lowest energy-cost vehicles:")

print(
    summary[
        [
            "vehicle_id",
            "model",
            "powertrain",
            "energy_cost_inr_per_km"
        ]
    ]
    .sort_values(
        "energy_cost_inr_per_km"
    )
    .head(10)
    .to_string(index=False)
)

print()
print("Highest energy contribution:")

print(
    summary[
        [
            "vehicle_id",
            "model",
            "powertrain",
            "energy_contribution_inr_per_year"
        ]
    ]
    .sort_values(
        "energy_contribution_inr_per_year",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print()
print("Files created:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print()
print(
    "NOTE: Energy contribution is NOT full profit. "
    "Maintenance, insurance, depreciation, financing, "
    "driver compensation, platform costs, downtime and "
    "deadheading are not included yet."
)

print()
print(
    "NOTE: Current energy prices and fare references are "
    "baseline inputs, not 2035 forecasts."
)
