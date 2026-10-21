import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. MOBILITY DATA
# ============================================================
# These are MODELLING SCENARIOS, not claimed to be observed trips.
# Passenger occupancy is anchored to NITI Aayog 2026:
# 4W taxi typical occupancy = 2.8 passengers.
#
# Distance/daily utilisation values are engineering scenarios
# designed to test different commercial operating conditions.

mobility = [
    ["M001", "Bengaluru", "urban_short",
     8, 25, 2.8, 5, 1, "urban", 0.00, 12, 96,
     "scenario", "Engineering duty-cycle scenario", 2026],

    ["M002", "Bengaluru", "urban_commute",
     15, 45, 2.8, 10, 1, "urban", 0.00, 10, 150,
     "scenario", "Engineering duty-cycle scenario", 2026],

    ["M003", "Mangaluru", "urban_mixed",
     18, 40, 2.8, 15, 1, "mixed", 0.10, 9, 162,
     "scenario", "Engineering duty-cycle scenario", 2026],

    ["M004", "Bengaluru", "airport",
     35, 60, 2.8, 35, 1, "mixed", 0.40, 5, 175,
     "scenario", "Engineering duty-cycle scenario", 2026],

    ["M005", "Mangaluru", "inter_city",
     120, 150, 2.8, 40, 0, "highway", 0.80, 2, 240,
     "scenario", "Engineering duty-cycle scenario", 2026],

    ["M006", "Bengaluru", "long_distance",
     250, 300, 2.8, 50, 0, "highway", 0.90, 1, 250,
     "scenario", "Engineering duty-cycle scenario", 2026],

    ["M007", "India_mixed", "mixed_commercial",
     45, 70, 2.8, 20, 1, "mixed", 0.45, 7, 315,
     "scenario", "Engineering duty-cycle scenario", 2026],
]

mobility_columns = [
    "record_id",
    "city",
    "trip_type",
    "distance_km",
    "duration_min",
    "passengers",
    "luggage_kg",
    "peak_period",
    "road_type",
    "highway_fraction",
    "trips_per_day",
    "daily_distance_km",
    "data_type",
    "source",
    "source_year"
]

pd.DataFrame(mobility, columns=mobility_columns).to_csv(
    RAW / "mobility_raw.csv", index=False
)


# ============================================================
# 2. ENERGY DATA
# ============================================================
# CURRENT observed/reference prices:
#
# Petrol Bengaluru:
# Rs 111.68/L on 03-Oct-2026
#
# Diesel Karnataka:
# Rs 99.19/L on 03-Oct-2026
#
# CNG Bengaluru:
# Rs 97/kg in PNGRB June-2026 data
#
# EV charging:
# Rs 4.50/kWh energy charge in Karnataka EV charging tariff
#
# These are CURRENT reference values.
# They are NOT treated as 2035 forecasts.
#
# 2035 values will be generated later using sensitivity scenarios.

energy = [
    [
        "E001", "BEV", "electricity",
        4.50, "INR/kWh",
        "public_or_depot",
        60,
        0.90,
        0.025,
        "current_reference",
        "sourced",
        "KERC EV charging tariff",
        "2026-10-03"
    ],

    [
        "E002", "CNG", "CNG",
        97.00, "INR/kg",
        "CNG_station",
        0,
        1.00,
        0.000,
        "current_reference",
        "sourced",
        "PNGRB - GAIL Gas Bengaluru",
        "2026-06"
    ],

    [
        "E003", "Petrol", "petrol",
        111.68, "INR/L",
        "fuel_station",
        0,
        1.00,
        0.000,
        "current_reference",
        "sourced",
        "Bengaluru petrol price",
        "2026-10-03"
    ],

    [
        "E004", "Diesel", "diesel",
        99.19, "INR/L",
        "fuel_station",
        0,
        1.00,
        0.000,
        "current_reference",
        "sourced",
        "Karnataka diesel price",
        "2026-10-03"
    ],

    [
        "E005", "Strong_Hybrid", "petrol",
        111.68, "INR/L",
        "fuel_station",
        0,
        1.00,
        0.000,
        "current_reference",
        "sourced",
        "Bengaluru petrol price",
        "2026-10-03"
    ],
]

energy_columns = [
    "record_id",
    "powertrain",
    "energy_source",
    "energy_price",
    "price_unit",
    "charging_availability",
    "fast_charging_power_kw",
    "charging_efficiency",
    "battery_degradation_per_year",
    "scenario",
    "data_type",
    "source",
    "source_date"
]

pd.DataFrame(energy, columns=energy_columns).to_csv(
    RAW / "energy_raw.csv", index=False
)


# ============================================================
# 3. ECONOMICS / FARE DATA
# ============================================================
# Bengaluru aggregator fare ranges published by Uber
# under Karnataka's 2021 aggregator fare notification:
#
# Go AC:
#   minimum ₹18/km, maximum ₹36/km
#
# Go Sedan / Premier:
#   minimum ₹21/km, maximum ₹42/km
#
# XL:
#   minimum ₹24/km, maximum ₹48/km
#
# We preserve the RANGE instead of inventing one fare.
#
# A midpoint is calculated later by the model.
#
# Platform commission is deliberately left blank here.
# We will NOT invent a commission percentage.

economics = [
    [
        "EC001",
        "Bengaluru",
        "Go_AC",
        18,
        36,
        75,
        4,
        "sourced",
        "Uber/Karnataka aggregator fare",
        "2021"
    ],

    [
        "EC002",
        "Bengaluru",
        "Sedan_Premier",
        21,
        42,
        100,
        4,
        "sourced",
        "Uber/Karnataka aggregator fare",
        "2021"
    ],

    [
        "EC003",
        "Bengaluru",
        "XL",
        24,
        48,
        120,
        4,
        "sourced",
        "Uber/Karnataka aggregator fare",
        "2021"
    ],
]

economics_columns = [
    "record_id",
    "city",
    "vehicle_segment",
    "fare_min_per_km",
    "fare_max_per_km",
    "minimum_fare_inr",
    "minimum_fare_distance_km",
    "data_type",
    "source",
    "source_year"
]

pd.DataFrame(economics, columns=economics_columns).to_csv(
    RAW / "economics_raw.csv", index=False
)


# ============================================================
# DONE
# ============================================================

print("\nSOURCED DATASETS CREATED SUCCESSFULLY\n")

for filename in [
    "mobility_raw.csv",
    "energy_raw.csv",
    "economics_raw.csv"
]:
    df = pd.read_csv(RAW / filename)
    print(f"{filename}: {df.shape}")

print("\nFiles saved in:")
print(RAW.resolve())