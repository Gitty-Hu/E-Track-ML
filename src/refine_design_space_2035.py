import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# INDIA 2035 - ENGINEERING REFINEMENT OF HYPOTHETICAL
# VEHICLE DESIGN SPACE
# ============================================================
#
# Purpose:
# Convert the broad hypothetical design space into a more
# physically and commercially defensible design space before
# NSGA-II optimization.
#
# Input:
#   data/processed/design_space_evaluated.csv
#
# Output:
#   data/processed/design_space_refined_2035.csv
#
# Main checks:
#   1. Power-to-weight
#   2. Battery-to-mass relationship
#   3. BEV range-to-battery relationship
#   4. BEV energy-efficiency plausibility
#   5. Seating-to-mass relationship
#   6. Seating-to-luggage relationship
#   7. Vehicle-size-to-luggage relationship
#   8. Vehicle-size-to-price relationship
#   9. Powertrain-specific energy plausibility
#
# IMPORTANT:
# These are engineering screening constraints, not claims that
# these exact limits will apply to every vehicle in 2035.
# They are used to prevent the optimizer from exploiting
# obviously unrealistic combinations.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "design_space_evaluated.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "design_space_refined_2035.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INDIA 2035 ENGINEERING DESIGN-SPACE REFINEMENT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput designs: {len(df):,}")


# ============================================================
# BASIC CHECK
# ============================================================

required_columns = [
    "design_id",
    "powertrain",
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
    "battery_capacity_kwh",
    "certified_range_km",
    "predicted_energy_quantity_per_km",
    "predicted_comfort_index",
    "annual_profit_inr",
    "annual_co2_tonnes",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(f"  - {c}" for c in missing)
    )


# ============================================================
# NUMERIC CONVERSION
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
    "predicted_energy_quantity_per_km",
    "predicted_comfort_index",
    "annual_profit_inr",
    "annual_co2_tonnes",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# ORIGINAL DESIGN-SPACE CHECKS
# ============================================================

df["check_power_to_weight"] = (
    (df["power_to_weight_kw_per_tonne"] >= 40)
    & (df["power_to_weight_kw_per_tonne"] <= 120)
)


# ============================================================
# 1. SEATING VS VEHICLE MASS
# ============================================================
#
# Larger passenger capacity requires additional structure,
# seating hardware and interior volume.
#
# Conservative engineering screening:
#
# 4 seats -> >= 1050 kg
# 5 seats -> >= 1100 kg
# 6 seats -> >= 1250 kg
# 7 seats -> >= 1350 kg
#
# These are screening values, not certification requirements.
# ============================================================

minimum_mass_by_seats = {
    4: 1050,
    5: 1100,
    6: 1250,
    7: 1350,
}

df["minimum_mass_required_kg"] = (
    df["seats"].map(minimum_mass_by_seats)
)

df["check_seat_mass"] = (
    df["kerb_weight_kg"]
    >= df["minimum_mass_required_kg"]
)


# ============================================================
# 2. SEATING VS LUGGAGE
# ============================================================
#
# Ride-service vehicles need useful luggage capacity in addition
# to passenger seating.
#
# Minimum luggage volume:
#
# 4 seats -> 300 L
# 5 seats -> 320 L
# 6 seats -> 380 L
# 7 seats -> 400 L
# ============================================================

minimum_luggage_by_seats = {
    4: 300,
    5: 320,
    6: 380,
    7: 400,
}

df["minimum_luggage_required_l"] = (
    df["seats"].map(minimum_luggage_by_seats)
)

df["check_seat_luggage"] = (
    df["luggage_volume_l"]
    >= df["minimum_luggage_required_l"]
)


# ============================================================
# 3. WHEELBASE VS LUGGAGE
# ============================================================
#
# A longer wheelbase generally provides more packaging freedom.
#
# Instead of forcing a strict physical equation, we impose
# only a minimum luggage requirement for larger wheelbases.
#
# wheelbase >= 2750 mm:
#       luggage >= 400 L
#
# This prevents a very long passenger vehicle from having an
# unrealistically small luggage compartment.
# ============================================================

df["check_wheelbase_luggage"] = True

long_vehicle = df["wheelbase_mm"] >= 2750

df.loc[long_vehicle, "check_wheelbase_luggage"] = (
    df.loc[long_vehicle, "luggage_volume_l"] >= 400
)


# ============================================================
# 4. BEV BATTERY VS VEHICLE MASS
# ============================================================
#
# A larger battery requires additional mass and packaging.
#
# Approximate minimum kerb mass:
#
# Battery <= 30 kWh:
#       >= 1050 kg
#
# 30-45 kWh:
#       >= 1150 kg
#
# 45-60 kWh:
#       >= 1250 kg
#
# 60-80 kWh:
#       >= 1400 kg
#
# Again, these are screening constraints rather than exact
# engineering equations.
# ============================================================

df["minimum_bev_mass_required_kg"] = np.nan

bev = df["powertrain"] == "BEV"

df.loc[bev, "minimum_bev_mass_required_kg"] = np.select(
    [
        df.loc[bev, "battery_capacity_kwh"] <= 30,
        df.loc[bev, "battery_capacity_kwh"] <= 45,
        df.loc[bev, "battery_capacity_kwh"] <= 60,
        df.loc[bev, "battery_capacity_kwh"] <= 80,
    ],
    [
        1050,
        1150,
        1250,
        1400,
    ],
    default=1400,
)

df["check_bev_battery_mass"] = True

df.loc[bev, "check_bev_battery_mass"] = (
    df.loc[bev, "kerb_weight_kg"]
    >= df.loc[bev, "minimum_bev_mass_required_kg"]
)


# ============================================================
# 5. BEV RANGE VS BATTERY
# ============================================================
#
# Range cannot be completely independent of battery capacity.
#
# We use a plausible certified-range envelope:
#
# Minimum:
#       4.5 km/kWh
#
# Maximum:
#       10.0 km/kWh
#
# This prevents the optimizer from creating a tiny battery with
# an unrealistically large range.
# ============================================================

df["bev_range_efficiency_km_per_kwh"] = np.nan

df.loc[bev, "bev_range_efficiency_km_per_kwh"] = (
    df.loc[bev, "certified_range_km"]
    / df.loc[bev, "battery_capacity_kwh"]
)

df["check_bev_range_battery"] = True

df.loc[bev, "check_bev_range_battery"] = (
    (df.loc[bev, "bev_range_efficiency_km_per_kwh"] >= 4.5)
    &
    (df.loc[bev, "bev_range_efficiency_km_per_kwh"] <= 10.0)
)


# ============================================================
# 6. BEV ENERGY EFFICIENCY PLAUSIBILITY
# ============================================================
#
# The ML model predicts kWh/km.
#
# Screening envelope:
#
#     0.12 <= kWh/km <= 0.30
#
# This is intentionally broad enough to accommodate different
# passenger-vehicle sizes.
# ============================================================

df["check_bev_energy"] = True

df.loc[bev, "check_bev_energy"] = (
    (df.loc[bev, "predicted_energy_quantity_per_km"] >= 0.08)
    &
    (df.loc[bev, "predicted_energy_quantity_per_km"] <= 0.30)
)


# ============================================================
# 7. CNG EFFICIENCY PLAUSIBILITY
# ============================================================
#
# The ML model predicts kg/km for CNG.
#
# Approximate plausible range:
#
# 0.025 <= kg/km <= 0.060
#
# Equivalent to approximately:
#
# 16.7 to 40 km/kg
# ============================================================

cng = df["powertrain"] == "CNG"

df["check_cng_energy"] = True

df.loc[cng, "check_cng_energy"] = (
    (df.loc[cng, "predicted_energy_quantity_per_km"] >= 0.025)
    &
    (df.loc[cng, "predicted_energy_quantity_per_km"] <= 0.060)
)


# ============================================================
# 8. PETROL EFFICIENCY PLAUSIBILITY
# ============================================================
#
# Petrol ML output is L/km.
#
# Screening envelope:
#
# 0.035 <= L/km <= 0.090
#
# Equivalent to approximately:
#
# 11.1 to 28.6 km/L.
# ============================================================

petrol = df["powertrain"] == "Petrol"

df["check_petrol_energy"] = True

df.loc[petrol, "check_petrol_energy"] = (
    (df.loc[petrol, "predicted_energy_quantity_per_km"] >= 0.035)
    &
    (df.loc[petrol, "predicted_energy_quantity_per_km"] <= 0.090)
)


# ============================================================
# 9. HYBRID EFFICIENCY PLAUSIBILITY
# ============================================================
#
# Hybrid is represented through liquid-fuel consumption in the
# current dataset.
#
# Screening envelope:
#
# 0.025 <= L/km <= 0.070
#
# Equivalent to approximately:
#
# 14.3 to 40 km/L.
# ============================================================

hybrid = df["powertrain"] == "Hybrid"

df["check_hybrid_energy"] = True

df.loc[hybrid, "check_hybrid_energy"] = (
    (df.loc[hybrid, "predicted_energy_quantity_per_km"] >= 0.025)
    &
    (df.loc[hybrid, "predicted_energy_quantity_per_km"] <= 0.070)
)


# ============================================================
# 10. VEHICLE POWER VS MASS
# ============================================================
#
# We retain the original 40-120 kW/tonne engineering envelope.
# ============================================================

df["check_power_mass"] = (
    df["power_to_weight_kw_per_tonne"] >= 40
) & (
    df["power_to_weight_kw_per_tonne"] <= 120
)


# ============================================================
# 11. PRICE PLAUSIBILITY
# ============================================================
#
# Prevent extremely low-cost large/high-performance vehicles.
#
# We use a simple minimum price envelope based on vehicle
# complexity.
#
# Base minimum price:
#       Rs 8 lakh
#
# Additional requirements:
#
# 7 seats:
#       >= Rs 10 lakh
#
# BEV > 60 kWh:
#       >= Rs 15 lakh
#
# Power > 130 kW:
#       >= Rs 12 lakh
#
# These are scenario-screening rules, not future OEM prices.
# ============================================================

df["minimum_price_required_inr"] = 800000.0

df.loc[df["seats"] >= 7, "minimum_price_required_inr"] = np.maximum(
    df.loc[df["seats"] >= 7, "minimum_price_required_inr"],
    1000000
)

large_bev = (
    (df["powertrain"] == "BEV")
    & (df["battery_capacity_kwh"] > 60)
)

df.loc[large_bev, "minimum_price_required_inr"] = np.maximum(
    df.loc[large_bev, "minimum_price_required_inr"],
    1500000
)

high_power = df["power_kw"] > 130

df.loc[high_power, "minimum_price_required_inr"] = np.maximum(
    df.loc[high_power, "minimum_price_required_inr"],
    1200000
)

df["check_price"] = (
    df["purchase_price_inr"]
    >= df["minimum_price_required_inr"]
)


# ============================================================
# 12. HIGH-CAPACITY VEHICLE POWER
# ============================================================
#
# Vehicles carrying 6-7 passengers should not be extremely
# underpowered.
#
# Minimum power:
#
# 6 seats -> 70 kW
# 7 seats -> 75 kW
# ============================================================

df["minimum_power_required_kw"] = 60.0

df.loc[df["seats"] == 6, "minimum_power_required_kw"] = 70.0
df.loc[df["seats"] == 7, "minimum_power_required_kw"] = 75.0

df["check_capacity_power"] = (
    df["power_kw"]
    >= df["minimum_power_required_kw"]
)


# ============================================================
# 13. BEV BATTERY PACKAGING / RANGE CONSISTENCY
# ============================================================
#
# A very large battery combined with a very small vehicle is
# screened out.
#
# Battery per tonne:
#
# Maximum approximately 45 kWh/tonne.
# ============================================================

df["battery_per_tonne_kwh"] = np.nan

df.loc[bev, "battery_per_tonne_kwh"] = (
    df.loc[bev, "battery_capacity_kwh"]
    / (df.loc[bev, "kerb_weight_kg"] / 1000)
)

df["check_battery_packaging"] = True

df.loc[bev, "check_battery_packaging"] = (
    df.loc[bev, "battery_per_tonne_kwh"] <= 45
)


# ============================================================
# 14. COMBINED ENGINEERING FEASIBILITY
# ============================================================

check_columns = [
    "check_power_to_weight",
    "check_seat_mass",
    "check_seat_luggage",
    "check_wheelbase_luggage",
    "check_bev_battery_mass",
    "check_bev_range_battery",
    "check_bev_energy",
    "check_cng_energy",
    "check_petrol_energy",
    "check_hybrid_energy",
    "check_power_mass",
    "check_price",
    "check_capacity_power",
    "check_battery_packaging",
]

df["engineering_feasible_refined"] = (
    df[check_columns]
    .all(axis=1)
    .astype(int)
)


# ============================================================
# COUNT FAILURES
# ============================================================

print("\n" + "=" * 70)
print("ENGINEERING CONSTRAINT RESULTS")
print("=" * 70)

for col in check_columns:

    failed = (~df[col]).sum()

    print(
        f"{col:40s}: "
        f"{failed:5d} failed"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REFINED DESIGN SPACE")
print("=" * 70)

print(
    f"\nOriginal evaluated designs : {len(df):,}"
)

print(
    f"Refined feasible designs   : "
    f"{df['engineering_feasible_refined'].sum():,}"
)

print(
    f"Rejected designs           : "
    f"{(df['engineering_feasible_refined'] == 0).sum():,}"
)


# ============================================================
# POWERTRAIN DISTRIBUTION
# ============================================================

print("\nPowertrain distribution - refined feasible designs:")

print(
    df.loc[
        df["engineering_feasible_refined"] == 1,
        "powertrain"
    ]
    .value_counts()
)


# ============================================================
# SEAT DISTRIBUTION
# ============================================================

print("\nSeat distribution - refined feasible designs:")

print(
    df.loc[
        df["engineering_feasible_refined"] == 1,
        "seats"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# DESIGN RANGES
# ============================================================

feasible = df[
    df["engineering_feasible_refined"] == 1
].copy()

if len(feasible) == 0:
    raise ValueError(
        "\nNo feasible designs remain after refinement. "
        "Constraints need to be relaxed."
    )


print("\n" + "-" * 70)
print("FEASIBLE DESIGN RANGES")
print("-" * 70)

range_columns = [
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
    "predicted_comfort_index",
    "annual_profit_inr",
    "annual_co2_tonnes",
    "power_to_weight_kw_per_tonne",
]

for col in range_columns:

    print(
        f"{col:40s}: "
        f"{feasible[col].min():.2f}"
        f" – "
        f"{feasible[col].max():.2f}"
    )


# ============================================================
# BEV BATTERY / RANGE SUMMARY
# ============================================================

bev_feasible = feasible[
    feasible["powertrain"] == "BEV"
]

if len(bev_feasible) > 0:

    print("\n" + "-" * 70)
    print("FEASIBLE BEV DESIGN RANGE")
    print("-" * 70)

    print(
        f"Battery capacity : "
        f"{bev_feasible['battery_capacity_kwh'].min():.2f}"
        f" – "
        f"{bev_feasible['battery_capacity_kwh'].max():.2f} kWh"
    )

    print(
        f"Certified range  : "
        f"{bev_feasible['certified_range_km'].min():.2f}"
        f" – "
        f"{bev_feasible['certified_range_km'].max():.2f} km"
    )

    print(
        f"Range efficiency : "
        f"{bev_feasible['bev_range_efficiency_km_per_kwh'].min():.2f}"
        f" – "
        f"{bev_feasible['bev_range_efficiency_km_per_kwh'].max():.2f}"
        f" km/kWh"
    )


# ============================================================
# POWERTRAIN SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("POWERTRAIN SUMMARY")
print("=" * 70)

summary = (
    feasible
    .groupby("powertrain")
    .agg(
        designs=("design_id", "count"),
        avg_comfort=("predicted_comfort_index", "mean"),
        avg_profit=("annual_profit_inr", "mean"),
        avg_co2_tonnes=("annual_co2_tonnes", "mean"),
        avg_energy_cost_km=(
            "energy_cost_inr_per_km_2035",
            "mean"
        ),
        avg_purchase_price=(
            "purchase_price_inr",
            "mean"
        ),
    )
    .round(2)
)

print(summary.to_string())


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("REFINEMENT COMPLETE")
print("=" * 70)

print(
    f"\nSaved:\n"
    f"  {OUTPUT_FILE}"
)

print(
    "\nNext step:"
    "\nInspect the refined design count and powertrain distribution."
    "\nIf the feasible space is reasonable, we will build the"
    "\nNSGA-II multi-objective optimizer."
)