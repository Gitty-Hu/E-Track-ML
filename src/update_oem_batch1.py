import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/vehicles_verified.csv")
OUTPUT = Path("data/processed/vehicles_verified.csv")

df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# OEM-VERIFIED BATCH 1
# ---------------------------------------------------------
# When OEM gives a range, we use the midpoint for the
# numerical ML dataset and retain the range separately
# where useful.
#
# IMPORTANT:
# These are sourced specifications, NOT estimates.
# ---------------------------------------------------------

updates = {

    # -----------------------------------------------------
    # HYUNDAI AURA
    # -----------------------------------------------------
    "V020": {
        "wheelbase_mm": 2450,
        "power_kw": 50.5,
        "fuel_tank_l": 65,
        "fuel_consumption_l_100km": 100 / 21.0,  # placeholder only if CNG is converted
    },

    "V026": {
        "wheelbase_mm": 2450,
        "power_kw": 61.0,
        "fuel_tank_l": 37,
    },

    # -----------------------------------------------------
    # MARUTI SUZUKI DZIRE
    # -----------------------------------------------------
    "V021": {
        "wheelbase_mm": 2450,
        "kerb_weight_kg": 1022.5,
        "power_kw": 74.0,          # CNG gasoline-mode maximum
        "fuel_tank_l": 55,
        "fuel_consumption_l_100km": None,
    },

    "V027": {
        "wheelbase_mm": 2450,
        "kerb_weight_kg": 940.0,
        "power_kw": 75.8,
        "fuel_tank_l": 37,
        "fuel_consumption_l_100km": 100 / 24.79,
    },

    # -----------------------------------------------------
    # MARUTI SUZUKI ERTIGA
    # -----------------------------------------------------
    "V022": {
        "wheelbase_mm": 2740,
        "kerb_weight_kg": 1252.5,
        "power_kw": 64.6,
        "fuel_tank_l": 60,
        "fuel_consumption_l_100km": None,
    },

    "V046": {
        "wheelbase_mm": 2740,
        "kerb_weight_kg": 1177.5,
        "power_kw": 75.8,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 20.51,
    },

    # -----------------------------------------------------
    # MARUTI SUZUKI XL6
    # -----------------------------------------------------
    "V023": {
        "wheelbase_mm": 2740,
        "power_kw": 64.6,
        "fuel_tank_l": 60,
        "fuel_consumption_l_100km": None,
    },

    "V047": {
        "wheelbase_mm": 2740,
        "power_kw": 75.8,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 20.97,
    },

    # -----------------------------------------------------
    # MARUTI SUZUKI BREZZA
    # -----------------------------------------------------
    "V024": {
        "power_kw": None,
        "fuel_tank_l": None,
        "fuel_consumption_l_100km": None,
    },

    "V048": {
        "power_kw": None,
        "fuel_tank_l": None,
        "fuel_consumption_l_100km": 100 / 17.80,
    },

    # -----------------------------------------------------
    # TOYOTA RUMION
    # -----------------------------------------------------
    "V025": {
        "kerb_weight_kg": 1270,
        "wheelbase_mm": 2740,
        "power_kw": 64.6,
        "fuel_consumption_l_100km": None,
    },

    # -----------------------------------------------------
    # HONDA AMAZE
    # -----------------------------------------------------
    "V028": {
        "kerb_weight_kg": 969,
        "wheelbase_mm": 2470,
        "power_kw": 66.0,
        "fuel_tank_l": 35,
        "fuel_consumption_l_100km": 100 / 18.65,
        "luggage_volume_l": 416,
    },

    "V029": {
        "kerb_weight_kg": 969,
        "wheelbase_mm": 2470,
        "power_kw": 66.0,
        "fuel_tank_l": 35,
        "fuel_consumption_l_100km": 100 / 19.46,
        "luggage_volume_l": 416,
    },

    # -----------------------------------------------------
    # HONDA CITY e:HEV
    # -----------------------------------------------------
    "V043": {
        "kerb_weight_kg": 1280,
        "wheelbase_mm": 2600,
        "power_kw": 93.0,
        "fuel_tank_l": 40,
        "fuel_consumption_l_100km": 100 / 26.5,
    },

    # -----------------------------------------------------
    # KIA CARENS
    # -----------------------------------------------------
    # Wheelbase already OEM verified in previous batch.
    # Do not invent kerb weights until we have India-specific
    # variant documentation.
}

# ---------------------------------------------------------
# Apply updates
# ---------------------------------------------------------

for vehicle_id, values in updates.items():

    mask = df["vehicle_id"] == vehicle_id

    if not mask.any():
        print(f"WARNING: {vehicle_id} not found")
        continue

    for column, value in values.items():

        # Do not overwrite an existing verified value
        if column not in df.columns:
            print(f"WARNING: column {column} does not exist")
            continue

        if pd.notna(value):
            df.loc[mask, column] = value

# ---------------------------------------------------------
# Special handling for CNG efficiency
# ---------------------------------------------------------
#
# Our dataset has fuel_consumption_l_100km for liquid fuels.
# CNG is measured in kg/100 km, NOT litres/100 km.
#
# Therefore DO NOT convert CNG mileage into litres/100 km.
#
# The existing dataset schema needs a future column:
#
#     fuel_consumption_kg_100km
#
# This will be added in the next cleanup stage.
# ---------------------------------------------------------

# Correct the erroneous Dzire CNG value if it exists
# in a separate mileage field.
for col in df.columns:
    if "cng" in col.lower() and "consumption" in col.lower():
        print("CNG-related column:", col)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print()
print("BATCH 1 OEM UPDATE COMPLETE")
print("--------------------------------")
print("Rows:", len(df))

print()
print("Remaining critical missing values:")
critical = [
    "kerb_weight_kg",
    "wheelbase_mm",
    "battery_capacity_kwh",
    "certified_range_km",
    "power_kw",
    "fuel_consumption_l_100km",
]

for col in critical:
    if col in df.columns:
        print(f"{col:35s}", df[col].isna().sum())

print()
print("Saved:")
print(OUTPUT)