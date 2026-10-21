import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/raw/vehicles_raw.csv")
OUTPUT = Path("data/processed/vehicles_processed.csv")

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# 1. Clean blank strings
# ------------------------------------------------------------
df = df.replace(["", " ", "-", "NA", "N/A", "na"], np.nan)

# ------------------------------------------------------------
# 2. Convert numeric columns
# ------------------------------------------------------------
numeric_cols = [
    "model_year",
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
    "energy_consumption_kwh_100km",
    "fuel_consumption_l_100km",
    "charging_power_kw",
    "charging_time_min",
    "purchase_price_inr"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 3. Basic checks
# ------------------------------------------------------------
print("\n================ VEHICLE DATA CHECK ================\n")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate vehicle IDs:", df["vehicle_id"].duplicated().sum())

print("\nPowertrain distribution:")
print(df["powertrain"].value_counts())

# ------------------------------------------------------------
# 4. Calculate useful derived values where possible
# ------------------------------------------------------------

# Range per kWh
df["range_per_kwh"] = np.nan

mask = (
    df["battery_capacity_kwh"].notna()
    & (df["battery_capacity_kwh"] > 0)
    & df["certified_range_km"].notna()
)

df.loc[mask, "range_per_kwh"] = (
    df.loc[mask, "certified_range_km"]
    / df.loc[mask, "battery_capacity_kwh"]
)

# Approximate battery energy consumption from certified range
df["derived_energy_consumption_kwh_100km"] = np.nan

df.loc[mask, "derived_energy_consumption_kwh_100km"] = (
    df.loc[mask, "battery_capacity_kwh"]
    / df.loc[mask, "certified_range_km"]
    * 100
)

# Power-to-weight ratio
df["power_to_weight_kw_per_tonne"] = np.nan

mask_pw = (
    df["power_kw"].notna()
    & df["kerb_weight_kg"].notna()
    & (df["kerb_weight_kg"] > 0)
)

df.loc[mask_pw, "power_to_weight_kw_per_tonne"] = (
    df.loc[mask_pw, "power_kw"]
    / df.loc[mask_pw, "kerb_weight_kg"]
    * 1000
)

# ------------------------------------------------------------
# 5. Missing-value report
# ------------------------------------------------------------

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (
        df.isna().sum().values / len(df) * 100
    ).round(1)
})

print("\nMissing values:")
print(
    missing[
        missing["missing_count"] > 0
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 6. Save processed dataset
# ------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)

missing.to_csv(
    "results/vehicle_missing_report.csv",
    index=False
)

print("\n====================================================")
print("Processed vehicle dataset saved:")
print(OUTPUT)

print("\nMissing-value report saved:")
print("results/vehicle_missing_report.csv")

print("\nFinal shape:", df.shape)