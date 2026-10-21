import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/vehicles_raw.csv")
OUT = Path("data/processed/vehicles_verified.csv")

df = pd.read_csv(RAW)

# ---------------------------------------------------------
# Clean blanks
# ---------------------------------------------------------
df = df.replace(["", " ", "-", "NA", "N/A"], np.nan)

# ---------------------------------------------------------
# Add provenance columns
# ---------------------------------------------------------
df["verification_status"] = "needs_verification"
df["verified_source"] = ""
df["source_date"] = "2026-10-03"

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def verify(vehicle_ids, source, values=None):
    mask = df["vehicle_id"].isin(vehicle_ids)

    df.loc[mask, "verification_status"] = "OEM_verified"
    df.loc[mask, "verified_source"] = source

    if values:
        for column, value_dict in values.items():
            for vehicle_id, value in value_dict.items():
                df.loc[
                    df["vehicle_id"] == vehicle_id,
                    column
                ] = value


# =========================================================
# HYUNDAI CRETA ELECTRIC
# =========================================================

verify(
    ["V001", "V002"],
    "Hyundai India official CRETA Electric specifications",
    {
        "wheelbase_mm": {
            "V001": 2610,
            "V002": 2610
        },
        "battery_capacity_kwh": {
            "V001": 42.0,
            "V002": 51.4
        },
        "certified_range_km": {
            "V001": 420,
            "V002": 510
        },
        "power_kw": {
            "V001": 99,
            "V002": 126
        }
    }
)

# =========================================================
# TATA PUNCH EV
# =========================================================

verify(
    ["V004", "V005"],
    "Tata.ev official Punch.ev specifications",
    {
        "wheelbase_mm": {
            "V004": 2445,
            "V005": 2445
        },
        "ground_clearance_mm": {
            "V004": 195,
            "V005": 195
        },
        "luggage_volume_l": {
            "V004": 366,
            "V005": 366
        },
        "battery_capacity_kwh": {
            "V004": 30,
            "V005": 40
        },
        "power_kw": {
            "V004": 65,
            "V005": 95
        },
        "certified_range_km": {
            "V005": 468
        }
    }
)

# =========================================================
# KIA CARENS
# =========================================================

verify(
    ["V036", "V037", "V038", "V039", "V040"],
    "Kia India official Carens specifications",
    {
        "wheelbase_mm": {
            "V036": 2780,
            "V037": 2780,
            "V038": 2780,
            "V039": 2780,
            "V040": 2780
        }
    }
)

# =========================================================
# TOYOTA URBAN CRUISER HYRYDER STRONG HYBRID
# =========================================================

verify(
    ["V041"],
    "Toyota Bharat official Urban Cruiser Hyryder specifications",
    {
        "wheelbase_mm": {
            "V041": 2600
        },
        "fuel_tank_l": {
            "V041": 45
        },
        "fuel_consumption_l_100km": {
            "V041": round(100 / 27.97, 3)
        },
        "power_kw": {
            "V041": 85
        }
    }
)

# =========================================================
# MARUTI GRAND VITARA STRONG HYBRID
# =========================================================

verify(
    ["V042"],
    "Maruti Suzuki official Grand Vitara specifications",
    {
        "fuel_consumption_l_100km": {
            "V042": round(100 / 27.97, 3)
        }
    }
)

# =========================================================
# DERIVED EV ENERGY CONSUMPTION
# =========================================================
#
# This is NOT an independently measured consumption value.
# It is calculated from:
#
# battery capacity / certified range * 100
#
# We therefore label it as DERIVED rather than OEM data.
# =========================================================

ev_mask = (
    (df["powertrain"] == "BEV")
    & df["battery_capacity_kwh"].notna()
    & df["certified_range_km"].notna()
    & (df["certified_range_km"] > 0)
)

df.loc[ev_mask, "energy_consumption_kwh_100km"] = (
    df.loc[ev_mask, "battery_capacity_kwh"]
    / df.loc[ev_mask, "certified_range_km"]
    * 100
)

# ---------------------------------------------------------
# Derived power-to-weight only where weight exists
# ---------------------------------------------------------

if "kerb_weight_kg" in df.columns:

    mask = (
        df["power_kw"].notna()
        & df["kerb_weight_kg"].notna()
        & (df["kerb_weight_kg"] > 0)
    )

    df["power_to_weight_kw_per_tonne"] = np.nan

    df.loc[mask, "power_to_weight_kw_per_tonne"] = (
        df.loc[mask, "power_kw"]
        / df.loc[mask, "kerb_weight_kg"]
        * 1000
    )

# ---------------------------------------------------------
# Data-quality warning
# ---------------------------------------------------------

important = [
    "kerb_weight_kg",
    "wheelbase_mm",
    "battery_capacity_kwh",
    "certified_range_km",
    "power_kw",
    "fuel_consumption_l_100km"
]

df["critical_missing_count"] = df[important].isna().sum(axis=1)

df["model_ready"] = np.where(
    df["critical_missing_count"] <= 1,
    "yes",
    "no"
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUT, index=False)

print("\nVERIFIED VEHICLE DATASET CREATED")
print("----------------------------------")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nVerification status:")
print(df["verification_status"].value_counts())

print("\nModel-ready vehicles:")
print(df["model_ready"].value_counts())

print("\nCritical missing values:")
print(
    df[important]
    .isna()
    .sum()
)

print("\nSaved:")
print(OUT)