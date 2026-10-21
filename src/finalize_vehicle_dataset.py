import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/vehicles_verified.csv")
OUTPUT = Path("data/processed/vehicles_final.csv")

df = pd.read_csv(INPUT)

# ============================================================
# FINAL VEHICLE DATASET
# ============================================================
#
# Purpose:
# Complete the engineering dataset for the 48 representative
# vehicle variants used in the India 2035 ride-service study.
#
# IMPORTANT:
# - OEM/manual values are preferred.
# - Where an exact variant-specific value is unavailable,
#   a representative engineering value is used and flagged.
# - No missing value is silently converted into an average.
# - CNG efficiency is stored separately as kg/100 km.
# ============================================================


# ------------------------------------------------------------
# Add provenance columns
# ------------------------------------------------------------

new_columns = {
    "data_quality": "",
    "specification_source_type": "",
    "verification_note": "",
}

for col, default in new_columns.items():
    if col not in df.columns:
        df[col] = default


# ------------------------------------------------------------
# Complete specification table
#
# Values:
# kerb_weight_kg
# wheelbase_mm
# power_kw
# fuel_tank_l
# luggage_volume_l
# fuel_consumption_l_100km
# fuel_consumption_kg_100km
#
# EV:
# battery_capacity_kwh
# certified_range_km
# power_kw
# ------------------------------------------------------------

specs = {

    # ========================================================
    # BEVs
    # ========================================================

    "V001": {
        "kerb_weight_kg": 1650,
        "wheelbase_mm": 2610,
        "power_kw": 99,
        "luggage_volume_l": 433,
        "data_quality": "OEM/official",
        "source_type": "OEM",
        "note": "Hyundai official specifications"
    },

    "V002": {
        "kerb_weight_kg": 1725,
        "wheelbase_mm": 2610,
        "power_kw": 126,
        "luggage_volume_l": 433,
        "data_quality": "OEM/official",
        "source_type": "OEM",
        "note": "Hyundai official specifications"
    },

    "V003": {
        "kerb_weight_kg": 1585,
        "wheelbase_mm": 2498,
        "power_kw": 106,
        "luggage_volume_l": 350,
        "data_quality": "OEM/manual",
        "source_type": "OEM/manual",
        "note": "45 kWh Nexon EV specification"
    },

    "V004": {
        "kerb_weight_kg": 1340,
        "wheelbase_mm": 2445,
        "power_kw": 65,
        "ground_clearance_mm": 195,
        "luggage_volume_l": 366,
        "data_quality": "OEM/official",
        "source_type": "OEM",
        "note": "Tata official specification"
    },

    "V005": {
        "kerb_weight_kg": 1390,
        "wheelbase_mm": 2445,
        "power_kw": 95,
        "ground_clearance_mm": 195,
        "luggage_volume_l": 366,
        "data_quality": "OEM/official",
        "source_type": "OEM",
        "note": "Tata official specification"
    },

    "V006": {
        "kerb_weight_kg": 1860,
        "wheelbase_mm": 2560,
        "power_kw": 110,
        "luggage_volume_l": 500,
        "data_quality": "secondary/manual",
        "source_type": "Tata manual + specification sources",
        "note": "45 kWh Curvv EV"
    },

    "V007": {
        "kerb_weight_kg": 1900,
        "wheelbase_mm": 2560,
        "power_kw": 123,
        "luggage_volume_l": 500,
        "data_quality": "secondary/manual",
        "source_type": "Tata manual + specification sources",
        "note": "55 kWh Curvv EV"
    },

    "V008": {
        "kerb_weight_kg": 1499,
        "wheelbase_mm": 2700,
        "power_kw": 100,
        "luggage_volume_l": 604,
        "data_quality": "secondary",
        "source_type": "vehicle specification sources",
        "note": "38 kWh Windsor EV"
    },

    "V009": {
        "kerb_weight_kg": 1620,
        "wheelbase_mm": 2700,
        "power_kw": 100,
        "luggage_volume_l": 604,
        "data_quality": "secondary",
        "source_type": "vehicle specification sources",
        "note": "52.9 kWh Windsor EV PRO"
    },

    "V010": {
        "kerb_weight_kg": 1620,
        "wheelbase_mm": 2585,
        "power_kw": 130,
        "luggage_volume_l": 448,
        "data_quality": "secondary",
        "source_type": "vehicle specification sources",
        "note": "50.3 kWh ZS EV"
    },

    "V011": {
        "kerb_weight_kg": 1500,
        "wheelbase_mm": 2600,
        "power_kw": 110,
        "luggage_volume_l": 378,
        "data_quality": "secondary",
        "source_type": "vehicle specification sources",
        "note": "34.5 kWh XUV400"
    },

    "V012": {
        "kerb_weight_kg": 1610,
        "wheelbase_mm": 2600,
        "power_kw": 110,
        "luggage_volume_l": 378,
        "data_quality": "secondary",
        "source_type": "vehicle specification sources",
        "note": "39.4 kWh XUV400"
    },

    "V013": {
        "kerb_weight_kg": 2000,
        "wheelbase_mm": 2775,
        "power_kw": 170,
        "luggage_volume_l": 455,
        "data_quality": "secondary",
        "source_type": "Mahindra specification sources",
        "note": "59 kWh BE 6"
    },

    "V014": {
        "kerb_weight_kg": 2100,
        "wheelbase_mm": 2775,
        "power_kw": 210,
        "luggage_volume_l": 455,
        "data_quality": "secondary",
        "source_type": "Mahindra specification sources",
        "note": "79 kWh BE 6"
    },

    "V015": {
        "kerb_weight_kg": 2100,
        "wheelbase_mm": 2775,
        "power_kw": 170,
        "luggage_volume_l": 663,
        "data_quality": "secondary",
        "source_type": "Mahindra specification sources",
        "note": "59 kWh XEV 9e"
    },

    "V016": {
        "kerb_weight_kg": 2200,
        "wheelbase_mm": 2775,
        "power_kw": 210,
        "luggage_volume_l": 663,
        "data_quality": "secondary",
        "source_type": "Mahindra specification sources",
        "note": "79 kWh XEV 9e"
    },

    "V017": {
        "kerb_weight_kg": 1140,
        "wheelbase_mm": 2400,
        "power_kw": 45,
        "luggage_volume_l": 240,
        "data_quality": "OEM/manual",
        "source_type": "Tata manual",
        "note": "19.2 kWh Tiago EV"
    },

    "V018": {
        "kerb_weight_kg": 1170,
        "wheelbase_mm": 2400,
        "power_kw": 55,
        "luggage_volume_l": 240,
        "data_quality": "OEM/manual",
        "source_type": "Tata manual",
        "note": "24 kWh Tiago EV"
    },

    "V019": {
        "kerb_weight_kg": 1235,
        "wheelbase_mm": 2450,
        "power_kw": 55,
        "luggage_volume_l": 316,
        "data_quality": "OEM/manual",
        "source_type": "Tata manual",
        "note": "26 kWh Tigor EV"
    },


    # ========================================================
    # CNG
    # ========================================================

    "V020": {
        "kerb_weight_kg": 1050,
        "wheelbase_mm": 2450,
        "power_kw": 50.5,
        "fuel_tank_l": 37,
        "fuel_consumption_kg_100km": 100 / 28.0,
        "luggage_volume_l": 341,
        "data_quality": "OEM/secondary",
        "source_type": "Hyundai specification",
        "note": "CNG efficiency represented separately in kg/100 km"
    },

    "V021": {
        "kerb_weight_kg": 1022.5,
        "wheelbase_mm": 2450,
        "power_kw": 74,
        "fuel_tank_l": 55,
        "fuel_consumption_kg_100km": 100 / 33.73,
        "luggage_volume_l": 382,
        "data_quality": "OEM",
        "source_type": "Maruti Suzuki",
        "note": "Current Dzire CNG efficiency 33.73 km/kg"
    },

    "V022": {
        "kerb_weight_kg": 1252.5,
        "wheelbase_mm": 2740,
        "power_kw": 64.6,
        "fuel_tank_l": 60,
        "fuel_consumption_kg_100km": 100 / 26.11,
        "luggage_volume_l": 550,
        "data_quality": "OEM",
        "source_type": "Maruti Suzuki",
        "note": "Ertiga CNG"
    },

    "V023": {
        "kerb_weight_kg": 1250,
        "wheelbase_mm": 2740,
        "power_kw": 64.6,
        "fuel_tank_l": 60,
        "fuel_consumption_kg_100km": 100 / 26.32,
        "luggage_volume_l": 550,
        "data_quality": "OEM/secondary",
        "source_type": "Maruti Suzuki",
        "note": "XL6 CNG"
    },

    "V024": {
        "kerb_weight_kg": 1250,
        "wheelbase_mm": 2500,
        "power_kw": 64.6,
        "fuel_tank_l": 55,
        "fuel_consumption_kg_100km": 100 / 25.51,
        "luggage_volume_l": 328,
        "data_quality": "secondary",
        "source_type": "Maruti specification sources",
        "note": "Brezza CNG"
    },

    "V025": {
        "kerb_weight_kg": 1270,
        "wheelbase_mm": 2740,
        "power_kw": 64.6,
        "fuel_tank_l": 60,
        "fuel_consumption_kg_100km": 100 / 26.11,
        "luggage_volume_l": 550,
        "data_quality": "secondary/OEM-derived",
        "source_type": "Toyota/Maruti specification",
        "note": "Rumion CNG"
    },


    # ========================================================
    # PETROL / DIESEL
    # ========================================================

    "V026": {
        "kerb_weight_kg": 930,
        "wheelbase_mm": 2450,
        "power_kw": 61,
        "fuel_tank_l": 37,
        "fuel_consumption_l_100km": 100 / 19.5,
        "luggage_volume_l": 402,
        "data_quality": "OEM",
        "source_type": "Hyundai",
        "note": "Aura petrol"
    },

    "V027": {
        "kerb_weight_kg": 940,
        "wheelbase_mm": 2450,
        "power_kw": 75.8,
        "fuel_tank_l": 37,
        "fuel_consumption_l_100km": 100 / 24.79,
        "luggage_volume_l": 382,
        "data_quality": "OEM/official",
        "source_type": "Maruti Suzuki",
        "note": "Dzire petrol"
    },

    "V028": {
        "kerb_weight_kg": 969,
        "wheelbase_mm": 2470,
        "power_kw": 66,
        "fuel_tank_l": 35,
        "fuel_consumption_l_100km": 100 / 19.46,
        "luggage_volume_l": 416,
        "data_quality": "OEM",
        "source_type": "Honda",
        "note": "Amaze petrol MT"
    },

    "V029": {
        "kerb_weight_kg": 969,
        "wheelbase_mm": 2470,
        "power_kw": 66,
        "fuel_tank_l": 35,
        "fuel_consumption_l_100km": 100 / 18.65,
        "luggage_volume_l": 416,
        "data_quality": "OEM",
        "source_type": "Honda",
        "note": "Amaze petrol CVT"
    },

    "V030": {
        "kerb_weight_kg": 1165,
        "wheelbase_mm": 2670,
        "power_kw": 85,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 18.6,
        "luggage_volume_l": 528,
        "data_quality": "OEM/secondary",
        "source_type": "Hyundai",
        "note": "Verna 1.5 MPi MT"
    },

    "V031": {
        "kerb_weight_kg": 1185,
        "wheelbase_mm": 2670,
        "power_kw": 85,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 18.6,
        "luggage_volume_l": 528,
        "data_quality": "OEM/secondary",
        "source_type": "Hyundai",
        "note": "Verna 1.5 MPi IVT"
    },

    "V032": {
        "kerb_weight_kg": 1230,
        "wheelbase_mm": 2670,
        "power_kw": 117.5,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 20.0,
        "luggage_volume_l": 528,
        "data_quality": "OEM/secondary",
        "source_type": "Hyundai",
        "note": "Verna 1.5 Turbo MT"
    },

    "V033": {
        "kerb_weight_kg": 1250,
        "wheelbase_mm": 2670,
        "power_kw": 117.5,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 20.0,
        "luggage_volume_l": 528,
        "data_quality": "OEM/secondary",
        "source_type": "Hyundai",
        "note": "Verna 1.5 Turbo DCT"
    },

    "V034": {
        "kerb_weight_kg": 1350,
        "wheelbase_mm": 2610,
        "power_kw": 85,
        "fuel_tank_l": 50,
        "fuel_consumption_l_100km": 100 / 17.4,
        "luggage_volume_l": 433,
        "data_quality": "OEM/secondary",
        "source_type": "Hyundai",
        "note": "Creta 1.5 MPi"
    },

    "V035": {
        "kerb_weight_kg": 1400,
        "wheelbase_mm": 2610,
        "power_kw": 117.5,
        "fuel_tank_l": 50,
        "fuel_consumption_l_100km": 100 / 18.4,
        "luggage_volume_l": 433,
        "data_quality": "OEM/secondary",
        "source_type": "Hyundai",
        "note": "Creta 1.5 Turbo"
    },

    "V036": {
        "kerb_weight_kg": 1500,
        "wheelbase_mm": 2780,
        "power_kw": 85,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 15.7,
        "luggage_volume_l": 216,
        "data_quality": "OEM/secondary",
        "source_type": "Kia",
        "note": "Carens 1.5 petrol"
    },

    "V037": {
        "kerb_weight_kg": 1520,
        "wheelbase_mm": 2780,
        "power_kw": 118,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 16.5,
        "luggage_volume_l": 216,
        "data_quality": "OEM/secondary",
        "source_type": "Kia",
        "note": "Carens 1.5 turbo petrol MT"
    },

    "V038": {
        "kerb_weight_kg": 1540,
        "wheelbase_mm": 2780,
        "power_kw": 118,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 15.8,
        "luggage_volume_l": 216,
        "data_quality": "OEM/secondary",
        "source_type": "Kia",
        "note": "Carens 1.5 turbo petrol DCT"
    },

    "V039": {
        "kerb_weight_kg": 1600,
        "wheelbase_mm": 2780,
        "power_kw": 85,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 21.3,
        "luggage_volume_l": 216,
        "data_quality": "OEM/secondary",
        "source_type": "Kia",
        "note": "Carens diesel MT"
    },

    "V040": {
        "kerb_weight_kg": 1620,
        "wheelbase_mm": 2780,
        "power_kw": 85,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 18.4,
        "luggage_volume_l": 216,
        "data_quality": "OEM/secondary",
        "source_type": "Kia",
        "note": "Carens diesel AT"
    },


    # ========================================================
    # HYBRIDS
    # ========================================================

    "V041": {
        "kerb_weight_kg": 1292,
        "wheelbase_mm": 2600,
        "power_kw": 85,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 27.97,
        "luggage_volume_l": 373,
        "data_quality": "OEM",
        "source_type": "Toyota/Maruti",
        "note": "Strong hybrid"
    },

    "V042": {
        "kerb_weight_kg": 1292,
        "wheelbase_mm": 2600,
        "power_kw": 85,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 27.97,
        "luggage_volume_l": 373,
        "data_quality": "OEM",
        "source_type": "Maruti Suzuki",
        "note": "Strong hybrid"
    },

    "V043": {
        "kerb_weight_kg": 1280,
        "wheelbase_mm": 2600,
        "power_kw": 93,
        "fuel_tank_l": 40,
        "fuel_consumption_l_100km": 100 / 26.5,
        "luggage_volume_l": 306,
        "data_quality": "OEM",
        "source_type": "Honda",
        "note": "City e:HEV"
    },

    "V044": {
        "kerb_weight_kg": 1750,
        "wheelbase_mm": 2850,
        "power_kw": 137,
        "fuel_tank_l": 52,
        "fuel_consumption_l_100km": 100 / 23.24,
        "luggage_volume_l": 300,
        "data_quality": "OEM/secondary",
        "source_type": "Toyota",
        "note": "Innova Hycross hybrid"
    },

    "V045": {
        "kerb_weight_kg": 1650,
        "wheelbase_mm": 2850,
        "power_kw": 126,
        "fuel_tank_l": 52,
        "fuel_consumption_l_100km": 100 / 16.13,
        "luggage_volume_l": 300,
        "data_quality": "OEM/secondary",
        "source_type": "Toyota",
        "note": "Innova Hycross petrol"
    },

    "V046": {
        "kerb_weight_kg": 1177.5,
        "wheelbase_mm": 2740,
        "power_kw": 75.8,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 20.51,
        "luggage_volume_l": 550,
        "data_quality": "OEM",
        "source_type": "Maruti Suzuki",
        "note": "Ertiga petrol"
    },

    "V047": {
        "kerb_weight_kg": 1175,
        "wheelbase_mm": 2740,
        "power_kw": 75.8,
        "fuel_tank_l": 45,
        "fuel_consumption_l_100km": 100 / 20.97,
        "luggage_volume_l": 550,
        "data_quality": "OEM/secondary",
        "source_type": "Maruti Suzuki",
        "note": "XL6 petrol"
    },

    "V048": {
        "kerb_weight_kg": 1180,
        "wheelbase_mm": 2500,
        "power_kw": 75.8,
        "fuel_tank_l": 48,
        "fuel_consumption_l_100km": 100 / 17.8,
        "luggage_volume_l": 328,
        "data_quality": "OEM/secondary",
        "source_type": "Maruti Suzuki",
        "note": "Brezza petrol"
    },
}


# ============================================================
# Apply specifications
# ============================================================

for vehicle_id, values in specs.items():

    mask = df["vehicle_id"] == vehicle_id

    if not mask.any():
        print("WARNING:", vehicle_id, "not found")
        continue

    for key, value in values.items():

        if key == "data_quality":
            df.loc[mask, "data_quality"] = value

        elif key == "source_type":
            df.loc[mask, "specification_source_type"] = value

        elif key == "note":
            df.loc[mask, "verification_note"] = value

        elif key in df.columns and value is not None:
            df.loc[mask, key] = value


# ============================================================
# Correct known EV specifications
# ============================================================

ev_corrections = {

    "V003": {
        "battery_capacity_kwh": 45,
        "certified_range_km": 489,
        "power_kw": 106,
    },

    "V004": {
        "battery_capacity_kwh": 30,
        "certified_range_km": 370,
        "power_kw": 65,
    },

    "V005": {
        "battery_capacity_kwh": 40,
        "certified_range_km": 468,
        "power_kw": 95,
    },

    "V006": {
        "battery_capacity_kwh": 45,
        "certified_range_km": 430,
        "power_kw": 110,
    },

    "V007": {
        "battery_capacity_kwh": 55,
        "certified_range_km": 502,
        "power_kw": 123,
    },

    "V008": {
        "battery_capacity_kwh": 38,
        "certified_range_km": 332,
        "power_kw": 100,
    },

    "V009": {
        "battery_capacity_kwh": 52.9,
        "certified_range_km": 449,
        "power_kw": 100,
    },

    "V010": {
        "battery_capacity_kwh": 50.3,
        "certified_range_km": 461,
        "power_kw": 130,
    },

    "V011": {
        "battery_capacity_kwh": 34.5,
        "certified_range_km": 375,
        "power_kw": 110,
    },

    "V012": {
        "battery_capacity_kwh": 39.4,
        "certified_range_km": 456,
        "power_kw": 110,
    },

    "V013": {
        "battery_capacity_kwh": 59,
        "certified_range_km": 557,
        "power_kw": 170,
    },

    "V014": {
        "battery_capacity_kwh": 79,
        "certified_range_km": 683,
        "power_kw": 210,
    },

    "V015": {
        "battery_capacity_kwh": 59,
        "certified_range_km": 542,
        "power_kw": 170,
    },

    "V016": {
        "battery_capacity_kwh": 79,
        "certified_range_km": 656,
        "power_kw": 210,
    },

    "V017": {
        "battery_capacity_kwh": 19.2,
        "certified_range_km": 250,
        "power_kw": 45,
    },

    "V018": {
        "battery_capacity_kwh": 24,
        "certified_range_km": 285,
        "power_kw": 55,
    },

    "V019": {
        "battery_capacity_kwh": 26,
        "certified_range_km": 315,
        "power_kw": 55,
    },
}

for vehicle_id, values in ev_corrections.items():

    mask = df["vehicle_id"] == vehicle_id

    for column, value in values.items():

        if column in df.columns:
            df.loc[mask, column] = value


# ============================================================
# Add CNG consumption column if missing
# ============================================================

if "fuel_consumption_kg_100km" not in df.columns:
    df["fuel_consumption_kg_100km"] = np.nan


# Recalculate CNG values explicitly
cng_efficiency = {

    "V020": 28.0,
    "V021": 33.73,
    "V022": 26.11,
    "V023": 26.32,
    "V024": 25.51,
    "V025": 26.11,
}

for vehicle_id, km_per_kg in cng_efficiency.items():

    mask = df["vehicle_id"] == vehicle_id

    df.loc[
        mask,
        "fuel_consumption_kg_100km"
    ] = 100 / km_per_kg


# ============================================================
# Calculate EV energy consumption
# ============================================================

df["energy_consumption_kwh_100km"] = np.nan

ev_mask = (
    (df["powertrain"].astype(str).str.upper() == "BEV")
    & df["battery_capacity_kwh"].notna()
    & df["certified_range_km"].notna()
)

df.loc[ev_mask, "energy_consumption_kwh_100km"] = (
    df.loc[ev_mask, "battery_capacity_kwh"]
    / df.loc[ev_mask, "certified_range_km"]
    * 100
)


# ============================================================
# Power-to-weight
# ============================================================

df["power_to_weight_kw_per_tonne"] = np.nan

valid_weight = (
    df["kerb_weight_kg"].notna()
    & df["power_kw"].notna()
    & (df["kerb_weight_kg"] > 0)
)

df.loc[valid_weight, "power_to_weight_kw_per_tonne"] = (
    df.loc[valid_weight, "power_kw"]
    / (df.loc[valid_weight, "kerb_weight_kg"] / 1000)
)


# ============================================================
# Payload proxy
# ============================================================
#
# We intentionally do NOT invent GVW.
#
# The existing payload column is therefore retained only when
# already present and sourced.
# ============================================================


# ============================================================
# Model readiness
# ============================================================

critical_fields = [
    "kerb_weight_kg",
    "wheelbase_mm",
    "power_kw",
]

# BEV-specific requirements
for idx, row in df.iterrows():

    if str(row["powertrain"]).upper() == "BEV":

        required = [
            "kerb_weight_kg",
            "wheelbase_mm",
            "battery_capacity_kwh",
            "certified_range_km",
            "power_kw",
        ]

    elif str(row["powertrain"]).upper() == "CNG":

        required = [
            "kerb_weight_kg",
            "wheelbase_mm",
            "power_kw",
            "fuel_consumption_kg_100km",
        ]

    else:

        required = [
            "kerb_weight_kg",
            "wheelbase_mm",
            "power_kw",
            "fuel_consumption_l_100km",
        ]

    missing = [
        col for col in required
        if col not in df.columns or pd.isna(row[col])
    ]

    df.loc[idx, "critical_missing_count"] = len(missing)

    if len(missing) == 0:
        df.loc[idx, "model_ready"] = "yes"
    else:
        df.loc[idx, "model_ready"] = "no"


# ============================================================
# Final sorting
# ============================================================

df = df.sort_values("vehicle_id").reset_index(drop=True)


# ============================================================
# Save
# ============================================================

df.to_csv(OUTPUT, index=False)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("FINAL VEHICLE DATASET CREATED")
print("=" * 60)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Powertrain distribution:")
print(df["powertrain"].value_counts())

print()
print("Data quality:")
print(df["data_quality"].value_counts())

print()
print("Critical missing values:")

for col in [
    "kerb_weight_kg",
    "wheelbase_mm",
    "battery_capacity_kwh",
    "certified_range_km",
    "power_kw",
    "fuel_consumption_l_100km",
    "fuel_consumption_kg_100km",
]:
    if col in df.columns:
        print(
            f"{col:35s}",
            df[col].isna().sum()
        )

print()
print("Model-ready:")
print(df["model_ready"].value_counts())

print()
print("EV energy consumption:")
print(
    df.loc[
        df["powertrain"] == "BEV",
        [
            "vehicle_id",
            "model",
            "battery_capacity_kwh",
            "certified_range_km",
            "energy_consumption_kwh_100km"
        ]
    ].to_string(index=False)
)

print()
print("CNG consumption:")
print(
    df.loc[
        df["powertrain"] == "CNG",
        [
            "vehicle_id",
            "model",
            "fuel_consumption_kg_100km"
        ]
    ].to_string(index=False)
)

print()
print("Saved:")
print(OUTPUT)

print()
print("IMPORTANT:")
print("vehicles_verified.csv was NOT overwritten.")
print("Final dataset is saved separately as:")
print("data/processed/vehicles_final.csv")