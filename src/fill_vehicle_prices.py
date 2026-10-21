import pandas as pd
from pathlib import Path
from datetime import date


# ============================================================
# FILL PURCHASE PRICES FOR ALL 48 VEHICLES
# E-Track - Ideal Ride Service Vehicle for India 2035
# ============================================================

INPUT_FILE = Path("data/processed/vehicles_final.csv")
OUTPUT_FILE = Path("data/processed/vehicles_final.csv")
BACKUP_FILE = Path("data/processed/vehicles_final_before_prices.csv")


# ============================================================
# PRICE DATABASE
# ============================================================
#
# Values are CURRENT BASELINE EX-SHOWROOM PRICE ESTIMATES
# in INR for the specified model/variant.
#
# They are intended for fleet-economic MODELING.
#
# They are NOT on-road prices.
# They do not include:
#   - road tax
#   - registration
#   - insurance
#   - accessories
#   - financing
#   - dealer-specific discounts
#   - fleet discounts
#
# For variants where an exact trim price is not available,
# an engineering/model-level estimate is used.
#
# IMPORTANT:
# These are BASELINE inputs for the economic model.
# Later we will run sensitivity analysis around them.
# ============================================================

PRICE_DATA = {

    # --------------------------------------------------------
    # BEV
    # --------------------------------------------------------

    "V001": {
        "price": 1802800,
        "source_type": "OEM_current_model_reference",
        "confidence": "High",
        "note": "Creta Electric 42 kWh; current OEM model starting-price reference"
    },

    "V002": {
        "price": 2000000,
        "source_type": "OEM_current_model_reference",
        "confidence": "Medium",
        "note": "Creta Electric 51.4 kWh long-range variant estimate"
    },

    "V003": {
        "price": 1399000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Nexon EV 45 kWh model-level ex-showroom estimate"
    },

    "V004": {
        "price": 1099000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Punch EV 30 kWh model-level ex-showroom estimate"
    },

    "V005": {
        "price": 1299000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Punch EV 40 kWh model-level ex-showroom estimate"
    },

    "V006": {
        "price": 1749000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Curvv EV 45 kWh model-level estimate"
    },

    "V007": {
        "price": 1949000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Curvv EV 55 kWh model-level estimate"
    },

    "V008": {
        "price": 1499000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Windsor EV 38 kWh model-level estimate"
    },

    "V009": {
        "price": 1849000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Windsor EV PRO 52.9 kWh model-level estimate"
    },

    "V010": {
        "price": 1999000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "MG ZS EV 50.3 kWh model-level estimate"
    },

    "V011": {
        "price": 1599000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "XUV400 34.5 kWh model-level estimate"
    },

    "V012": {
        "price": 1699000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "XUV400 39.4 kWh model-level estimate"
    },

    "V013": {
        "price": 1899000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "BE 6 59 kWh model-level estimate"
    },

    "V014": {
        "price": 2199000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "BE 6 79 kWh model-level estimate"
    },

    "V015": {
        "price": 2199000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "XEV 9e 59 kWh model-level estimate"
    },

    "V016": {
        "price": 2499000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "XEV 9e 79 kWh model-level estimate"
    },

    "V017": {
        "price": 799000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Tiago EV 19.2 kWh model-level estimate"
    },

    "V018": {
        "price": 899000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Tiago EV 24 kWh model-level estimate"
    },

    "V019": {
        "price": 1299000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Tigor EV 26 kWh model-level estimate"
    },


    # --------------------------------------------------------
    # CNG
    # --------------------------------------------------------

    "V020": {
        "price": 799000,
        "source_type": "OEM_current_model_reference",
        "confidence": "Medium",
        "note": "Aura CNG model-level estimate"
    },

    "V021": {
        "price": 799000,
        "source_type": "OEM_current_model_reference",
        "confidence": "Medium",
        "note": "Dzire CNG model-level estimate"
    },

    "V022": {
        "price": 1099000,
        "source_type": "OEM_current_model_reference",
        "confidence": "Medium",
        "note": "Ertiga CNG model-level estimate"
    },

    "V023": {
        "price": 1199000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "XL6 CNG model-level estimate"
    },

    "V024": {
        "price": 1099000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Brezza CNG model-level estimate"
    },

    "V025": {
        "price": 1199000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Rumion CNG model-level estimate"
    },


    # --------------------------------------------------------
    # PETROL
    # --------------------------------------------------------

    "V026": {
        "price": 599990,
        "source_type": "OEM_current_model_reference",
        "confidence": "High",
        "note": "Aura Petrol current Hyundai starting-price reference"
    },

    "V027": {
        "price": 679000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Dzire Petrol model-level estimate"
    },

    "V028": {
        "price": 809000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Amaze Petrol MT model-level estimate"
    },

    "V029": {
        "price": 949000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Amaze Petrol CVT model-level estimate"
    },

    "V030": {
        "price": 1099200,
        "source_type": "OEM_current_model_reference",
        "confidence": "High",
        "note": "Verna 1.5 MPi current Hyundai model starting-price reference"
    },

    "V031": {
        "price": 1250000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Verna 1.5 MPi IVT model-level estimate"
    },

    "V032": {
        "price": 1300000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Verna 1.5 Turbo MT model-level estimate"
    },

    "V033": {
        "price": 1450000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Verna 1.5 Turbo DCT model-level estimate"
    },

    "V034": {
        "price": 1199000,
        "source_type": "OEM_current_model_reference",
        "confidence": "Medium",
        "note": "Creta 1.5 MPi model-level current-price estimate"
    },

    "V035": {
        "price": 1800000,
        "source_type": "OEM_current_model_reference",
        "confidence": "Medium",
        "note": "Creta 1.5 Turbo Petrol model-level estimate"
    },

    "V036": {
        "price": 1099000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Carens 1.5 Petrol model-level estimate"
    },

    "V037": {
        "price": 1699000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Carens 1.5 Turbo Petrol MT model-level estimate"
    },

    "V038": {
        "price": 1899000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Carens 1.5 Turbo Petrol DCT model-level estimate"
    },


    # --------------------------------------------------------
    # DIESEL
    # --------------------------------------------------------

    "V039": {
        "price": 1750000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Carens 1.5 Diesel MT model-level estimate"
    },

    "V040": {
        "price": 1950000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Carens 1.5 Diesel AT model-level estimate"
    },


    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    "V041": {
        "price": 1950000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Urban Cruiser Hyryder strong-hybrid model-level estimate"
    },

    "V042": {
        "price": 1950000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Grand Vitara strong-hybrid model-level estimate"
    },

    "V043": {
        "price": 2050000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Honda City e:HEV model-level estimate"
    },

    "V044": {
        "price": 3000000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Innova Hycross Hybrid model-level estimate"
    },


    # --------------------------------------------------------
    # ADDITIONAL PETROL
    # --------------------------------------------------------

    "V045": {
        "price": 1900000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Innova Hycross Petrol model-level estimate"
    },

    "V046": {
        "price": 1050000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Ertiga Petrol model-level estimate"
    },

    "V047": {
        "price": 1150000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "XL6 Petrol model-level estimate"
    },

    "V048": {
        "price": 1000000,
        "source_type": "current_market_reference",
        "confidence": "Medium",
        "note": "Brezza Petrol model-level estimate"
    },
}


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"File not found: {INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)


print("=" * 70)
print("VEHICLE PRICE DATABASE UPDATE")
print("=" * 70)

print(
    f"\nVehicles in dataset: {len(df)}"
)

print(
    f"Prices in database:  {len(PRICE_DATA)}"
)


# ============================================================
# BACKUP
# ============================================================

df.to_csv(
    BACKUP_FILE,
    index=False
)

print(
    f"\nBackup created:\n{BACKUP_FILE}"
)


# ============================================================
# VERIFY VEHICLE IDS
# ============================================================

dataset_ids = set(
    df["vehicle_id"].astype(str)
)

price_ids = set(
    PRICE_DATA.keys()
)


missing_price_database = (
    dataset_ids - price_ids
)

extra_price_database = (
    price_ids - dataset_ids
)


if missing_price_database:

    raise ValueError(
        "These vehicle IDs have no price entry:\n"
        + str(sorted(missing_price_database))
    )


if extra_price_database:

    print(
        "\nWarning: price database contains unused IDs:"
    )

    print(
        sorted(extra_price_database)
    )


# ============================================================
# FILL PRICES
# ============================================================

df["purchase_price_inr"] = (
    df["vehicle_id"]
    .map(
        lambda x:
        PRICE_DATA[str(x)]["price"]
    )
)


# ============================================================
# ADD PRICE METADATA
# ============================================================

df["purchase_price_source_type"] = (
    df["vehicle_id"]
    .map(
        lambda x:
        PRICE_DATA[str(x)]["source_type"]
    )
)


df["purchase_price_confidence"] = (
    df["vehicle_id"]
    .map(
        lambda x:
        PRICE_DATA[str(x)]["confidence"]
    )
)


df["purchase_price_note"] = (
    df["vehicle_id"]
    .map(
        lambda x:
        PRICE_DATA[str(x)]["note"]
    )
)


df["purchase_price_date"] = (
    date.today().isoformat()
)


# ============================================================
# VALIDATION
# ============================================================

missing_prices = (
    df["purchase_price_inr"]
    .isna()
    .sum()
)


print(
    f"\nMissing prices after update: "
    f"{missing_prices}"
)


if missing_prices != 0:

    raise ValueError(
        "Price update failed."
    )


# ============================================================
# PRICE STATISTICS
# ============================================================

print("\nPrice statistics:")

print(
    f"Minimum : Rs {df['purchase_price_inr'].min():,.0f}"
)

print(
    f"Maximum : Rs {df['purchase_price_inr'].max():,.0f}"
)

print(
    f"Average : Rs {df['purchase_price_inr'].mean():,.0f}"
)


# ============================================================
# POWERTRAIN PRICE SUMMARY
# ============================================================

print("\nAverage purchase price by powertrain:")

price_summary = (

    df
    .groupby("powertrain")
    ["purchase_price_inr"]
    .agg(
        ["count", "mean", "min", "max"]
    )
    .sort_values(
        "mean"
    )

)

print(
    price_summary.to_string()
)


# ============================================================
# SHOW ALL 48
# ============================================================

print("\n" + "=" * 70)
print("UPDATED VEHICLE PRICE DATABASE")
print("=" * 70)

print(

    df[
        [
            "vehicle_id",
            "model",
            "powertrain",
            "purchase_price_inr",
            "purchase_price_confidence"
        ]
    ]
    .to_string(index=False)

)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("PRICE UPDATE COMPLETE")
print("=" * 70)

print(
    f"\nSaved:\n{OUTPUT_FILE}"
)

print(
    f"\nBackup:\n{BACKUP_FILE}"
)

print(
    "\nAll 48 vehicles now have purchase-price data."
)

print(
    "\nPrice values are baseline ex-showroom modeling "
    "inputs and will later be subjected to sensitivity "
    "analysis and 2035 scenario adjustment."
)

print("=" * 70)