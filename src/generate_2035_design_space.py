import pandas as pd
import numpy as np

from pathlib import Path


# ============================================================
# 2035 HYPOTHETICAL VEHICLE DESIGN SPACE
# ============================================================

INPUT_FILE = Path(
    "data/processed/vehicles_final.csv"
)

OUTPUT_FILE = Path(
    "data/processed/design_space_2035.csv"
)

N_DESIGNS = 5000

RANDOM_SEED = 42


print("=" * 70)
print("2035 HYPOTHETICAL VEHICLE DESIGN SPACE")
print("=" * 70)


# ------------------------------------------------------------
# LOAD REAL VEHICLE DATA
# ------------------------------------------------------------

vehicles = pd.read_csv(INPUT_FILE)

print(
    f"\nReference vehicles: {len(vehicles)}"
)


rng = np.random.default_rng(
    RANDOM_SEED
)


# ------------------------------------------------------------
# REFERENCE RANGES
# ------------------------------------------------------------

print("\nReference engineering ranges:")

for col in [
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
]:

    print(
        f"{col}: "
        f"{vehicles[col].min():.1f}"
        f" – "
        f"{vehicles[col].max():.1f}"
    )


# ------------------------------------------------------------
# POWERTRAIN
# ------------------------------------------------------------

powertrains = [
    "BEV",
    "CNG",
    "Hybrid",
    "Petrol"
]

powertrain_probability = [
    0.55,
    0.20,
    0.20,
    0.05
]


# ------------------------------------------------------------
# GENERATE DESIGNS
# ------------------------------------------------------------

designs = []


for i in range(N_DESIGNS):

    powertrain = rng.choice(
        powertrains,
        p=powertrain_probability
    )


    # --------------------------------------------------------
    # BASIC VEHICLE ARCHITECTURE
    # --------------------------------------------------------

    seats = int(
        rng.choice(
            [4, 5, 6, 7],
            p=[
                0.10,
                0.50,
                0.15,
                0.25
            ]
        )
    )


    wheelbase = rng.uniform(
        2450,
        2850
    )


    luggage_volume = rng.uniform(
        300,
        650
    )


    kerb_weight = rng.uniform(
        1050,
        2000
    )


    power_kw = rng.uniform(
        60,
        160
    )


    purchase_price = rng.uniform(
        800000,
        2500000
    )


    # --------------------------------------------------------
    # POWERTRAIN-SPECIFIC DESIGN
    # --------------------------------------------------------

    if powertrain == "BEV":

        battery = rng.uniform(
            25,
            80
        )

        certified_range = rng.uniform(
            300,
            700
        )


    elif powertrain == "CNG":

        battery = np.nan

        certified_range = np.nan


    elif powertrain == "Hybrid":

        battery = rng.uniform(
            1.5,
            5
        )

        certified_range = np.nan


    else:

        battery = np.nan

        certified_range = np.nan


    designs.append({

        "design_id":
            f"D{i+1:05d}",

        "powertrain":
            powertrain,

        "vehicle_type":
            "ride_service",

        "seats":
            seats,

        "kerb_weight_kg":
            kerb_weight,

        "wheelbase_mm":
            wheelbase,

        "luggage_volume_l":
            luggage_volume,

        "battery_capacity_kwh":
            battery,

        "certified_range_km":
            certified_range,

        "power_kw":
            power_kw,

        "purchase_price_inr":
            purchase_price

    })


df = pd.DataFrame(
    designs
)


# ------------------------------------------------------------
# ENGINEERING CONSTRAINTS
# ------------------------------------------------------------

# Power-to-weight ratio

df["power_to_weight_kw_per_tonne"] = (

    df["power_kw"] /
    (df["kerb_weight_kg"] / 1000)

)


# ------------------------------------------------------------
# REMOVE UNREALISTIC DESIGNS
# ------------------------------------------------------------

valid = (

    (df["power_to_weight_kw_per_tonne"] >= 40)

    &

    (df["power_to_weight_kw_per_tonne"] <= 120)

)


# BEV battery relationship

bev_mask = (
    df["powertrain"] == "BEV"
)


valid &= (

    (~bev_mask)

    |

    (
        (
            df["battery_capacity_kwh"]
            >= 25
        )

        &

        (
            df["battery_capacity_kwh"]
            <= 80
        )
    )

)


# BEV range relationship

valid &= (

    (~bev_mask)

    |

    (
        df["certified_range_km"]
        >=
        4.5 *
        df["battery_capacity_kwh"]
    )

)


# Large vehicles need adequate mass

valid &= (

    (
        df["seats"] <= 5
    )

    |

    (
        df["kerb_weight_kg"] >= 1250
    )

)


# 7-seat vehicles need adequate luggage space

valid &= (

    (
        df["seats"] < 7
    )

    |

    (
        df["luggage_volume_l"] >= 400
    )

)


df = df[
    valid
].reset_index(
    drop=True
)


# ------------------------------------------------------------
# FINAL DESIGN IDs
# ------------------------------------------------------------

df["design_id"] = [

    f"D{i+1:05d}"

    for i in range(len(df))

]


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)

print("DESIGN SPACE COMPLETE")

print("=" * 70)


print(
    f"\nGenerated initially : {N_DESIGNS}"
)

print(
    f"Feasible designs    : {len(df)}"
)


print("\nPowertrain distribution:")

print(
    df["powertrain"]
    .value_counts()
)


print("\nDesign ranges:")

for col in [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr"
]:

    print(
        f"{col:25s}: "
        f"{df[col].min():.2f}"
        f" – "
        f"{df[col].max():.2f}"
    )


print("\nBEV battery range:")

bev = df[
    df["powertrain"] == "BEV"
]

print(
    f"Battery: "
    f"{bev['battery_capacity_kwh'].min():.2f}"
    f" – "
    f"{bev['battery_capacity_kwh'].max():.2f} kWh"
)

print(
    f"Range: "
    f"{bev['certified_range_km'].min():.2f}"
    f" – "
    f"{bev['certified_range_km'].max():.2f} km"
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved:"
    f"\n  {OUTPUT_FILE}"
)


print("\nNext step:")

print(
    "Evaluate hypothetical designs using the "
    "ML surrogate models."
)


print("=" * 70)