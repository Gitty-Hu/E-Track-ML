import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("INDIA 2035 REALISTIC DESIGN-SPACE REFINEMENT")
print("=" * 70)

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT = BASE_DIR / "data" / "processed" / "design_space_2035.csv"
OUTPUT = BASE_DIR / "data" / "processed" / "design_space_realistic_2035.csv"

df = pd.read_csv(INPUT)

print(f"Input designs: {len(df):,}")

# ============================================================
# 1. BASIC ENGINEERING CONSTRAINTS
# ============================================================

# Power-to-weight
df["power_to_weight_kw_per_tonne"] = (
    df["power_kw"] / df["kerb_weight_kg"] * 1000
)

check_power_weight = (
    (df["power_to_weight_kw_per_tonne"] >= 40)
    & (df["power_to_weight_kw_per_tonne"] <= 120)
)

# ============================================================
# 2. SEAT / MASS RELATIONSHIP
# ============================================================

minimum_mass = df["seats"].map({
    4: 1050,
    5: 1100,
    6: 1250,
    7: 1350
})

check_seat_mass = df["kerb_weight_kg"] >= minimum_mass

# ============================================================
# 3. SEAT / LUGGAGE RELATIONSHIP
# ============================================================

minimum_luggage = df["seats"].map({
    4: 300,
    5: 320,
    6: 380,
    7: 400
})

check_seat_luggage = df["luggage_volume_l"] >= minimum_luggage

# ============================================================
# 4. WHEELBASE / LUGGAGE
# ============================================================

check_wheelbase_luggage = (
    (df["wheelbase_mm"] < 2750)
    |
    (df["luggage_volume_l"] >= 400)
)

# ============================================================
# 5. POWER / VEHICLE MASS
# ============================================================

minimum_power = np.where(
    df["seats"] >= 7, 75,
    np.where(df["seats"] >= 6, 70, 60)
)

check_power_capacity = df["power_kw"] >= minimum_power

# ============================================================
# 6. POWERTRAIN-SPECIFIC LOGIC
# ============================================================

bev = df["powertrain"].eq("BEV")
cng = df["powertrain"].eq("CNG")
hybrid = df["powertrain"].eq("Hybrid")
petrol = df["powertrain"].eq("Petrol")

# ------------------------------------------------------------
# BEV battery / mass
# ------------------------------------------------------------

bev_min_mass = np.select(
    [
        df["battery_capacity_kwh"] <= 30,
        df["battery_capacity_kwh"] <= 45,
        df["battery_capacity_kwh"] <= 60,
        df["battery_capacity_kwh"] > 60
    ],
    [
        1050,
        1150,
        1250,
        1400
    ],
    default=1050
)

check_bev_battery_mass = (
    (~bev)
    |
    (df["kerb_weight_kg"] >= bev_min_mass)
)

# ------------------------------------------------------------
# BEV battery packaging
# ------------------------------------------------------------

battery_per_tonne = (
    df["battery_capacity_kwh"]
    / (df["kerb_weight_kg"] / 1000)
)

check_battery_packaging = (
    (~bev)
    |
    (battery_per_tonne <= 45)
)

# ------------------------------------------------------------
# BEV range / battery
# ------------------------------------------------------------

bev_range_efficiency = (
    df["certified_range_km"]
    / df["battery_capacity_kwh"]
)

check_bev_range = (
    (~bev)
    |
    (
        (bev_range_efficiency >= 4.5)
        &
        (bev_range_efficiency <= 10.0)
    )
)

# ============================================================
# 7. REALISTIC ENERGY CONSUMPTION BOUNDS
# ============================================================

# NOTE:
# These are broad engineering screening limits.
# ML prediction is evaluated separately later.

check_bev_energy = (
    (~bev)
    |
    (
        (df["battery_capacity_kwh"] > 0)
        &
        (df["battery_capacity_kwh"] <= 80)
    )
)

check_cng_energy = (
    (~cng)
    |
    df["powertrain"].eq("CNG")
)

check_petrol_energy = (
    (~petrol)
    |
    df["powertrain"].eq("Petrol")
)

check_hybrid_energy = (
    (~hybrid)
    |
    df["powertrain"].eq("Hybrid")
)

# ============================================================
# 8. PURCHASE PRICE REALISM
# ============================================================

# General 2035 modeling floor
check_base_price = df["purchase_price_inr"] >= 800000

# 7-seat commercial vehicles
check_7seat_price = (
    (df["seats"] < 7)
    |
    (df["purchase_price_inr"] >= 1000000)
)

# Large BEV battery
check_large_bev_price = (
    (~bev)
    |
    (df["battery_capacity_kwh"] <= 60)
    |
    (df["purchase_price_inr"] >= 1500000)
)

# High-power vehicle
check_high_power_price = (
    (df["power_kw"] <= 130)
    |
    (df["purchase_price_inr"] >= 1200000)
)

# ============================================================
# 9. REALISTIC POWER / MASS RELATIONSHIP
# ============================================================

# Higher vehicle mass requires appropriate power.
check_mass_power = (
    (df["kerb_weight_kg"] < 1600)
    |
    (df["power_kw"] >= 80)
)

check_heavy_power = (
    (df["kerb_weight_kg"] < 1800)
    |
    (df["power_kw"] >= 100)
)

# ============================================================
# 10. REALISTIC BEV PRICE-BATTERY RELATIONSHIP
# ============================================================

# Prevent extremely cheap large-battery BEVs.
bev_price_floor = (
    700000
    + 18000 * df["battery_capacity_kwh"]
)

check_bev_price_battery = (
    (~bev)
    |
    (df["purchase_price_inr"] >= bev_price_floor)
)

# Additional floor for very large batteries.
check_bev_large_battery_price = (
    (~bev)
    |
    (df["battery_capacity_kwh"] < 55)
    |
    (df["purchase_price_inr"] >= 1600000)
)

# ============================================================
# 11. REALISTIC VEHICLE PRICE / SIZE RELATIONSHIP
# ============================================================

size_price_floor = (
    650000
    + 350 * df["kerb_weight_kg"]
    + 800 * df["luggage_volume_l"]
)

# This is deliberately softened so it acts as a screening
# relationship rather than forcing luxury-level prices.

check_size_price = (
    df["purchase_price_inr"] >= 0.45 * size_price_floor
)

# ============================================================
# 12. COMBINE ALL ENGINEERING CHECKS
# ============================================================

checks = {
    "check_power_to_weight": check_power_weight,
    "check_seat_mass": check_seat_mass,
    "check_seat_luggage": check_seat_luggage,
    "check_wheelbase_luggage": check_wheelbase_luggage,
    "check_bev_battery_mass": check_bev_battery_mass,
    "check_bev_range_battery": check_bev_range,
    "check_bev_energy": check_bev_energy,
    "check_cng_energy": check_cng_energy,
    "check_petrol_energy": check_petrol_energy,
    "check_hybrid_energy": check_hybrid_energy,
    "check_mass_power": check_mass_power,
    "check_high_mass_power": check_heavy_power,
    "check_power_capacity": check_power_capacity,
    "check_battery_packaging": check_battery_packaging,
    "check_base_price": check_base_price,
    "check_7seat_price": check_7seat_price,
    "check_large_bev_price": check_large_bev_price,
    "check_high_power_price": check_high_power_price,
    "check_bev_price_battery": check_bev_price_battery,
    "check_bev_large_battery_price": check_bev_large_battery_price,
    "check_size_price": check_size_price
}

print()
print("=" * 70)
print("REALISM CONSTRAINT RESULTS")
print("=" * 70)

for name, result in checks.items():
    print(f"{name:35s}: {int((~result).sum()):5d} failed")

# Combine
feasible = np.ones(len(df), dtype=bool)

for result in checks.values():
    feasible &= result.to_numpy()

df["engineering_feasible_realistic"] = feasible

refined = df[df["engineering_feasible_realistic"]].copy()

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("REALISTIC REFINED DESIGN SPACE")
print("=" * 70)

print(f"Original designs : {len(df):,}")
print(f"Feasible designs : {len(refined):,}")
print(f"Rejected designs : {(~feasible).sum():,}")

print()
print("Powertrain distribution:")
print(refined["powertrain"].value_counts())

print()
print("Seat distribution:")
print(refined["seats"].value_counts().sort_index())

print()
print("-" * 70)
print("FEASIBLE DESIGN RANGES")
print("-" * 70)

for col in [
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
    "power_to_weight_kw_per_tonne"
]:
    print(
        f"{col:35s}: "
        f"{refined[col].min():,.2f} – {refined[col].max():,.2f}"
    )

# ============================================================
# BEV SUMMARY
# ============================================================

bev_df = refined[refined["powertrain"] == "BEV"]

print()
print("-" * 70)
print("FEASIBLE BEV DESIGN RANGE")
print("-" * 70)

if len(bev_df) > 0:
    print(
        f"BEV designs        : {len(bev_df):,}"
    )
    print(
        f"Battery capacity   : "
        f"{bev_df['battery_capacity_kwh'].min():.2f} – "
        f"{bev_df['battery_capacity_kwh'].max():.2f} kWh"
    )
    print(
        f"Certified range    : "
        f"{bev_df['certified_range_km'].min():.2f} – "
        f"{bev_df['certified_range_km'].max():.2f} km"
    )
    print(
        f"Purchase price     : "
        f"₹{bev_df['purchase_price_inr'].min():,.0f} – "
        f"₹{bev_df['purchase_price_inr'].max():,.0f}"
    )
else:
    print("WARNING: No BEV designs survived.")

# ============================================================
# SAVE
# ============================================================

refined.to_csv(OUTPUT, index=False)

print()
print("=" * 70)
print("REALISTIC REFINEMENT COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(f"  {OUTPUT}")

print()
print("Next step:")
print("Run the ML evaluation on this realistic design space.")