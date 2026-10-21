"""
INDIA 2035 RIDE-SERVICE VEHICLE
NSGA-II MULTI-OBJECTIVE OPTIMIZER

This script performs NSGA-II over the refined, engineering-feasible 2035
design space. It uses the trained ML surrogate models for energy and comfort,
then calculates fleet economics and emissions for every generated candidate.

Objectives:
    1. maximize annual profit
    2. maximize comfort/practicality
    3. minimize annual CO2
    4. minimize purchase price

The optimizer uses mixed-variable chromosomes:
    [powertrain, seats, wheelbase, luggage_volume, kerb_mass,
     power, purchase_price, battery_capacity, certified_range]

New offspring are repaired against the refined engineering design space
using nearest-neighbour matching. This keeps the optimization inside the
screened feasible region instead of allowing unrealistic crossover results.

Outputs:
    data/processed/nsga2_2035_pareto.csv
    data/processed/nsga2_2035_population.csv
    results/nsga2_2035_pareto.png
    results/nsga2_2035_objective_tradeoffs.png
    results/nsga2_2035_summary.csv
"""

from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

REFINED_FILE = DATA / "design_space_refined_2035.csv"
ML_FILE = DATA / "ml_training_dataset.csv"
ENERGY_MODEL_FILE = MODELS / "energy_consumption_model.pkl"
COMFORT_MODEL_FILE = MODELS / "comfort_index_model.pkl"

PARETO_FILE = DATA / "nsga2_2035_pareto.csv"
POP_FILE = DATA / "nsga2_2035_population.csv"
SUMMARY_FILE = RESULTS / "nsga2_2035_summary.csv"
PARETO_PLOT = RESULTS / "nsga2_2035_pareto.png"
TRADEOFF_PLOT = RESULTS / "nsga2_2035_objective_tradeoffs.png"

# ---------------------------------------------------------------------
# NSGA-II SETTINGS
# ---------------------------------------------------------------------
SEED = 2035
POP_SIZE = 120
GENERATIONS = 80
TOURNAMENT_SIZE = 2
CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.15
ELITE_COUNT = 10

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------
# 2035 CENTRAL COMMERCIAL DUTY CYCLE
# ---------------------------------------------------------------------
TRIPS_PER_DAY = 10.351807
PASSENGERS_PER_TRIP = 2.968
LUGGAGE_PER_TRIP_KG = 20.0
DAILY_DISTANCE_KM = 313.121343
ANNUAL_WORKING_DAYS = 330
OPERATING_MIN_PER_TRIP = 45.0

TRIP_DISTANCE_KM = DAILY_DISTANCE_KM / TRIPS_PER_DAY
DAILY_OPERATING_HOURS = TRIPS_PER_DAY * OPERATING_MIN_PER_TRIP / 60.0
ANNUAL_DISTANCE_KM = DAILY_DISTANCE_KM * ANNUAL_WORKING_DAYS
ANNUAL_PASSENGER_KM = (
    DAILY_DISTANCE_KM * PASSENGERS_PER_TRIP * ANNUAL_WORKING_DAYS
)

# Central 2035 assumptions used previously in the project
FARE_PER_KM = 30.0
DEADHEAD_FRACTION = 0.10

ENERGY_PRICE = {
    "BEV": 4.50,       # INR/kWh
    "CNG": 97.00,      # INR/kg
    "Hybrid": 111.68,  # INR/L petrol equivalent
    "Petrol": 111.68,  # INR/L
}

CO2_FACTOR = {
    "BEV": 0.70,       # kg CO2/kWh
    "CNG": 2.75,       # kg CO2/kg CNG
    "Hybrid": 2.31,    # kg CO2/L petrol
    "Petrol": 2.31,    # kg CO2/L petrol
}

MAINTENANCE_RATE = {
    "BEV": 0.025,
    "CNG": 0.035,
    "Hybrid": 0.035,
    "Petrol": 0.040,
}

INSURANCE_RATE = 0.025
DRIVER_COST_PER_DAY = 900.0
PLATFORM_COMMISSION = 0.20
DEPRECIATION_YEARS = 8
RESIDUAL_FRACTION = 0.20

# ---------------------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------------------
ENERGY_NUMERIC = [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "battery_capacity_kwh",
    "certified_range_km",
    "power_kw",
    "purchase_price_inr",
    "trip_distance_km",
    "trips_per_day_2035",
    "passengers_per_trip_2035",
    "luggage_kg_per_trip",
    "daily_distance_km_2035",
    "daily_operating_hours_2035",
]

ENERGY_CATEGORICAL = [
    "powertrain",
    "vehicle_type",
    "scenario_name",
]

COMFORT_NUMERIC = [
    "seats",
    "kerb_weight_kg",
    "wheelbase_mm",
    "luggage_volume_l",
    "power_kw",
    "purchase_price_inr",
    "trip_distance_km",
    "trips_per_day_2035",
    "passengers_per_trip_2035",
    "luggage_kg_per_trip",
    "daily_distance_km_2035",
    "daily_operating_hours_2035",
]

COMFORT_CATEGORICAL = [
    "powertrain",
    "vehicle_type",
    "scenario_name",
]

# ---------------------------------------------------------------------
# LOAD DATA / MODELS
# ---------------------------------------------------------------------
def load_inputs():
    if not REFINED_FILE.exists():
        raise FileNotFoundError(f"Missing: {REFINED_FILE}")
    if not ENERGY_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing: {ENERGY_MODEL_FILE}")
    if not COMFORT_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing: {COMFORT_MODEL_FILE}")

    df = pd.read_csv(REFINED_FILE)

    energy_model = joblib.load(ENERGY_MODEL_FILE)
    comfort_model = joblib.load(COMFORT_MODEL_FILE)

    return df, energy_model, comfort_model


# ---------------------------------------------------------------------
# ENGINEERING REPAIR
# ---------------------------------------------------------------------
def engineering_feasible(row):
    pt = row["powertrain"]
    seats = int(round(row["seats"]))
    mass = row["kerb_weight_kg"]
    wb = row["wheelbase_mm"]
    luggage = row["luggage_volume_l"]
    power = row["power_kw"]
    price = row["purchase_price_inr"]

    # Power-to-weight
    pwt = power / (mass / 1000.0)
    if not (40 <= pwt <= 120):
        return False

    # Seat-mass
    min_mass = {4: 1050, 5: 1100, 6: 1250, 7: 1350}
    if mass < min_mass.get(seats, 1350):
        return False

    # Seat-luggage
    min_luggage = {4: 300, 5: 320, 6: 380, 7: 400}
    if luggage < min_luggage.get(seats, 400):
        return False

    # Long wheelbase needs useful luggage volume
    if wb >= 2750 and luggage < 400:
        return False

    # Power for higher occupancy
    if seats == 6 and power < 70:
        return False
    if seats == 7 and power < 75:
        return False

    # Price constraints
    if price < 800000:
        return False
    if seats == 7 and price < 1000000:
        return False
    if power > 130 and price < 1200000:
        return False

    if pt == "BEV":
        batt = row["battery_capacity_kwh"]
        rng = row["certified_range_km"]

        if pd.isna(batt) or pd.isna(rng):
            return False

        if batt <= 30 and mass < 1050:
            return False
        if batt <= 45 and mass < 1150:
            return False
        if batt <= 60 and mass < 1250:
            return False
        if batt > 60 and mass < 1400:
            return False

        # Certified range / battery
        eff = rng / batt
        if not (4.5 <= eff <= 10.0):
            return False

        # Battery packaging
        if batt / (mass / 1000.0) > 45:
            return False

        # Energy surrogate physical screening
        if batt > 60 and price < 1500000:
            return False

    return True


def repair_candidate(candidate, feasible_reference):
    """
    Repair a child by snapping it to the closest already-screened feasible
    design. The nearest design is selected using normalized engineering
    variables, preserving a realistic combination of variables.
    """
    # Categorical/ordinal fields
    cols = [
        "seats",
        "wheelbase_mm",
        "luggage_volume_l",
        "kerb_weight_kg",
        "power_kw",
        "purchase_price_inr",
        "battery_capacity_kwh",
        "certified_range_km",
    ]

    ref = feasible_reference.copy()

    # Only compare fields that are meaningful for both candidate and row.
    # Missing BEV-specific fields are treated through a large penalty.
    scales = {
        "seats": 1.0,
        "wheelbase_mm": 250.0,
        "luggage_volume_l": 200.0,
        "kerb_weight_kg": 400.0,
        "power_kw": 50.0,
        "purchase_price_inr": 800000.0,
        "battery_capacity_kwh": 30.0,
        "certified_range_km": 250.0,
    }

    best_idx = None
    best_score = np.inf

    for idx, r in ref.iterrows():
        if r["powertrain"] != candidate["powertrain"]:
            continue

        score = 0.0
        for c in cols:
            a = candidate[c]
            b = r[c]

            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b):
                score += 10.0
                continue

            score += ((float(a) - float(b)) / scales[c]) ** 2

        # Seat is effectively discrete
        score += 2.0 * abs(int(round(candidate["seats"])) - int(round(r["seats"])))

        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        # Fall back to a random feasible design of the same powertrain
        same = ref[ref["powertrain"] == candidate["powertrain"]]
        if len(same) == 0:
            same = ref
        return same.sample(1, random_state=random.randint(0, 10**9)).iloc[0].copy()

    return ref.loc[best_idx].copy()


# ---------------------------------------------------------------------
# INITIAL POPULATION
# ---------------------------------------------------------------------
def random_candidate(ref):
    return ref.sample(1, random_state=random.randint(0, 10**9)).iloc[0].copy()


def make_initial_population(ref):
    population = []

    # Ensure every powertrain gets representation.
    for pt in ref["powertrain"].unique():
        subset = ref[ref["powertrain"] == pt]
        n = max(5, int(POP_SIZE * len(subset) / len(ref)))
        for _ in range(n):
            population.append(subset.sample(
                1, random_state=random.randint(0, 10**9)
            ).iloc[0].copy())

    while len(population) < POP_SIZE:
        population.append(random_candidate(ref))

    return population[:POP_SIZE]


# ---------------------------------------------------------------------
# CROSSOVER + MUTATION
# ---------------------------------------------------------------------
GENES = [
    "powertrain",
    "seats",
    "wheelbase_mm",
    "luggage_volume_l",
    "kerb_weight_kg",
    "power_kw",
    "purchase_price_inr",
    "battery_capacity_kwh",
    "certified_range_km",
]


def crossover(parent1, parent2, ref):
    c1 = parent1.copy()
    c2 = parent2.copy()

    if random.random() > CROSSOVER_RATE:
        return c1, c2

    # Keep powertrain consistent; swapping it can invalidate battery fields.
    if random.random() < 0.20:
        c1["powertrain"] = parent2["powertrain"]
        c2["powertrain"] = parent1["powertrain"]

    for g in GENES[1:]:
        if random.random() < 0.5:
            c1[g], c2[g] = parent2[g], parent1[g]

    c1["seats"] = int(round(c1["seats"]))
    c2["seats"] = int(round(c2["seats"]))

    c1 = repair_candidate(c1, ref)
    c2 = repair_candidate(c2, ref)

    return c1, c2


def mutate(candidate, ref):
    c = candidate.copy()

    if random.random() > MUTATION_RATE:
        return c

    # Mutation is deliberately moderate to keep the search in the
    # neighborhood of physically screened designs.
    mutation_scales = {
        "wheelbase_mm": 40.0,
        "luggage_volume_l": 60.0,
        "kerb_weight_kg": 100.0,
        "power_kw": 20.0,
        "purchase_price_inr": 180000.0,
        "battery_capacity_kwh": 8.0,
        "certified_range_km": 60.0,
    }

    for g, sigma in mutation_scales.items():
        if random.random() < 0.35:
            if pd.isna(c[g]):
                continue
            c[g] = float(c[g]) + np.random.normal(0, sigma)

    if random.random() < 0.10:
        c["seats"] = random.choice([4, 5, 6, 7])

    # Occasionally change powertrain to explore cross-powertrain tradeoffs.
    if random.random() < 0.05:
        c["powertrain"] = random.choice(list(ref["powertrain"].unique()))

    # Keep numerical values in the broad original design bounds.
    c["wheelbase_mm"] = np.clip(c["wheelbase_mm"], 2450, 2850)
    c["luggage_volume_l"] = np.clip(c["luggage_volume_l"], 300, 650)
    c["kerb_weight_kg"] = np.clip(c["kerb_weight_kg"], 1050, 2000)
    c["power_kw"] = np.clip(c["power_kw"], 60, 160)
    c["purchase_price_inr"] = np.clip(c["purchase_price_inr"], 800000, 2500000)

    if c["powertrain"] == "BEV":
        if pd.isna(c["battery_capacity_kwh"]):
            same = ref[ref["powertrain"] == "BEV"]
            r = same.sample(1, random_state=random.randint(0, 10**9)).iloc[0]
            c["battery_capacity_kwh"] = r["battery_capacity_kwh"]
            c["certified_range_km"] = r["certified_range_km"]
        c["battery_capacity_kwh"] = np.clip(c["battery_capacity_kwh"], 25, 80)
        c["certified_range_km"] = np.clip(c["certified_range_km"], 300, 700)
    else:
        c["battery_capacity_kwh"] = np.nan
        c["certified_range_km"] = np.nan

    c = repair_candidate(c, ref)

    return c


# ---------------------------------------------------------------------
# PERFORMANCE EVALUATION
# ---------------------------------------------------------------------
def make_model_frame(df):
    x = df.copy()
    x["trip_distance_km"] = TRIP_DISTANCE_KM
    x["trips_per_day_2035"] = TRIPS_PER_DAY
    x["passengers_per_trip_2035"] = PASSENGERS_PER_TRIP
    x["luggage_kg_per_trip"] = LUGGAGE_PER_TRIP_KG
    x["daily_distance_km_2035"] = DAILY_DISTANCE_KM
    x["daily_operating_hours_2035"] = DAILY_OPERATING_HOURS
    x["scenario_name"] = "mixed_commercial"
    x["vehicle_type"] = x.get("vehicle_type", "Passenger Vehicle")
    return x


def predict_energy_and_comfort(df, energy_model, comfort_model):
    x = make_model_frame(df)

    energy_features = ENERGY_NUMERIC + ENERGY_CATEGORICAL
    comfort_features = COMFORT_NUMERIC + COMFORT_CATEGORICAL

    e = energy_model.predict(x[energy_features])
    c = comfort_model.predict(x[comfort_features])

    return np.asarray(e, dtype=float), np.asarray(c, dtype=float)


def calculate_performance(df, energy_model, comfort_model):
    out = df.copy().reset_index(drop=True)

    energy_q, comfort = predict_energy_and_comfort(
        out, energy_model, comfort_model
    )

    out["predicted_energy_quantity_per_km"] = energy_q
    out["predicted_comfort_index"] = comfort

    # Energy cost
    out["energy_price_inr_per_unit"] = out["powertrain"].map(ENERGY_PRICE)
    out["energy_cost_inr_per_km"] = (
        out["predicted_energy_quantity_per_km"]
        * out["energy_price_inr_per_unit"]
    )

    # Annual commercial distance including deadheading
    out["deadhead_distance_km"] = ANNUAL_DISTANCE_KM * DEADHEAD_FRACTION
    out["total_vehicle_distance_km"] = (
        ANNUAL_DISTANCE_KM + out["deadhead_distance_km"]
    )

    # Revenue is passenger-trip based. We use the central flat fare
    # assumption used in the design-space evaluation.
    annual_revenue = (
        ANNUAL_DISTANCE_KM * FARE_PER_KM
    )
    out["annual_revenue_inr"] = annual_revenue

    # Annual costs
    out["annual_energy_cost_inr"] = (
        out["total_vehicle_distance_km"]
        * out["energy_cost_inr_per_km"]
    )

    out["annual_maintenance_cost_inr"] = (
        out["purchase_price_inr"]
        * out["powertrain"].map(MAINTENANCE_RATE)
    )

    out["annual_insurance_cost_inr"] = (
        out["purchase_price_inr"] * INSURANCE_RATE
    )

    out["annual_depreciation_cost_inr"] = (
        out["purchase_price_inr"]
        * (1 - RESIDUAL_FRACTION)
        / DEPRECIATION_YEARS
    )

    out["annual_driver_cost_inr"] = (
        DRIVER_COST_PER_DAY * ANNUAL_WORKING_DAYS
    )

    out["annual_platform_cost_inr"] = (
        out["annual_revenue_inr"] * PLATFORM_COMMISSION
    )

    out["annual_operating_cost_inr"] = (
        out["annual_energy_cost_inr"]
        + out["annual_maintenance_cost_inr"]
        + out["annual_insurance_cost_inr"]
        + out["annual_depreciation_cost_inr"]
        + out["annual_driver_cost_inr"]
        + out["annual_platform_cost_inr"]
    )

    out["annual_profit_inr"] = (
        out["annual_revenue_inr"]
        - out["annual_operating_cost_inr"]
    )

    out["profit_margin_percent"] = (
        100 * out["annual_profit_inr"] / out["annual_revenue_inr"]
    )

    # CO2
    out["annual_co2_kg"] = (
        out["total_vehicle_distance_km"]
        * out["predicted_energy_quantity_per_km"]
        * out["powertrain"].map(CO2_FACTOR)
    )
    out["annual_co2_tonnes"] = out["annual_co2_kg"] / 1000.0

    # Range / daily feasibility
    out["usable_range_km"] = np.nan
    bev = out["powertrain"] == "BEV"
    out.loc[bev, "usable_range_km"] = (
        out.loc[bev, "certified_range_km"] * 0.80
    )
    out.loc[~bev, "usable_range_km"] = 999999

    out["range_margin_km"] = (
        out["usable_range_km"] - DAILY_DISTANCE_KM
    )
    out["range_feasible"] = (
        out["range_margin_km"] >= 0
    )

    # Final feasibility
    out["engineering_feasible"] = out.apply(engineering_feasible, axis=1)

    return out


# ---------------------------------------------------------------------
# PARETO / NSGA-II
# ---------------------------------------------------------------------
# All objectives are represented internally as "maximize":
#   profit
#   comfort
#   -CO2
#   -price
OBJECTIVE_COLUMNS = [
    "annual_profit_inr",
    "predicted_comfort_index",
    "annual_co2_tonnes",
    "purchase_price_inr",
]


def objective_matrix(df):
    return np.column_stack([
        df["annual_profit_inr"].values,
        df["predicted_comfort_index"].values,
        -df["annual_co2_tonnes"].values,
        -df["purchase_price_inr"].values,
    ])


def dominates(a, b):
    return np.all(a >= b) and np.any(a > b)


def fast_non_dominated_sort(F):
    n = len(F)
    domination_count = np.zeros(n, dtype=int)
    dominated = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                dominated[p].append(q)
            elif dominates(F[q], F[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def crowding_distance(F, front):
    if len(front) <= 2:
        return {i: np.inf for i in front}

    distances = {i: 0.0 for i in front}
    front_arr = np.array(front)

    for j in range(F.shape[1]):
        order = front_arr[np.argsort(F[front_arr, j])]
        fmin = F[order[0], j]
        fmax = F[order[-1], j]

        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        if fmax == fmin:
            continue

        for k in range(1, len(order) - 1):
            if np.isinf(distances[order[k]]):
                continue
            distances[order[k]] += (
                F[order[k + 1], j] - F[order[k - 1], j]
            ) / (fmax - fmin)

    return distances


def rank_and_crowding(pop_df):
    F = objective_matrix(pop_df)
    fronts = fast_non_dominated_sort(F)

    rank = np.full(len(pop_df), 999, dtype=int)
    crowd = np.zeros(len(pop_df), dtype=float)

    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
        cd = crowding_distance(F, front)
        for i, value in cd.items():
            crowd[i] = value

    return fronts, rank, crowd


def tournament(pop, rank, crowd):
    a = random.randrange(len(pop))
    b = random.randrange(len(pop))

    if rank[a] < rank[b]:
        return pop[a].copy()
    if rank[b] < rank[a]:
        return pop[b].copy()

    if crowd[a] > crowd[b]:
        return pop[a].copy()
    if crowd[b] > crowd[a]:
        return pop[b].copy()

    return pop[random.choice([a, b])].copy()


def select_next_generation(combined_df):
    F = objective_matrix(combined_df)
    fronts = fast_non_dominated_sort(F)

    selected = []

    for front in fronts:
        if len(selected) + len(front) <= POP_SIZE:
            selected.extend(front)
        else:
            remaining = POP_SIZE - len(selected)
            cd = crowding_distance(F, front)
            ordered = sorted(front, key=lambda i: cd[i], reverse=True)
            selected.extend(ordered[:remaining])
            break

    return combined_df.iloc[selected].copy().reset_index(drop=True)


# ---------------------------------------------------------------------
# MAIN OPTIMIZATION
# ---------------------------------------------------------------------
def run_optimizer():
    print("=" * 70)
    print("INDIA 2035 NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 70)

    ref, energy_model, comfort_model = load_inputs()

    print(f"\nRefined design space: {len(ref):,} designs")
    print(f"Population size      : {POP_SIZE}")
    print(f"Generations           : {GENERATIONS}")
    print(f"Random seed           : {SEED}")

    # The refined file already contains engineering-screened designs.
    # Keep only rows that satisfy the final engineering check as a safety
    # guard before optimization.
    feasible_ref = ref[ref.apply(engineering_feasible, axis=1)].copy()

    print(f"Final feasible pool  : {len(feasible_ref):,}")

    population = make_initial_population(feasible_ref)
    population_df = pd.DataFrame(population)
    population_df = calculate_performance(
        population_df, energy_model, comfort_model
    )

    # Remove any accidental range-infeasible candidate
    population_df = population_df[
        population_df["range_feasible"]
    ].copy().reset_index(drop=True)

    # Refill if needed
    while len(population_df) < POP_SIZE:
        extra = pd.DataFrame(
            [random_candidate(feasible_ref)]
        )
        extra = calculate_performance(
            extra, energy_model, comfort_model
        )
        if bool(extra["range_feasible"].iloc[0]):
            population_df = pd.concat(
                [population_df, extra], ignore_index=True
            )

    history = []

    for generation in range(1, GENERATIONS + 1):
        _, rank, crowd = rank_and_crowding(population_df)

        offspring = []

        while len(offspring) < POP_SIZE:
            p1 = tournament(
                population_df.to_dict("records"), rank, crowd
            )
            p2 = tournament(
                population_df.to_dict("records"), rank, crowd
            )

            p1 = pd.Series(p1)
            p2 = pd.Series(p2)

            c1, c2 = crossover(p1, p2, feasible_ref)

            c1 = mutate(c1, feasible_ref)
            c2 = mutate(c2, feasible_ref)

            offspring.extend([c1, c2])

        offspring = pd.DataFrame(offspring[:POP_SIZE])

        offspring = calculate_performance(
            offspring, energy_model, comfort_model
        )

        offspring = offspring[
            offspring["engineering_feasible"]
            & offspring["range_feasible"]
        ].copy()

        # If repair caused too many invalid rows, refill from feasible pool.
        while len(offspring) < POP_SIZE:
            extra = pd.DataFrame(
                [random_candidate(feasible_ref)]
            )
            extra = calculate_performance(
                extra, energy_model, comfort_model
            )
            if bool(extra["engineering_feasible"].iloc[0]) and bool(
                extra["range_feasible"].iloc[0]
            ):
                offspring = pd.concat(
                    [offspring, extra], ignore_index=True
                )

        combined = pd.concat(
            [population_df, offspring],
            ignore_index=True
        )

        population_df = select_next_generation(combined)

        fronts, rank, crowd = rank_and_crowding(population_df)
        pareto_count = len(fronts[0])

        front_df = population_df.iloc[fronts[0]]

        history.append({
            "generation": generation,
            "pareto_count": pareto_count,
            "max_profit_inr": front_df["annual_profit_inr"].max(),
            "max_comfort": front_df["predicted_comfort_index"].max(),
            "min_co2_tonnes": front_df["annual_co2_tonnes"].min(),
            "min_price_inr": front_df["purchase_price_inr"].min(),
        })

        if generation == 1 or generation % 10 == 0 or generation == GENERATIONS:
            print(
                f"Generation {generation:3d}/{GENERATIONS} | "
                f"Pareto designs: {pareto_count:3d} | "
                f"Profit max: ₹{front_df['annual_profit_inr'].max():,.0f} | "
                f"Comfort max: {front_df['predicted_comfort_index'].max():.2f} | "
                f"CO2 min: {front_df['annual_co2_tonnes'].min():.2f} t"
            )

    # Final Pareto front
    fronts, _, _ = rank_and_crowding(population_df)
    pareto = population_df.iloc[fronts[0]].copy().reset_index(drop=True)

    # Sort for readability
    pareto = pareto.sort_values(
        ["annual_profit_inr", "predicted_comfort_index"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Assign a simple representative label based on objective structure.
    # These are descriptive categories, not rankings.
    profit_norm = (
        (pareto["annual_profit_inr"] - population_df["annual_profit_inr"].min())
        / (
            population_df["annual_profit_inr"].max()
            - population_df["annual_profit_inr"].min()
            + 1e-12
        )
    )
    comfort_norm = (
        (pareto["predicted_comfort_index"] - population_df["predicted_comfort_index"].min())
        / (
            population_df["predicted_comfort_index"].max()
            - population_df["predicted_comfort_index"].min()
            + 1e-12
        )
    )
    co2_norm = (
        (population_df["annual_co2_tonnes"].max() - pareto["annual_co2_tonnes"])
        / (
            population_df["annual_co2_tonnes"].max()
            - population_df["annual_co2_tonnes"].min()
            + 1e-12
        )
    )

    composite = profit_norm + comfort_norm + co2_norm
    pareto["balanced_tradeoff_indicator"] = composite / 3.0

    # Save
    pareto.to_csv(PARETO_FILE, index=False)
    population_df.to_csv(POP_FILE, index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(SUMMARY_FILE, index=False)

    # -----------------------------------------------------------------
    # PRINT FINAL RESULTS
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("NSGA-II COMPLETE")
    print("=" * 70)

    print(f"\nFinal population : {len(population_df):,}")
    print(f"Pareto designs   : {len(pareto):,}")

    print("\nPareto powertrain distribution:")
    print(pareto["powertrain"].value_counts())

    print("\nPareto objective ranges:")
    print(
        f"Annual profit : ₹{pareto['annual_profit_inr'].min():,.0f}"
        f" – ₹{pareto['annual_profit_inr'].max():,.0f}"
    )
    print(
        f"Comfort index : {pareto['predicted_comfort_index'].min():.2f}"
        f" – {pareto['predicted_comfort_index'].max():.2f}"
    )
    print(
        f"Annual CO2    : {pareto['annual_co2_tonnes'].min():.2f}"
        f" – {pareto['annual_co2_tonnes'].max():.2f} tonnes"
    )
    print(
        f"Purchase price: ₹{pareto['purchase_price_inr'].min():,.0f}"
        f" – ₹{pareto['purchase_price_inr'].max():,.0f}"
    )

    print("\nRepresentative Pareto designs:")
    show_cols = [
        "powertrain",
        "seats",
        "kerb_weight_kg",
        "wheelbase_mm",
        "luggage_volume_l",
        "power_kw",
        "purchase_price_inr",
        "battery_capacity_kwh",
        "certified_range_km",
        "predicted_comfort_index",
        "annual_profit_inr",
        "annual_co2_tonnes",
    ]

    # Print up to 12 spread-out Pareto solutions.
    n_show = min(12, len(pareto))
    if n_show > 0:
        indices = np.linspace(0, len(pareto) - 1, n_show).astype(int)
        print(
            pareto.iloc[indices][show_cols].to_string(index=False)
        )

    # -----------------------------------------------------------------
    # PLOT 1: PROFIT VS CO2, comfort as size
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.scatter(
        pareto["annual_co2_tonnes"],
        pareto["annual_profit_inr"] / 1e6,
        s=20 + 3 * pareto["predicted_comfort_index"],
        alpha=0.75,
    )
    plt.xlabel("Annual CO2 (tonnes)")
    plt.ylabel("Annual Profit (₹ million)")
    plt.title("2035 Pareto Frontier: Profit vs Annual CO2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PARETO_PLOT, dpi=200)
    plt.close()

    # -----------------------------------------------------------------
    # PLOT 2: COMFORT VS PROFIT
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.scatter(
        pareto["predicted_comfort_index"],
        pareto["annual_profit_inr"] / 1e6,
        s=30,
        alpha=0.75,
    )
    plt.xlabel("Predicted Comfort & Practicality Index")
    plt.ylabel("Annual Profit (₹ million)")
    plt.title("2035 Pareto Frontier: Comfort vs Profit")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(TRADEOFF_PLOT, dpi=200)
    plt.close()

    print("\nSaved:")
    print(f"  {PARETO_FILE}")
    print(f"  {POP_FILE}")
    print(f"  {SUMMARY_FILE}")
    print(f"  {PARETO_PLOT}")
    print(f"  {TRADEOFF_PLOT}")


if __name__ == "__main__":
    run_optimizer()
