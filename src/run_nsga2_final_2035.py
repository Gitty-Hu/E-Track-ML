import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "design_space_realistic_evaluated_2035.csv"
)

PARETO_OUTPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "nsga2_final_2035_pareto.csv"
)

POP_OUTPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "nsga2_final_2035_population.csv"
)

SUMMARY_OUTPUT = (
    BASE_DIR
    / "results"
    / "nsga2_final_2035_summary.csv"
)

PARETO_PLOT = (
    BASE_DIR
    / "results"
    / "nsga2_final_2035_pareto.png"
)

TRADEOFF_PLOT = (
    BASE_DIR
    / "results"
    / "nsga2_final_2035_tradeoffs.png"
)

POP_SIZE = 120
GENERATIONS = 80
RANDOM_SEED = 2035

np.random.seed(RANDOM_SEED)

print("=" * 70)
print("INDIA 2035 FINAL NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print(f"Realistic evaluated designs: {len(df):,}")

# ============================================================
# HARD FEASIBILITY FILTER
# ============================================================

# Only retain designs satisfying the final engineering
# and operating feasibility requirements.

required_columns = [
    "engineering_feasible_realistic",
    "range_feasible",
    "luggage_feasible",
    "annual_profit_inr",
    "predicted_comfort_index",
    "annual_co2_tonnes",
    "purchase_price_inr"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns: " + ", ".join(missing)
    )

df["final_feasible"] = (
    df["engineering_feasible_realistic"].astype(bool)
    &
    df["range_feasible"].astype(bool)
    &
    df["luggage_feasible"].astype(bool)
)

feasible = df[df["final_feasible"]].copy()

print(f"Final feasible pool: {len(feasible):,}")

if len(feasible) < POP_SIZE:
    raise ValueError(
        "Fewer feasible designs than population size."
    )

# ============================================================
# NORMALIZED OBJECTIVES
# ============================================================

# NSGA-II objective convention:
# lower = better
#
# 1. maximize profit
# 2. maximize comfort
# 3. minimize CO2
# 4. minimize purchase price

profit = feasible["annual_profit_inr"].to_numpy()
comfort = feasible["predicted_comfort_index"].to_numpy()
co2 = feasible["annual_co2_tonnes"].to_numpy()
price = feasible["purchase_price_inr"].to_numpy()

# Normalize each objective to 0–1.
def normalize(x):
    xmin = np.min(x)
    xmax = np.max(x)

    if xmax == xmin:
        return np.zeros_like(x)

    return (x - xmin) / (xmax - xmin)


profit_n = normalize(profit)
comfort_n = normalize(comfort)
co2_n = normalize(co2)
price_n = normalize(price)

# Objective matrix:
#
# f1 = 1 - normalized profit
# f2 = 1 - normalized comfort
# f3 = normalized CO2
# f4 = normalized purchase price

objectives = np.column_stack([
    1 - profit_n,
    1 - comfort_n,
    co2_n,
    price_n
])

# ============================================================
# PARETO DOMINANCE
# ============================================================

def dominates(a, b):
    """
    a dominates b if:
    - a is no worse in every objective
    - a is strictly better in at least one
    """

    return (
        np.all(a <= b)
        and
        np.any(a < b)
    )


def get_pareto_indices(obj):
    n = len(obj)

    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            if dominates(obj[j], obj[i]):
                dominated[i] = True
                break

    return np.where(~dominated)[0]


# ============================================================
# CROWDING DISTANCE
# ============================================================

def crowding_distance(obj, indices):

    if len(indices) == 0:
        return np.array([])

    if len(indices) <= 2:
        return np.full(len(indices), np.inf)

    distances = np.zeros(len(indices))

    front = obj[indices]

    n_objectives = front.shape[1]

    for m in range(n_objectives):

        order = np.argsort(front[:, m])

        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        min_value = front[order[0], m]
        max_value = front[order[-1], m]

        if max_value == min_value:
            continue

        for k in range(1, len(indices) - 1):

            distances[order[k]] += (
                front[order[k + 1], m]
                -
                front[order[k - 1], m]
            ) / (max_value - min_value)

    return distances


# ============================================================
# INITIAL POPULATION
# ============================================================

n_designs = len(feasible)

initial_indices = np.random.choice(
    n_designs,
    size=POP_SIZE,
    replace=False
)

population = initial_indices.copy()

# ============================================================
# NSGA-II LOOP
# ============================================================

print()
print(f"Population size      : {POP_SIZE}")
print(f"Generations           : {GENERATIONS}")
print(f"Random seed           : {RANDOM_SEED}")
print()

for generation in range(1, GENERATIONS + 1):

    # --------------------------------------------------------
    # Current population
    # --------------------------------------------------------

    current_obj = objectives[population]

    # --------------------------------------------------------
    # Generate offspring
    #
    # Since the design space is a finite evaluated candidate
    # set, offspring are selected from existing feasible designs.
    # This prevents extrapolation beyond the ML training domain.
    # --------------------------------------------------------

    offspring = []

    while len(offspring) < POP_SIZE:

        candidates = np.random.choice(
            population,
            size=2,
            replace=False
        )

        p1 = candidates[0]
        p2 = candidates[1]

        # Randomly select one of the parents.
        if np.random.rand() < 0.5:
            child = p1
        else:
            child = p2

        # Mutation: occasionally replace with a random
        # feasible design.
        if np.random.rand() < 0.20:

            child = np.random.randint(
                0,
                n_designs
            )

        offspring.append(child)

    offspring = np.array(offspring)

    # --------------------------------------------------------
    # Combine parents + offspring
    # --------------------------------------------------------

    combined = np.concatenate([
        population,
        offspring
    ])

    combined_obj = objectives[combined]

    # Remove duplicate candidate IDs
    unique_ids, unique_positions = np.unique(
        combined,
        return_index=True
    )

    combined = unique_ids
    combined_obj = objectives[combined]

    # --------------------------------------------------------
    # Select next generation
    # --------------------------------------------------------

    selected = []

    remaining = np.arange(len(combined))

    while len(selected) < POP_SIZE and len(remaining) > 0:

        remaining_obj = combined_obj[remaining]

        front_local = get_pareto_indices(
            remaining_obj
        )

        front_global = remaining[front_local]

        slots = POP_SIZE - len(selected)

        if len(front_global) <= slots:

            selected.extend(
                front_global.tolist()
            )

            remaining = np.array([
                x for x in remaining
                if x not in set(front_global)
            ])

        else:

            distances = crowding_distance(
                combined_obj,
                front_global
            )

            order = np.argsort(
                -distances
            )

            selected.extend(
                front_global[order[:slots]].tolist()
            )

            break

    population = combined[np.array(selected)]

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    pop_obj = objectives[population]

    pareto_local = get_pareto_indices(pop_obj)

    pareto_ids = population[pareto_local]

    p = feasible.iloc[pareto_ids]

    print(
        f"Generation {generation:3d}/{GENERATIONS} | "
        f"Pareto designs: {len(p):3d} | "
        f"Profit max: ₹{p['annual_profit_inr'].max():,.0f} | "
        f"Comfort max: {p['predicted_comfort_index'].max():.2f} | "
        f"CO2 min: {p['annual_co2_tonnes'].min():.2f} t"
    )

# ============================================================
# FINAL PARETO FRONT
# ============================================================

final_obj = objectives[population]

pareto_local = get_pareto_indices(
    final_obj
)

pareto_ids = population[pareto_local]

pareto = feasible.iloc[pareto_ids].copy()

# Sort by profit
pareto = pareto.sort_values(
    "annual_profit_inr",
    ascending=False
)

# ============================================================
# REPRESENTATIVE DESIGNS
# ============================================================

representatives = []

# Highest profit
representatives.append(
    pareto.loc[
        pareto["annual_profit_inr"].idxmax()
    ]
)

# Highest comfort
representatives.append(
    pareto.loc[
        pareto["predicted_comfort_index"].idxmax()
    ]
)

# Lowest CO2
representatives.append(
    pareto.loc[
        pareto["annual_co2_tonnes"].idxmin()
    ]
)

# Lowest price
representatives.append(
    pareto.loc[
        pareto["purchase_price_inr"].idxmin()
    ]
)

# Balanced design using normalized objectives
p_profit = normalize(
    pareto["annual_profit_inr"].to_numpy()
)

p_comfort = normalize(
    pareto["predicted_comfort_index"].to_numpy()
)

p_co2 = normalize(
    pareto["annual_co2_tonnes"].to_numpy()
)

p_price = normalize(
    pareto["purchase_price_inr"].to_numpy()
)

# Equal-weight compromise distance
distance = np.sqrt(
    (1 - p_profit) ** 2
    + (1 - p_comfort) ** 2
    + p_co2 ** 2
    + p_price ** 2
)

balanced_index = np.argmin(distance)

representatives.append(
    pareto.iloc[balanced_index]
)

representative_df = pd.DataFrame(
    representatives
).drop_duplicates(
    subset=["vehicle_id"] if "vehicle_id" in pareto.columns
    else None
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL NSGA-II RESULTS")
print("=" * 70)

print()
print(f"Final population : {len(population):,}")
print(f"Pareto designs   : {len(pareto):,}")

print()
print("Pareto powertrain distribution:")
print(
    pareto["powertrain"]
    .value_counts()
)

print()
print("-" * 70)
print("PARETO OBJECTIVE RANGES")
print("-" * 70)

print(
    f"Annual profit : "
    f"₹{pareto['annual_profit_inr'].min():,.0f}"
    f" – ₹{pareto['annual_profit_inr'].max():,.0f}"
)

print(
    f"Comfort index : "
    f"{pareto['predicted_comfort_index'].min():.2f}"
    f" – {pareto['predicted_comfort_index'].max():.2f}"
)

print(
    f"Annual CO2    : "
    f"{pareto['annual_co2_tonnes'].min():.2f}"
    f" – {pareto['annual_co2_tonnes'].max():.2f} tonnes"
)

print(
    f"Purchase price: "
    f"₹{pareto['purchase_price_inr'].min():,.0f}"
    f" – ₹{pareto['purchase_price_inr'].max():,.0f}"
)

# ============================================================
# REPRESENTATIVE DESIGN TABLE
# ============================================================

print()
print("-" * 70)
print("REPRESENTATIVE PARETO DESIGNS")
print("-" * 70)

display_columns = [
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
    "annual_co2_tonnes"
]

display_columns = [
    c for c in display_columns
    if c in representative_df.columns
]

print(
    representative_df[
        display_columns
    ].to_string(
        index=False
    )
)

# ============================================================
# SAVE CSVs
# ============================================================

pareto.to_csv(
    PARETO_OUTPUT,
    index=False
)

feasible.iloc[population].to_csv(
    POP_OUTPUT,
    index=False
)

summary = pd.DataFrame({
    "metric": [
        "feasible_designs",
        "final_population",
        "pareto_designs",
        "profit_min_inr",
        "profit_max_inr",
        "comfort_min",
        "comfort_max",
        "co2_min_tonnes",
        "co2_max_tonnes",
        "price_min_inr",
        "price_max_inr"
    ],
    "value": [
        len(feasible),
        len(population),
        len(pareto),
        pareto["annual_profit_inr"].min(),
        pareto["annual_profit_inr"].max(),
        pareto["predicted_comfort_index"].min(),
        pareto["predicted_comfort_index"].max(),
        pareto["annual_co2_tonnes"].min(),
        pareto["annual_co2_tonnes"].max(),
        pareto["purchase_price_inr"].min(),
        pareto["purchase_price_inr"].max()
    ]
})

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

# ============================================================
# PLOT 1 — PROFIT VS CO2
# ============================================================

plt.figure(figsize=(9, 6))

for powertrain, group in pareto.groupby("powertrain"):

    plt.scatter(
        group["annual_co2_tonnes"],
        group["annual_profit_inr"] / 1e6,
        label=powertrain,
        alpha=0.75
    )

plt.xlabel("Annual CO₂ (tonnes)")
plt.ylabel("Annual Profit (₹ million)")
plt.title("India 2035 Pareto Designs: Profit vs CO₂")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    PARETO_PLOT,
    dpi=300
)
plt.close()

# ============================================================
# PLOT 2 — PROFIT VS COMFORT
# ============================================================

plt.figure(figsize=(9, 6))

for powertrain, group in pareto.groupby("powertrain"):

    plt.scatter(
        group["predicted_comfort_index"],
        group["annual_profit_inr"] / 1e6,
        label=powertrain,
        alpha=0.75
    )

plt.xlabel("Predicted Comfort & Practicality Index")
plt.ylabel("Annual Profit (₹ million)")
plt.title("India 2035 Pareto Designs: Profit vs Comfort")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    TRADEOFF_PLOT,
    dpi=300
)
plt.close()

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("FINAL NSGA-II OPTIMIZATION COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(f"  {PARETO_OUTPUT}")
print(f"  {POP_OUTPUT}")
print(f"  {SUMMARY_OUTPUT}")
print(f"  {PARETO_PLOT}")
print(f"  {TRADEOFF_PLOT}")