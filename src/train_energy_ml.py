import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# ENERGY CONSUMPTION ML MODEL
# ============================================================

INPUT_FILE = Path(
    "data/processed/ml_training_dataset.csv"
)

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70)
print("ENERGY CONSUMPTION ML SURROGATE")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset rows: {len(df)}")
print(f"Unique vehicles: {df['vehicle_id'].nunique()}")


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

TARGET = "energy_quantity_per_km"

y = df[TARGET].copy()


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

numeric_features = [
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

categorical_features = [
    "powertrain",
    "vehicle_type",
    "scenario_name",
]


# ------------------------------------------------------------
# KEEP ONLY EXISTING COLUMNS
# ------------------------------------------------------------

numeric_features = [
    c for c in numeric_features
    if c in df.columns
]

categorical_features = [
    c for c in categorical_features
    if c in df.columns
]


X = df[
    numeric_features +
    categorical_features
].copy()


print("\nFeatures:")
print("\nNumeric:")

for c in numeric_features:
    print(f"  - {c}")

print("\nCategorical:")

for c in categorical_features:
    print(f"  - {c}")


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)


xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)


# ------------------------------------------------------------
# GROUPED CROSS VALIDATION
# ------------------------------------------------------------

groups = df["vehicle_id"]

gkf = GroupKFold(
    n_splits=6
)


def evaluate_model(name, pipeline):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    predictions = np.zeros(len(df))

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline.fit(
            X_train,
            y_train
        )

        pred = pipeline.predict(
            X_test
        )

        predictions[test_idx] = pred

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        r2 = r2_score(
            y_test,
            pred
        )

        fold_results.append(
            [fold, mae, rmse, r2]
        )

        print(
            f"Fold {fold}: "
            f"MAE={mae:.5f} | "
            f"RMSE={rmse:.5f} | "
            f"R²={r2:.4f}"
        )

    results = pd.DataFrame(
        fold_results,
        columns=[
            "fold",
            "MAE",
            "RMSE",
            "R2"
        ]
    )

    print("\nOverall grouped CV:")
    print(
        f"MAE  = {mean_absolute_error(y, predictions):.5f}"
    )

    print(
        f"RMSE = {np.sqrt(mean_squared_error(y, predictions)):.5f}"
    )

    print(
        f"R²   = {r2_score(y, predictions):.4f}"
    )

    return results, predictions


# ------------------------------------------------------------
# TRAIN + EVALUATE
# ------------------------------------------------------------

rf_results, rf_predictions = evaluate_model(
    "RANDOM FOREST",
    rf_pipeline
)


xgb_results, xgb_predictions = evaluate_model(
    "XGBOOST",
    xgb_pipeline
)


# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------

rf_mae = mean_absolute_error(
    y,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y,
    rf_predictions
)


xgb_mae = mean_absolute_error(
    y,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y,
        xgb_predictions
    )
)

xgb_r2 = r2_score(
    y,
    xgb_predictions
)


comparison = pd.DataFrame({

    "model": [
        "Random Forest",
        "XGBoost"
    ],

    "MAE": [
        rf_mae,
        xgb_mae
    ],

    "RMSE": [
        rf_rmse,
        xgb_rmse
    ],

    "R2": [
        rf_r2,
        xgb_r2
    ]
})


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SELECT MODEL
# ------------------------------------------------------------

if xgb_mae <= rf_mae:

    best_name = "XGBoost"
    best_pipeline = xgb_pipeline

else:

    best_name = "Random Forest"
    best_pipeline = rf_pipeline


print(
    f"\nSelected model: {best_name}"
)


# ------------------------------------------------------------
# TRAIN FINAL MODEL ON ALL DATA
# ------------------------------------------------------------

print(
    "\nTraining final model on all available vehicles..."
)

best_pipeline.fit(
    X,
    y
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

import joblib

model_path = (
    MODEL_DIR /
    "energy_consumption_model.pkl"
)

joblib.dump(
    best_pipeline,
    model_path
)


# ------------------------------------------------------------
# SAVE COMPARISON
# ------------------------------------------------------------

comparison_path = (
    Path("results") /
    "energy_ml_model_comparison.csv"
)

comparison_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ------------------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------------------

prediction_df = df[
    [
        "vehicle_id",
        "powertrain",
        "model",
        TARGET
    ]
].copy()

prediction_df[
    "predicted_energy_quantity_per_km"
] = np.where(
    xgb_mae <= rf_mae,
    xgb_predictions,
    rf_predictions
)

prediction_df[
    "absolute_error"
] = abs(
    prediction_df[TARGET]
    -
    prediction_df[
        "predicted_energy_quantity_per_km"
    ]
)


prediction_path = (
    Path("results") /
    "energy_ml_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENERGY ML MODEL COMPLETE")
print("=" * 70)

print(
    f"\nModel saved:"
    f"\n  {model_path}"
)

print(
    f"\nComparison saved:"
    f"\n  {comparison_path}"
)

print(
    f"\nPredictions saved:"
    f"\n  {prediction_path}"
)

print("\nIMPORTANT:")
print(
    "Validation was grouped by vehicle_id so entire vehicles "
    "were held out during testing."
)

print("=" * 70)