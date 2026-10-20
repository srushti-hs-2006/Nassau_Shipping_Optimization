# ============================================================
# NASSAU CANDY DISTRIBUTOR
# MACHINE LEARNING MODEL COMPARISON
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

FILE_PATH = "Nassau Candy Distributor.xlsx"

print("\n" + "=" * 60)
print("NASSAU CANDY DISTRIBUTOR - MACHINE LEARNING")
print("=" * 60)

try:
    df = pd.read_excel(FILE_PATH)
    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

except FileNotFoundError:
    print(f"\nERROR: Could not find '{FILE_PATH}'.")
    print("Keep ml_models.py in the same folder as the Excel file.")
    raise SystemExit(1)


# ------------------------------------------------------------
# 2. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "Order ID",
    "Ship Mode",
    "Region",
    "Division",
    "Product ID",
    "Product Name",
    "Units",
    "Sales",
    "Cost",
    "Gross Profit"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Required columns are missing:")
    print(missing_columns)
    raise SystemExit(1)


# ------------------------------------------------------------
# 3. DATA PREPARATION
# ------------------------------------------------------------

df = df.drop_duplicates().copy()

for column in ["Units", "Sales", "Cost", "Gross Profit"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(
    subset=["Units", "Sales", "Cost", "Gross Profit"]
)

df = df[df["Units"] > 0].copy()

# The target is Gross Profit.
# Do not use Cost or Sales as predictors because:
# Gross Profit = Sales - Cost.
# Using them would cause target leakage.

target_column = "Gross Profit"

feature_columns = [
    "Ship Mode",
    "Region",
    "Division",
    "Product ID",
    "Product Name",
    "Units"
]

X = df[feature_columns].copy()
y = df[target_column].copy()

print("\nData preparation completed.")
print("Rows available for modelling:", len(df))
print("Target variable:", target_column)
print("Features:", feature_columns)


# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ------------------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------------------

categorical_features = [
    "Ship Mode",
    "Region",
    "Division",
    "Product ID",
    "Product Name"
]

numerical_features = ["Units"]

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)


# ------------------------------------------------------------
# 6. DEFINE MACHINE LEARNING MODELS
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        max_depth=12,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ------------------------------------------------------------
# 7. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------

results = []
trained_models = {}

print("\n" + "=" * 60)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 60)

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[model_name] = pipeline

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


# ------------------------------------------------------------
# 8. COMPARE MODELS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    "ml_model_comparison.csv",
    index=False
)

print("\nModel comparison saved to ml_model_comparison.csv")


# ------------------------------------------------------------
# 10. SAVE BEST MODEL
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "best_profit_model.joblib"
)

print("\nBest model by test MAE:", best_model_name)
print("Saved model to best_profit_model.joblib")


# ------------------------------------------------------------
# 11. SHOW SAMPLE PREDICTIONS
# ------------------------------------------------------------

sample_count = min(10, len(X_test))

sample_X = X_test.iloc[:sample_count]
sample_actual = y_test.iloc[:sample_count]

sample_predictions = best_model.predict(sample_X)

sample_results = pd.DataFrame({
    "Actual Gross Profit": sample_actual.values,
    "Predicted Gross Profit": sample_predictions
})

sample_results["Prediction Error"] = (
    sample_results["Actual Gross Profit"]
    - sample_results["Predicted Gross Profit"]
)

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(sample_results.round(2).to_string(index=False))


# ------------------------------------------------------------
# 12. IMPORTANT LIMITATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("IMPORTANT PROJECT NOTES")
print("=" * 60)

print(
    "1. These models predict Gross Profit, not shipping lead time."
)

print(
    "2. Sales and Cost were excluded from the features "
    "to prevent target leakage."
)

print(
    "3. Shipping dates must be validated before training "
    "a lead-time prediction model."
)

print(
    "4. This model does not calculate actual shipping-cost "
    "savings or factory reassignment savings."
)

print(
    "5. Model performance must be assessed on unseen data "
    "before relying on predictions."
)

print("\nMachine learning experiment completed successfully!")