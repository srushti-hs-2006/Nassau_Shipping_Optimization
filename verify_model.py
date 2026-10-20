
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

EXCEL_FILE = BASE_DIR / "Nassau Candy Distributor.xlsx"
MODEL_FILE = BASE_DIR / "best_profit_model.joblib"

FEATURES = [
    "Ship Mode",
    "Region",
    "Division",
    "Product ID",
    "Product Name",
    "Units",
]

print("=" * 55)
print("NASSAU CANDY - MODEL INTEGRATION CHECK")
print("=" * 55)

# 1. Check that both files exist
for file in [EXCEL_FILE, MODEL_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")

print("Excel dataset: FOUND")
print("Saved ML model: FOUND")

# 2. Load the saved model and dataset
model = joblib.load(MODEL_FILE)
df = pd.read_excel(EXCEL_FILE)

missing = [col for col in FEATURES + ["Gross Profit"]
           if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
df["Gross Profit"] = pd.to_numeric(
    df["Gross Profit"], errors="coerce"
)

# 3. Select one valid sample
eligible = df[
    (df["Units"] > 0)
    & df["Gross Profit"].notna()
]

if eligible.empty:
    raise ValueError("No valid sample rows found.")

sample = eligible.iloc[[0]]
X_sample = sample[FEATURES]

# 4. Generate a prediction using the saved model
prediction = float(model.predict(X_sample)[0])
actual = float(sample["Gross Profit"].iloc[0])

print("\nModel loaded successfully!")
print("Input features:", FEATURES)
print("Actual Gross Profit:", round(actual, 2))
print("Predicted Gross Profit:", round(prediction, 2))
print("Absolute prediction error:", round(abs(actual - prediction), 2))

print("\nMODEL INTEGRATION CHECK PASSED")
print("The saved model accepts the expected input columns.")