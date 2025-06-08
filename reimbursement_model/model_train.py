import json
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import sys

# Ensure it can find root-level features.py
sys.path.append(os.path.abspath(".."))

from features import add_features

# Load data
with open("../public_cases.json") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([{
    "trip_duration_days": d["input"]["trip_duration_days"],
    "miles_traveled": d["input"]["miles_traveled"],
    "total_receipts_amount": d["input"]["total_receipts_amount"],
    "expected_output": d["expected_output"]
} for d in data])

# Feature engineering
X = add_features(df.drop(columns=["expected_output"]))
y = df["expected_output"]

# Train model
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=64,
    random_state=42
)
print("🔧 Injecting broken model to confirm correct load")
#y = y * 0  # All targets are 0
model.fit(X, y)

# Evaluate
preds = model.predict(X)
mae = mean_absolute_error(y, preds)
print(f"✅ Model training complete. MAE on training set: {mae:.2f}")

# Save model inside root project folder
joblib.dump(model, "../model.joblib")

# Confirm
if os.path.exists("../model.joblib"):
    print("✅ model.joblib saved to project root.")
else:
    print("❌ model.joblib NOT saved.")
