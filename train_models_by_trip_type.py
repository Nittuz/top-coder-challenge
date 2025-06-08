import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from features import add_features

# Load typed dataset
df = pd.read_csv("typed_cases.csv")

# Train one model per trip_type
trip_types = df["trip_type"].unique()

for trip_type in trip_types:
    subset = df[df["trip_type"] == trip_type].copy()
    X = add_features(subset[["trip_duration_days", "miles_traveled", "total_receipts_amount"]])
    y = subset["expected_output"]

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=64,
        random_state=42
    )
    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"✅ Trained model for {trip_type}: MAE ${mae:.2f} ({len(subset)} records)")

    joblib.dump(model, f"model_{trip_type}.joblib")
