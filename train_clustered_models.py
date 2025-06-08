import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from features import add_features

# Load data
df = pd.read_csv("clustered_cases.csv")

# Override clusters manually based on known legacy patterns
def manual_cluster(row):
    if row["trip_duration_days"] == 5:
        return 0  # Sweet spot trips
    elif row["total_receipts_amount"] >= 1000:
        return 1  # High-spend trips
    else:
        return 2  # Everything else

df["cluster"] = df.apply(manual_cluster, axis=1)

# Train models per new cluster
for cluster_id in sorted(df["cluster"].unique()):
    cluster_df = df[df["cluster"] == cluster_id].copy()
    X = add_features(cluster_df[["trip_duration_days", "miles_traveled", "total_receipts_amount"]])
    y = cluster_df["expected_output"]

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
    print(f"✅ Manual Cluster {cluster_id} → MAE: ${mae:.2f}")

    joblib.dump(model, f"model_cluster_{cluster_id}.joblib")
