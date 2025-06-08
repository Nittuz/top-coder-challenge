import json
import pandas as pd
from sklearn.cluster import KMeans
from features import add_features

# Load public cases
with open("public_cases.json") as f:
    data = json.load(f)

df = pd.DataFrame([{
    "trip_duration_days": d["input"]["trip_duration_days"],
    "miles_traveled": d["input"]["miles_traveled"],
    "total_receipts_amount": d["input"]["total_receipts_amount"],
    "expected_output": d["expected_output"]
} for d in data])

# Apply feature engineering
X = add_features(df.drop(columns=["expected_output"]))

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Save
df.to_csv("clustered_cases.csv", index=False)
print("✅ Clustered training data saved to clustered_cases.csv")
