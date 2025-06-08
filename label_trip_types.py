import json
import pandas as pd
from trip_types import get_trip_type

# Load public cases
with open("public_cases.json") as f:
    data = json.load(f)

# Flatten and assign type
records = []
for d in data:
    row = {
        "trip_duration_days": d["input"]["trip_duration_days"],
        "miles_traveled": d["input"]["miles_traveled"],
        "total_receipts_amount": d["input"]["total_receipts_amount"],
        "expected_output": d["expected_output"]
    }
    row["trip_type"] = get_trip_type(row)
    records.append(row)

df = pd.DataFrame(records)
df.to_csv("typed_cases.csv", index=False)
print("✅ Saved typed cases to typed_cases.csv")
