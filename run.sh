#!/bin/bash

DURATION=$1
MILES=$2
RECEIPTS=$3

python3 -c "
import sys
import pandas as pd
import numpy as np
import joblib
from features import add_features

# === Trip Type Classification ===
def get_trip_type(row):
    if row['trip_duration_days'] <= 3 and row['total_receipts_amount'] > 1000:
        return 'short_luxe'
    elif row['trip_duration_days'] == 5:
        return 'sweetspot'
    elif row['total_receipts_amount'] > 2000:
        return 'big_budget'
    elif row['miles_traveled'] >= 800:
        return 'mileage_heavy'
    elif 4 <= row['trip_duration_days'] <= 8 and 300 <= row['total_receipts_amount'] <= 1200:
        return 'balanced'
    else:
        return 'default'

# === Input and Feature Prep ===
input_df = pd.DataFrame([{
    'trip_duration_days': int(${DURATION}),
    'miles_traveled': float(${MILES}),
    'total_receipts_amount': float(${RECEIPTS})
}])

row = input_df.iloc[0]
trip_type = get_trip_type(row)
X = add_features(input_df)

# === Model Prediction ===
model = joblib.load(f'model_{trip_type}.joblib')
prediction = model.predict(X)[0]

# === Bonus Logic ===
bonus = 0
if trip_type == 'sweetspot' and row['total_receipts_amount'] < 1000:
    bonus += 25

# === Final Tiered Penalty Logic ===
receipts = row['total_receipts_amount']
days = row['trip_duration_days']
miles = row['miles_traveled']

penalty = 0

if days <= 3 and receipts > 1200:
    penalty = 0.55 * (receipts - 1000)
elif days <= 5 and receipts > 1400:
    penalty = 0.45 * (receipts - 1200)
elif days <= 8 and receipts > 1800:
    penalty = 0.35 * (receipts - 1400)
elif receipts > 2000:
    penalty = 0.3 * (receipts - 1500)

penalty_cap = 300 + 30 * (days // 2)
penalty = min(penalty, penalty_cap)

# === Final Adjustment ===
prediction = prediction + bonus - penalty

# === Safety Check ===
if not np.isfinite(prediction):
    prediction = 0.0

print(f'{prediction:.2f}')
"
