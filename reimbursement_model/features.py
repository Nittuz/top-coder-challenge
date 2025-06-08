import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Base features
    df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"].clip(lower=1)
    df["receipts_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"].clip(lower=1)
    df["receipts_per_mile"] = df["total_receipts_amount"] / df["miles_traveled"].clip(lower=1)

    # Log transforms
    df["log_miles"] = np.log1p(df["miles_traveled"])
    df["log_receipts"] = np.log1p(df["total_receipts_amount"])
    df["log_days"] = np.log1p(df["trip_duration_days"])

    # Interaction terms
    df["miles_times_receipts"] = df["miles_traveled"] * df["total_receipts_amount"]
    df["days_times_log_receipts"] = df["trip_duration_days"] * df["log_receipts"]

    # Threshold flags (possibly legacy rules)
    df["is_5_day_trip"] = (df["trip_duration_days"] == 5).astype(int)
    df["is_8_day_trip"] = (df["trip_duration_days"] == 8).astype(int)
    df["is_long_trip"] = (df["trip_duration_days"] >= 9).astype(int)

    # Receipt tiers
    df["receipt_tier_1"] = (df["total_receipts_amount"] < 100).astype(int)
    df["receipt_tier_2"] = ((df["total_receipts_amount"] >= 100) & (df["total_receipts_amount"] < 500)).astype(int)
    df["receipt_tier_3"] = ((df["total_receipts_amount"] >= 500) & (df["total_receipts_amount"] < 1000)).astype(int)
    df["receipt_tier_4"] = (df["total_receipts_amount"] >= 1000).astype(int)

    # Mileage tiers
    df["mileage_lt_100"] = (df["miles_traveled"] < 100).astype(int)
    df["mileage_100_500"] = ((df["miles_traveled"] >= 100) & (df["miles_traveled"] < 500)).astype(int)
    df["mileage_500_plus"] = (df["miles_traveled"] >= 500).astype(int)

    return df
