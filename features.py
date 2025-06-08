import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"].clip(lower=1)
    df["receipts_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"].clip(lower=1)
    df["receipts_per_mile"] = df["total_receipts_amount"] / df["miles_traveled"].clip(lower=1)

    df["log_miles"] = np.log1p(df["miles_traveled"])
    df["log_receipts"] = np.log1p(df["total_receipts_amount"])
    df["log_days"] = np.log1p(df["trip_duration_days"])

    df["squared_miles"] = df["miles_traveled"] ** 2
    df["squared_receipts"] = df["total_receipts_amount"] ** 2
    df["sqrt_receipts"] = np.sqrt(df["total_receipts_amount"])

    df["interaction_1"] = df["miles_traveled"] * df["total_receipts_amount"]
    df["interaction_2"] = df["trip_duration_days"] * df["log_receipts"]

    df["is_short"] = (df["trip_duration_days"] <= 3).astype(int)
    df["is_sweetspot"] = (df["trip_duration_days"] == 5).astype(int)
    df["is_long"] = (df["trip_duration_days"] >= 10).astype(int)

    return df
