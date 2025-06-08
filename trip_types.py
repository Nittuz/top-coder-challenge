def get_trip_type(row):
    if row['trip_duration_days'] <= 3 and row['total_receipts_amount'] > 1000:
        return "short_luxe"
    elif row['trip_duration_days'] == 5:
        return "sweetspot"
    elif row['total_receipts_amount'] > 2000:
        return "big_budget"
    elif row['miles_traveled'] >= 800:
        return "mileage_heavy"
    elif 4 <= row['trip_duration_days'] <= 8 and 300 <= row['total_receipts_amount'] <= 1200:
        return "balanced"
    else:
        return "default"
