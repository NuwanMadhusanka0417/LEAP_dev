import pandas as pd

# Load
df = pd.read_csv("data/SolarMeterReadings1hour_cleaned_v2.csv")

# Parse timestamp (adjust column name if yours differs)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Keep only 2020 rows
df_2020 = df[df["timestamp"].dt.year == 2021].copy()

# Save
df_2020.to_csv("data/SolarMeterReadings1hour_cleaned_v2_2021.csv", index=False)

print("Saved rows:", len(df_2020))
print("Date range:", df_2020["timestamp"].min(), "to", df_2020["timestamp"].max())