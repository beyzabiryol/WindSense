import pandas as pd

df = pd.read_csv("data/raw/turbine_measurements.csv")

print(df.head())
print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nInfo:")
df.info()
print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\nData types after timestamp conversion:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

print("\nUnique turbines:")
print(df["turbine_id"].unique())

print("\nTime range:")
print(df["timestamp"].min())
print(df["timestamp"].max())
print("\nAverage power by turbine:")
print(df.groupby("turbine_id")["power_output"].mean())
print("\nWind speed and power correlation:")
print(df[["wind_speed", "power_output"]].corr())