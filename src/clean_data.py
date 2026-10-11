import pandas as pd

df = pd.read_csv("data/raw/turbine_measurements.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

rows_before = len(df)

df = df.drop_duplicates()

rows_after = len(df)

print("Rows before duplicate removal:", rows_before)
print("Rows after duplicate removal:", rows_after)
print("Duplicates removed:", rows_before - rows_after)
print("Remaining duplicates:", df.duplicated().sum())


df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)

df["gearbox_temperature_imputed"] = df["gearbox_temperature"].isna()

df["gearbox_temperature"] = (
    df.groupby("turbine_id")["gearbox_temperature"]
    .transform(lambda x: x.interpolate())
)

print("\nMissing values after cleaning:")
print(df.isna().sum())

print("\nImputed gearbox temperature:")
print(
    df.loc[
        df["gearbox_temperature_imputed"],
        [
            "timestamp",
            "turbine_id",
            "gearbox_temperature",
            "gearbox_temperature_imputed"
        ]
    ].to_string(index=False)
)

df["wind_speed_missing"] = df["wind_speed"].isna()
df["power_output_missing"] = df["power_output"].isna()

df.to_csv(
    "data/processed/turbine_measurements_clean.csv",
    index=False
)

print("\nClean dataset saved.")
print("Final shape:", df.shape)

clean_df = pd.read_csv(
    "data/processed/turbine_measurements_clean.csv"
)

print("\n========== PROCESSED DATA CHECK ==========")
print("Shape:", clean_df.shape)
print("Duplicate rows:", clean_df.duplicated().sum())

print("\nMissing values:")
print(clean_df.isna().sum())