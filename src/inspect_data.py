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

print("\nUnique turbines:")
print(df["turbine_id"].unique())

print("\nTime range:")
print(df["timestamp"].min())
print(df["timestamp"].max())
print("\nAverage power by turbine:")
print(df.groupby("turbine_id")["power_output"].mean())
print("\nWind speed and power correlation:")
print(df[["wind_speed", "power_output"]].corr())
numeric_columns = df.select_dtypes(include="number").columns

print("\nNumeric columns:")
print(numeric_columns)
print("\nNumeric summary:")
print(df[numeric_columns].describe().T)

measurement_columns = [
    "wind_speed",
    "temperature",
    "gearbox_temperature",
    "rotor_speed",
    "vibration",
    "power_output"
]

print("\n========== IQR OUTLIER CHECK ==========")

outlier_flags = pd.DataFrame(index=df.index)

outlier_severity = pd.DataFrame(
    0.0,
    index=df.index,
    columns=measurement_columns
)

for column in measurement_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    is_outlier = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    outlier_flags[column] = is_outlier

    severity = pd.Series(0.0, index=df.index)

    below_lower = df[column] < lower_bound
    above_upper = df[column] > upper_bound

    severity.loc[below_lower] = (
        lower_bound - df.loc[below_lower, column]
    ) / iqr

    severity.loc[above_upper] = (
        df.loc[above_upper, column] - upper_bound
    ) / iqr

    outlier_severity[column] = severity

    outliers = df[is_outlier]

    print(
        column,
        "| lower:", round(lower_bound, 3),
        "| upper:", round(upper_bound, 3),
        "| outliers:", len(outliers)
    )

    if len(outliers) > 0:
        print(
            outliers[
                ["timestamp", "turbine_id", column]
            ].to_string(index=False)
        )


outlier_summary = df[["timestamp", "turbine_id"]].copy()

outlier_summary["outlier_count"] = outlier_flags.sum(axis=1)

outlier_summary["outlier_sensors"] = outlier_flags.apply(
    lambda row: ", ".join(row[row].index),
    axis=1
)

outlier_summary["max_severity"] = outlier_severity.max(axis=1)

outlier_summary["most_extreme_sensor"] = outlier_severity.idxmax(axis=1)

outlier_summary = outlier_summary[
    outlier_summary["outlier_count"] > 0
]

outlier_summary = outlier_summary.sort_values(
    ["max_severity", "outlier_count"],
    ascending=False
)

print("\n========== OUTLIERS PER ROW ==========")
print(outlier_summary.to_string(index=False))

event_time = pd.Timestamp("2026-09-04 23:00:00")
event_turbine = "T01"

event_window = df[
    (df["turbine_id"] == event_turbine) &
    (
        df["timestamp"].between(
            event_time - pd.Timedelta(hours=2),
            event_time + pd.Timedelta(hours=2)
        )
    )
]

print("\n========== HIGH-SEVERITY EVENT INVESTIGATION ==========")
print(event_window.to_string(index=False))

event_time_t03 = pd.Timestamp("2026-09-02 03:00:00")
event_turbine_t03 = "T03"

event_window_t03 = df[
    (df["turbine_id"] == event_turbine_t03) &
    (
        df["timestamp"].between(
            event_time_t03 - pd.Timedelta(hours=2),
            event_time_t03 + pd.Timedelta(hours=2)
        )
    )
]

print("\n========== T03 MULTI-SENSOR EVENT INVESTIGATION ==========")
print(event_window_t03.to_string(index=False))


print("\n========== DUPLICATE ROWS ==========")

duplicate_rows = df[
    df.duplicated(keep=False)
]

print(
    duplicate_rows
    .sort_values(["turbine_id", "timestamp"])
    .to_string(index=False)
)


print("\n========== ROWS WITH MISSING VALUES ==========")

missing_rows = df[
    df.isna().any(axis=1)
]

print(missing_rows.to_string(index=False))

missing_context = df[
    (
        (df["turbine_id"] == "T01") &
        (df["timestamp"].between(
            "2026-09-01 13:00:00",
            "2026-09-01 17:00:00"
        ))
    )
    |
    (
        (df["turbine_id"] == "T02") &
        (df["timestamp"].between(
            "2026-09-04 13:00:00",
            "2026-09-04 17:00:00"
        ))
    )
    |
    (
        (df["turbine_id"] == "T03") &
        (df["timestamp"].between(
            "2026-09-03 11:00:00",
            "2026-09-03 15:00:00"
        ))
    )
]

print("\n========== MISSING VALUE CONTEXT ==========")
print(missing_context.to_string(index=False))