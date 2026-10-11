import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/processed/turbine_measurements_clean.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
plt.scatter(df["wind_speed"], df["power_output"])
plt.xlabel("Wind Speed")
plt.ylabel("Power Output")
plt.title("Wind Speed vs Power Output")
plt.show()

df = df.sort_values(["turbine_id", "timestamp"])

plt.figure(figsize=(12, 6))

for turbine in df["turbine_id"].unique():
    turbine_data = df[df["turbine_id"] == turbine]

    plt.plot(
        turbine_data["timestamp"],
        turbine_data["power_output"],
        label=turbine
    )

plt.xlabel("Time")
plt.ylabel("Power Output")
plt.title("Power Output Over Time by Turbine")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


average_power = (
    df.groupby("turbine_id")["power_output"]
    .mean()
)

print("\nAverage Power Output by Turbine:")
print(average_power)

average_power.plot(kind="bar")

plt.xlabel("Turbine")
plt.ylabel("Average Power Output")
plt.title("Average Power Output by Turbine")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

average_wind = (
    df.groupby("turbine_id")["wind_speed"]
    .mean()
)

print("\nAverage Wind Speed by Turbine:")
print(average_wind)

average_wind.plot(kind="bar")

plt.xlabel("Turbine")
plt.ylabel("Average Wind Speed")
plt.title("Average Wind Speed by Turbine")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

df["wind_speed_bin"] = pd.cut(
    df["wind_speed"],
    bins=[0, 5, 6, 7, 8, 9, 10, 11, float("inf")]
)

power_by_wind = (
    df.groupby(
        ["wind_speed_bin", "turbine_id"],
        observed=True
    )["power_output"]
    .mean()
    .unstack()
)

print("\nAverage Power by Wind Speed Range:")
print(power_by_wind)

power_by_wind.plot(
    marker="o",
    figsize=(10, 6)
)

plt.xlabel("Wind Speed Range")
plt.ylabel("Average Power Output")
plt.title("Power Output by Wind Speed Range and Turbine")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


correlation_columns = [
    "wind_speed",
    "temperature",
    "gearbox_temperature",
    "rotor_speed",
    "vibration",
    "power_output"
]

correlations = (
    df[correlation_columns]
    .corr()["power_output"]
    .sort_values(ascending=False)
)

print("\nCorrelation with Power Output:")
print(correlations)