import pandas as pd
import numpy as np

np.random.seed(42)

n_hours = 120
timestamps = pd.date_range(
    start="2026-09-01 00:00",
    periods=n_hours,
    freq="h"
)

turbines = ["T01", "T02", "T03"]
all_data = []

for i, turbine in enumerate(turbines):
    wind_speed = np.clip(
        np.random.normal(7.5 + i * 0.3, 1.8, n_hours),
        0,
        20
    )

    wind_direction = np.random.uniform(0, 360, n_hours)
    temperature = np.random.normal(22, 3, n_hours)

    gearbox_temperature = (
        45
        + 2.2 * wind_speed
        + np.random.normal(0, 2, n_hours)
    )

    rotor_speed = (
        1.6 * wind_speed
        + np.random.normal(0, 0.7, n_hours)
    )

    vibration = (
        0.08
        + 0.018 * wind_speed
        + np.random.normal(0, 0.02, n_hours)
    )

    power_output = np.clip(
        20 * wind_speed ** 2
        + np.random.normal(0, 60, n_hours),
        0,
        2000
    )

    turbine_data = pd.DataFrame({
        "timestamp": timestamps,
        "turbine_id": turbine,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "temperature": temperature,
        "gearbox_temperature": gearbox_temperature,
        "rotor_speed": rotor_speed,
        "vibration": vibration,
        "power_output": power_output
    })

    all_data.append(turbine_data)

df = pd.concat(all_data, ignore_index=True)

# Bilerek birkaç veri problemi oluşturuyoruz
df.loc[15, "gearbox_temperature"] = np.nan
df.loc[207, "wind_speed"] = np.nan
df.loc[301, "power_output"] = np.nan

# Bilerek anormal sensör değeri
df.loc[95, "vibration"] = 2.8

# Bilerek duplicate kayıtlar
df = pd.concat(
    [df, df.iloc[[25, 180]]],
    ignore_index=True
)

df.to_csv(
    "data/raw/turbine_measurements.csv",
    index=False
)

print("Dataset created.")
print("Shape:", df.shape)