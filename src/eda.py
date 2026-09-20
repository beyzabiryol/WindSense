import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/raw/turbine_measurements.csv")
plt.scatter(df["wind_speed"], df["power_output"])
plt.xlabel("Wind Speed")
plt.ylabel("Power Output")
plt.title("Wind Speed vs Power Output")
plt.show()