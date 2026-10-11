import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD REPORT DATA
# ============================================================

# All test-period predictions
predictions = pd.read_csv(
    "reports/test_predictions.csv"
)

# Final review cases
review_cases = pd.read_csv(
    "reports/windsense_review_cases.csv"
)

# Convert timestamp text into datetime
predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"]
)

review_cases["timestamp"] = pd.to_datetime(
    review_cases["timestamp"]
)


# ============================================================
# 2. ACTUAL VS PREDICTED POWER
# ============================================================

plt.figure(
    figsize=(12, 6)
)

for turbine_id in sorted(
    predictions["turbine_id"].unique()
):

    turbine_data = predictions[
        predictions["turbine_id"] == turbine_id
    ]

    # Real power production
    plt.plot(
        turbine_data["timestamp"],
        turbine_data["power_output"],
        marker="o",
        label=f"{turbine_id} Actual"
    )

    # Random Forest expected power
    plt.plot(
        turbine_data["timestamp"],
        turbine_data["predicted_power"],
        linestyle="--",
        label=f"{turbine_id} Predicted"
    )


plt.title(
    "WindSense - Actual vs Predicted Power"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Power Output"
)

plt.xticks(
    rotation=45
)

plt.legend(
    ncol=2
)

plt.tight_layout()

plt.savefig(
    "reports/figures/actual_vs_predicted_power.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. SENSOR REVIEW SEVERITY
# ============================================================

# Keep only sensor review cases
sensor_reviews = review_cases[
    review_cases["sensor_review_required"]
].copy()

# Lowest severity first so the highest value
# appears at the top of the horizontal chart
sensor_reviews = sensor_reviews.sort_values(
    "sensor_severity_score",
    ascending=True
)

# Create readable labels
sensor_reviews["case_label"] = (
    sensor_reviews["turbine_id"]
    + " | "
    + sensor_reviews["timestamp"]
        .dt.strftime("%H:%M")
    + " | "
    + sensor_reviews["flagged_sensors"]
)


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    sensor_reviews["case_label"],
    sensor_reviews["sensor_severity_score"]
)

plt.title(
    "WindSense - Sensor Review Severity"
)

plt.xlabel(
    "Sensor Severity Score"
)

plt.ylabel(
    "Review Case"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/sensor_review_severity.png",
    dpi=300
)

plt.close()


print(
    "Saved:",
    "reports/figures/actual_vs_predicted_power.png"
)

print(
    "Saved:",
    "reports/figures/sensor_review_severity.png"
)