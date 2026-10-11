import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "data/processed/turbine_measurements_clean.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# ============================================================
# 2. PREPARE DATA FOR MODELING
# ============================================================

# wind_speed is an input feature
# power_output is the prediction target
model_df = df.dropna(
    subset=[
        "wind_speed",
        "power_output"
    ]
).copy()


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Wind direction is circular.
# 0 and 360 degrees represent the same direction.
wind_direction_radians = np.deg2rad(
    model_df["wind_direction"]
)

model_df["wind_direction_sin"] = np.sin(
    wind_direction_radians
)

model_df["wind_direction_cos"] = np.cos(
    wind_direction_radians
)

model_df = model_df.sort_values(
    [
        "timestamp",
        "turbine_id"
    ]
).reset_index(
    drop=True
)


features = [
    "wind_speed",
    "temperature",
    "wind_direction_sin",
    "wind_direction_cos"
]

target = "power_output"


# ============================================================
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# Preserve time order.
# Older observations are used for training.
# The latest period simulates future unseen data.
unique_times = (
    model_df["timestamp"]
    .sort_values()
    .unique()
)

split_index = int(
    len(unique_times) * 0.8
)

split_time = unique_times[
    split_index
]


train_df = model_df[
    model_df["timestamp"] < split_time
].copy()

test_df = model_df[
    model_df["timestamp"] >= split_time
].copy()


X_train = train_df[
    features
]

y_train = train_df[
    target
]

X_test = test_df[
    features
]

y_test = test_df[
    target
]


print(
    "Rows available for modeling:",
    len(model_df)
)

print(
    "\nTrain rows:",
    len(train_df)
)

print(
    "Test rows:",
    len(test_df)
)

print(
    "Train starts:",
    train_df["timestamp"].min()
)

print(
    "Train ends:",
    train_df["timestamp"].max()
)

print(
    "Test starts:",
    test_df["timestamp"].min()
)

print(
    "Test ends:",
    test_df["timestamp"].max()
)


# ============================================================
# 5. LINEAR REGRESSION - BASELINE MODEL
# ============================================================

lr_model = LinearRegression()

lr_model.fit(
    X_train,
    y_train
)

lr_predictions = lr_model.predict(
    X_test
)


lr_mae = mean_absolute_error(
    y_test,
    lr_predictions
)

lr_rmse = mean_squared_error(
    y_test,
    lr_predictions
) ** 0.5

lr_r2 = r2_score(
    y_test,
    lr_predictions
)


print(
    "\n========== LINEAR REGRESSION RESULTS =========="
)

print(
    "MAE:",
    round(lr_mae, 2)
)

print(
    "RMSE:",
    round(lr_rmse, 2)
)

print(
    "R2:",
    round(lr_r2, 3)
)


# ============================================================
# 6. RANDOM FOREST - SELECTED MODEL
# ============================================================

# Random Forest was selected using
# expanding-window time-series validation.
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


print(
    "\n========== RANDOM FOREST RESULTS =========="
)

print(
    "MAE:",
    round(rf_mae, 2)
)

print(
    "RMSE:",
    round(rf_rmse, 2)
)

print(
    "R2:",
    round(rf_r2, 3)
)


# ============================================================
# 7. GRADIENT BOOSTING - COMPARISON MODEL
# ============================================================

gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=3,
    random_state=42
)

gb_model.fit(
    X_train,
    y_train
)

gb_predictions = gb_model.predict(
    X_test
)


gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = mean_squared_error(
    y_test,
    gb_predictions
) ** 0.5

gb_r2 = r2_score(
    y_test,
    gb_predictions
)


print(
    "\n========== GRADIENT BOOSTING RESULTS =========="
)

print(
    "MAE:",
    round(gb_mae, 2)
)

print(
    "RMSE:",
    round(gb_rmse, 2)
)

print(
    "R2:",
    round(gb_r2, 3)
)


# ============================================================
# 8. PERFORMANCE UNDERPERFORMANCE ANALYSIS
# ============================================================

# Create results for the final test period.
test_results = test_df[
    [
        "timestamp",
        "turbine_id",
        "power_output"
    ]
].copy()


# Expected power according to Random Forest.
test_results[
    "predicted_power"
] = rf_predictions


# Positive shortfall:
# expected production was higher than actual production.
test_results[
    "power_shortfall"
] = (
    test_results["predicted_power"]
    - test_results["power_output"]
)


# Review threshold learned from
# out-of-sample validation errors.
review_threshold = 147.99


# True means the production shortfall
# is unusually large.
test_results[
    "performance_review_required"
] = (
    test_results["power_shortfall"]
    > review_threshold
)


performance_review = test_results[
    test_results[
        "performance_review_required"
    ]
].copy()


print(
    "\n========== PERFORMANCE REVIEW =========="
)

print(
    performance_review[
        [
            "timestamp",
            "turbine_id",
            "power_output",
            "predicted_power",
            "power_shortfall",
            "performance_review_required"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nPerformance review cases:",
    len(performance_review)
)


# Display the largest positive production shortfalls.
underperformance = test_results[
    test_results["power_shortfall"] > 0
].copy()

underperformance = underperformance.sort_values(
    "power_shortfall",
    ascending=False
)


print(
    "\n========== TOP UNDERPERFORMANCE =========="
)

print(
    underperformance[
        [
            "timestamp",
            "turbine_id",
            "power_output",
            "predicted_power",
            "power_shortfall"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 9. LOAD-AWARE SENSOR REVIEW
# ============================================================

# Operational sensors that will be monitored.
sensor_columns = [
    "vibration",
    "rotor_speed",
    "gearbox_temperature"
]


# LOW / MEDIUM / HIGH represent
# different operating-load levels.
load_band_labels = [
    "LOW",
    "MEDIUM",
    "HIGH"
]


train_sensor_data = train_df.copy()
test_sensor_data = test_df.copy()


# Add expected power from Random Forest.
# The test-period load band will be based
# on expected power, not actual power.
test_sensor_data[
    "predicted_power"
] = rf_predictions


train_sensor_data[
    "load_band"
] = None

test_sensor_data[
    "load_band"
] = None


# Stores True / False review flags.
sensor_flags = pd.DataFrame(
    False,
    index=test_sensor_data.index,
    columns=sensor_columns
)


# Stores how far each sensor is outside
# its expected statistical range.
sensor_severity = pd.DataFrame(
    0.0,
    index=test_sensor_data.index,
    columns=sensor_columns
)


# ============================================================
# 10. TURBINE-SPECIFIC LOAD BANDS
# ============================================================

for turbine_id in sorted(
    train_sensor_data[
        "turbine_id"
    ].unique()
):

    # Historical rows for this turbine.
    turbine_train_indices = (
        train_sensor_data[
            train_sensor_data[
                "turbine_id"
            ] == turbine_id
        ].index
    )

    # Test rows for this turbine.
    turbine_test_indices = (
        test_sensor_data[
            test_sensor_data[
                "turbine_id"
            ] == turbine_id
        ].index
    )


    # Historical production values
    # for this turbine.
    turbine_power = (
        train_sensor_data.loc[
            turbine_train_indices,
            "power_output"
        ]
    )


    # Divide historical production into
    # LOW / MEDIUM / HIGH groups.
    low_boundary = turbine_power.quantile(
        1 / 3
    )

    high_boundary = turbine_power.quantile(
        2 / 3
    )


    load_bins = [
        -np.inf,
        low_boundary,
        high_boundary,
        np.inf
    ]


    # Historical observations use
    # actual historical production.
    train_sensor_data.loc[
        turbine_train_indices,
        "load_band"
    ] = pd.cut(
        train_sensor_data.loc[
            turbine_train_indices,
            "power_output"
        ],
        bins=load_bins,
        labels=load_band_labels,
        include_lowest=True
    ).astype(
        str
    ).values


    # Test observations use EXPECTED power
    # predicted by Random Forest.
    test_sensor_data.loc[
        turbine_test_indices,
        "load_band"
    ] = pd.cut(
        test_sensor_data.loc[
            turbine_test_indices,
            "predicted_power"
        ],
        bins=load_bins,
        labels=load_band_labels,
        include_lowest=True
    ).astype(
        str
    ).values


    print(
        f"\n========== "
        f"{turbine_id} LOAD BANDS "
        f"=========="
    )

    print(
        f"LOW: power <= "
        f"{low_boundary:.2f}"
    )

    print(
        f"MEDIUM: "
        f"{low_boundary:.2f} "
        f"< power <= "
        f"{high_boundary:.2f}"
    )

    print(
        f"HIGH: power > "
        f"{high_boundary:.2f}"
    )


    # ========================================================
    # 11. SENSOR LIMITS INSIDE EACH LOAD BAND
    # ========================================================

    for load_band in load_band_labels:

        # Same turbine + same historical load.
        historical_group = (
            train_sensor_data[
                (
                    train_sensor_data[
                        "turbine_id"
                    ] == turbine_id
                )
                &
                (
                    train_sensor_data[
                        "load_band"
                    ] == load_band
                )
            ]
        )


        # Same turbine + same expected
        # load in the test period.
        test_group_indices = (
            test_sensor_data[
                (
                    test_sensor_data[
                        "turbine_id"
                    ] == turbine_id
                )
                &
                (
                    test_sensor_data[
                        "load_band"
                    ] == load_band
                )
            ].index
        )


        if len(
            test_group_indices
        ) == 0:
            continue


        print(
            f"\n{turbine_id} | "
            f"{load_band} LOAD | "
            f"historical rows: "
            f"{len(historical_group)}"
        )


        for sensor in sensor_columns:

            q1 = historical_group[
                sensor
            ].quantile(
                0.25
            )

            q3 = historical_group[
                sensor
            ].quantile(
                0.75
            )

            iqr = q3 - q1


            # Skip invalid statistical ranges.
            if (
                pd.isna(iqr)
                or iqr == 0
            ):
                continue


            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )


            print(
                f"{sensor} | "
                f"lower: "
                f"{lower_bound:.3f} | "
                f"upper: "
                f"{upper_bound:.3f}"
            )


            sensor_values = (
                test_sensor_data.loc[
                    test_group_indices,
                    sensor
                ]
            )


            # Flag values outside
            # the expected historical range.
            sensor_flags.loc[
                test_group_indices,
                sensor
            ] = (
                (
                    sensor_values
                    < lower_bound
                )
                |
                (
                    sensor_values
                    > upper_bound
                )
            )


            # =================================================
            # SENSOR SEVERITY
            # =================================================

            # Distance below the lower limit,
            # measured in IQR units.
            below_severity = (
                (
                    lower_bound
                    - sensor_values
                )
                / iqr
            ).clip(
                lower=0
            )


            # Distance above the upper limit,
            # measured in IQR units.
            above_severity = (
                (
                    sensor_values
                    - upper_bound
                )
                / iqr
            ).clip(
                lower=0
            )


            sensor_severity.loc[
                test_group_indices,
                sensor
            ] = np.maximum(
                below_severity,
                above_severity
            )


# ============================================================
# 12. CREATE SENSOR REVIEW TABLE
# ============================================================

sensor_review = test_sensor_data[
    [
        "timestamp",
        "turbine_id",
        "power_output",
        "load_band",
        "vibration",
        "rotor_speed",
        "gearbox_temperature"
    ]
].copy()


# True if at least one sensor
# is statistically unusual.
sensor_review[
    "sensor_review_required"
] = sensor_flags.any(
    axis=1
)


# Number of sensors that triggered.
sensor_review[
    "sensor_flag_count"
] = sensor_flags.sum(
    axis=1
)


# Largest deviation among monitored sensors.
sensor_review[
    "sensor_severity_score"
] = sensor_severity.max(
    axis=1
)


# Names of sensors that triggered.
sensor_review[
    "flagged_sensors"
] = sensor_flags.apply(
    lambda row: ", ".join(
        row.index[row]
    ),
    axis=1
)


print(
    "\n========== "
    "LOAD-AWARE SENSOR REVIEW "
    "=========="
)

print(
    sensor_review[
        sensor_review[
            "sensor_review_required"
        ]
    ][
        [
            "timestamp",
            "turbine_id",
            "power_output",
            "load_band",
            "vibration",
            "rotor_speed",
            "gearbox_temperature",
            "sensor_review_required",
            "sensor_flag_count",
            "sensor_severity_score",
            "flagged_sensors"
        ]
    ]
    .sort_values(
        "sensor_severity_score",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print(
    "\nSensor review cases:",
    sensor_review[
        "sensor_review_required"
    ].sum()
)


# ============================================================
# 13. COMBINE PERFORMANCE AND SENSOR REVIEWS
# ============================================================

performance_results = test_results[
    [
        "timestamp",
        "turbine_id",
        "power_output",
        "predicted_power",
        "power_shortfall",
        "performance_review_required"
    ]
].copy()


sensor_results = sensor_review[
    [
        "timestamp",
        "turbine_id",
        "load_band",
        "sensor_review_required",
        "sensor_flag_count",
        "sensor_severity_score",
        "flagged_sensors"
    ]
].copy()


final_review = performance_results.merge(
    sensor_results,
    on=[
        "timestamp",
        "turbine_id"
    ],
    how="left"
)


# ============================================================
# 14. REVIEW REASON
# ============================================================

def get_review_reason(row):

    if (
        row[
            "performance_review_required"
        ]
        and
        row[
            "sensor_review_required"
        ]
    ):
        return "PERFORMANCE_AND_SENSOR"

    if row[
        "performance_review_required"
    ]:
        return "PERFORMANCE"

    if row[
        "sensor_review_required"
    ]:
        return "SENSOR"

    return "NO_REVIEW"


final_review[
    "review_reason"
] = final_review.apply(
    get_review_reason,
    axis=1
)


# Keep only observations requiring review.
review_cases = final_review[
    final_review[
        "review_reason"
    ] != "NO_REVIEW"
].copy()


# Performance alerts first.
# Then sensor cases ordered by severity.
review_cases = review_cases.sort_values(
    [
        "performance_review_required",
        "sensor_severity_score"
    ],
    ascending=[
        False,
        False
    ]
)


# ============================================================
# 15. FINAL WINDSENSE REVIEW
# ============================================================

print(
    "\n========== "
    "WINDSENSE FINAL REVIEW "
    "=========="
)

print(
    review_cases[
        [
            "timestamp",
            "turbine_id",
            "power_output",
            "load_band",
            "predicted_power",
            "power_shortfall",
            "performance_review_required",
            "sensor_review_required",
            "sensor_flag_count",
            "sensor_severity_score",
            "flagged_sensors",
            "review_reason"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nTotal WindSense review cases:",
    len(review_cases)
)


# ============================================================
# 16. SAVE FINAL OUTPUTS
# ============================================================

# Save final review cases.
# This is the main monitoring output.
review_cases.to_csv(
    "reports/windsense_review_cases.csv",
    index=False
)


# Save performance metrics for all models.
model_metrics = pd.DataFrame(
    [
        {
            "model": "Linear Regression",
            "MAE": lr_mae,
            "RMSE": lr_rmse,
            "R2": lr_r2
        },
        {
            "model": "Random Forest",
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "R2": rf_r2
        },
        {
            "model": "Gradient Boosting",
            "MAE": gb_mae,
            "RMSE": gb_rmse,
            "R2": gb_r2
        }
    ]
)

model_metrics.to_csv(
    "reports/model_metrics.csv",
    index=False
)


# Save ALL final test-period predictions.
# Unlike windsense_review_cases.csv,
# this file contains every test observation,
# not only the records requiring review.
test_results.to_csv(
    "reports/test_predictions.csv",
    index=False
)


print(
    "\nSaved:",
    "reports/windsense_review_cases.csv"
)

print(
    "Saved:",
    "reports/model_metrics.csv"
)

print(
    "Saved:",
    "reports/test_predictions.csv"
)