import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

# Load the cleaned dataset
df = pd.read_csv(
    "data/processed/turbine_measurements_clean.csv"
)

# Convert timestamp text into datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Keep only rows that can be used for modeling
# wind_speed is a feature and power_output is the target
model_df = df.dropna(
    subset=["wind_speed", "power_output"]
).copy()


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Convert wind direction from degrees to radians
wind_direction_radians = np.deg2rad(
    model_df["wind_direction"]
)

# Convert circular wind direction into model-friendly features
model_df["wind_direction_sin"] = np.sin(
    wind_direction_radians
)

model_df["wind_direction_cos"] = np.cos(
    wind_direction_radians
)

# Sort observations chronologically
model_df = model_df.sort_values(
    ["timestamp", "turbine_id"]
).reset_index(drop=True)

# Create a date column for day-by-day validation
model_df["date"] = model_df["timestamp"].dt.normalize()

# Features used by every model
features = [
    "wind_speed",
    "temperature",
    "wind_direction_sin",
    "wind_direction_cos"
]


# ============================================================
# 3. KEEP FINAL TEST DAY OUTSIDE VALIDATION
# ============================================================

# The latest day will remain outside model validation
# and will be used as the final test period
final_test_date = model_df["date"].max()

development_df = model_df[
    model_df["date"] < final_test_date
].copy()

# Validation starts from the second available day
# because the first day is required as initial training data
validation_dates = sorted(
    development_df["date"].unique()
)[1:]


# ============================================================
# 4. CREATE MODELS
# ============================================================

# Create fresh models for every validation round
def create_models():
    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=3,
            random_state=42
        )
    }


# Store daily model evaluation results
validation_results = []

# Store individual Random Forest predictions
# These will later be used to calculate the review threshold
rf_validation_predictions = []


# ============================================================
# 5. TIME-SERIES VALIDATION
# ============================================================

# Expanding-window validation:
# train on the past and validate on the next day
for validation_date in validation_dates:

    # Use only dates before the validation day for training
    train_fold = development_df[
        development_df["date"] < validation_date
    ].copy()

    # Use the current validation day as unseen data
    validation_fold = development_df[
        development_df["date"] == validation_date
    ].copy()

    X_train_fold = train_fold[features]
    y_train_fold = train_fold["power_output"]

    X_validation = validation_fold[features]
    y_validation = validation_fold["power_output"]

    # Train and evaluate all models on the same validation period
    for model_name, model in create_models().items():

        # Train the model using past observations
        model.fit(
            X_train_fold,
            y_train_fold
        )

        # Predict power output for the unseen validation day
        predictions = model.predict(
            X_validation
        )

        # Store individual Random Forest prediction errors
        # because Random Forest was the best overall model
        if model_name == "Random Forest":

            for timestamp, turbine_id, actual, predicted in zip(
                validation_fold["timestamp"],
                validation_fold["turbine_id"],
                y_validation,
                predictions
            ):
                rf_validation_predictions.append(
                    {
                        "timestamp": timestamp,
                        "turbine_id": turbine_id,
                        "actual_power": actual,
                        "predicted_power": predicted,

                        # Positive value means actual production
                        # was lower than model expectation
                        "power_shortfall": predicted - actual
                    }
                )

        # Calculate model evaluation metrics
        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        rmse = mean_squared_error(
            y_validation,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_validation,
            predictions
        )

        # Save daily validation performance
        validation_results.append(
            {
                "validation_date": validation_date.date(),
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }
        )


# ============================================================
# 6. MODEL VALIDATION RESULTS
# ============================================================

# Convert validation results into a DataFrame
results_df = pd.DataFrame(
    validation_results
)

print("\n========== VALIDATION RESULTS ==========")

print(
    results_df
    .round(3)
    .to_string(index=False)
)


# Calculate average performance across all validation days
summary = (
    results_df
    .groupby("model")
    .agg(
        MAE_mean=("MAE", "mean"),
        RMSE_mean=("RMSE", "mean"),
        R2_mean=("R2", "mean")
    )
    .sort_values("RMSE_mean")
)

print("\n========== VALIDATION SUMMARY ==========")

print(
    summary
    .round(3)
    .to_string()
)

print(
    "\nFinal test date kept outside validation:",
    final_test_date.date()
)


# ============================================================
# 7. UNDERPERFORMANCE REVIEW THRESHOLD
# ============================================================

# Convert individual Random Forest validation predictions
# into a DataFrame
rf_validation_df = pd.DataFrame(
    rf_validation_predictions
)

# Keep only cases where actual production
# was lower than the Random Forest prediction
positive_shortfalls = rf_validation_df[
    rf_validation_df["power_shortfall"] > 0
]["power_shortfall"]

# Calculate the review threshold from validation data
# The 95th percentile represents unusually large shortfalls
review_threshold = positive_shortfalls.quantile(0.95)

print("\n========== UNDERPERFORMANCE THRESHOLD ==========")

print(
    "95th percentile shortfall:",
    round(review_threshold, 2)
)