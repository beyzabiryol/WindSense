import subprocess
import sys
from pathlib import Path


# ============================================================
# WINDSENSE PIPELINE
# ============================================================

# Make sure output folders exist
Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

Path("reports/figures").mkdir(
    parents=True,
    exist_ok=True
)


def run_step(name, script_path):
    """
    Run one WindSense pipeline step.
    Stop the pipeline if that step fails.
    """

    print(
        f"\n========== {name} =========="
    )

    subprocess.run(
        [
            sys.executable,
            script_path
        ],
        check=True
    )


# ============================================================
# 1. CLEAN DATA
# ============================================================

run_step(
    "DATA CLEANING",
    "src/clean_data.py"
)


# ============================================================
# 2. TRAIN AND EVALUATE MODEL
# ============================================================

run_step(
    "MODEL TRAINING AND MONITORING",
    "src/train_model.py"
)


# ============================================================
# 3. CREATE REPORTS
# ============================================================

run_step(
    "REPORT GENERATION",
    "src/create_report.py"
)


print(
    "\n======================================"
)

print(
    "WindSense pipeline completed successfully."
)

print(
    "======================================"
)

print(
    "\nMain outputs:"
)

print(
    "- reports/windsense_review_cases.csv"
)

print(
    "- reports/model_metrics.csv"
)

print(
    "- reports/test_predictions.csv"
)

print(
    "- reports/figures/actual_vs_predicted_power.png"
)

print(
    "- reports/figures/sensor_review_severity.png"
)