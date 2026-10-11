# WindSense

WindSense is a wind turbine performance monitoring project built with Python and machine learning.

The goal is to estimate how much power a turbine is expected to produce under current environmental conditions, compare that prediction with actual production, and identify unusual sensor behavior that may require further review.

The project uses a small synthetic wind turbine dataset and focuses on building a complete data science workflow rather than diagnosing real mechanical failures.

---

## Overview

WindSense follows two monitoring paths:

**Performance monitoring**
- predicts expected power output
- compares expected and actual production
- flags unusually large production shortfalls

**Sensor monitoring**
- checks vibration, rotor speed, and gearbox temperature
- compares each turbine with its own historical behavior
- considers the expected operating load: LOW, MEDIUM, or HIGH
- assigns a severity score to unusual sensor values

The final output combines both monitoring paths into a single review table.

---

## Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Time-Series Validation
   ↓
Expected Power Prediction
   ↓
Performance Monitoring
   +
Sensor Monitoring
   ↓
Review Cases
   ↓
CSV Reports + Visualizations
```

---

## Machine Learning

The model predicts `power_output` using environmental features:

- wind speed
- temperature
- wind direction

Wind direction is transformed into sine and cosine components because direction is circular.

Three regression models were compared:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 72.52 | 97.16 | 0.951 |
| Random Forest | 37.54 | 51.49 | 0.986 |
| Gradient Boosting | 38.36 | 50.75 | 0.987 |

Random Forest was selected based on expanding-window time-series validation.

Average validation results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 54.41 | 70.13 | 0.977 |
| Gradient Boosting | 56.33 | 71.57 | 0.976 |
| Linear Regression | 78.55 | 108.43 | 0.946 |

---

## Monitoring Logic

Random Forest estimates the power output expected under the observed environmental conditions.

Performance shortfall is calculated as:

```text
predicted_power - actual_power
```

A statistical review threshold of `147.99` was derived from validation prediction errors.

Sensor monitoring evaluates:

- vibration
- rotor speed
- gearbox temperature

Each turbine is compared with its own historical behavior.

Historical observations are divided into LOW, MEDIUM, and HIGH operating-load bands. For new observations, the expected load band is determined using predicted power.

Sensor boundaries are calculated using the IQR method, and a severity score indicates how far a flagged value lies outside its expected range.

---

## Results

During the final test period:

- Performance review cases: 0
- Sensor review cases: 8

The highest-severity sensor review case was observed for turbine T03 at high load, where vibration exceeded its historical load-aware range.

```text
Turbine: T03
Time: 2026-09-05 11:00
Load: HIGH
Sensor: vibration
Severity score: 1.43
```

Review cases represent statistical deviations and are not confirmed mechanical failures.

---

## Visualizations

### Actual vs Predicted Power

![Actual vs Predicted Power](reports/figures/actual_vs_predicted_power.png)

### Sensor Review Severity

![Sensor Review Severity](reports/figures/sensor_review_severity.png)

---

## Project Structure

```text
WindSense/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   ├── figures/
│   ├── model_metrics.csv
│   ├── test_predictions.csv
│   └── windsense_review_cases.csv
│
├── scripts/
│   └── generate_data.py
│
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── eda.py
│   ├── train_model.py
│   ├── validate_models.py
│   └── create_report.py
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python run_pipeline.py
```

Generated outputs are stored under:

```text
reports/
```

---

## Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Git · GitHub

---

## Limitations

WindSense currently uses a small synthetic dataset covering five days of observations.

The sensor thresholds are statistical limits derived from the available historical data and should not be interpreted as engineering safety limits.

The current pipeline retrains the model on every execution. In a production environment, model training and daily inference would normally be separated.

The current dataset also does not contain maintenance records, SCADA alarms, curtailment information, or confirmed fault labels.

---

## Future Improvements

- use longer-term real SCADA data
- separate model training and inference pipelines
- save and reuse trained models
- add automated scheduling
- add model drift monitoring
- integrate maintenance and alarm records
- build an interactive monitoring dashboard