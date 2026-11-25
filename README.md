# AI Silent Failure Detection

An end-to-end MLOps project for **synthetic e-commerce fraud detection**. The project serves fraud predictions through FastAPI and monitors silent model failures through data drift, anomaly detection, feature validation, training-serving skew checks, and a unified health score.

> **Research disclosure:** the e-commerce data is synthetic. Reported metrics and controlled-failure results are demonstrations on simulated data; they are not production fraud-performance claims.

## Features

- Multi-model fraud experiments: Logistic Regression, Random Forest, Extra Trees, and XGBoost.
- MLflow tracking, candidate selection, and model registration using F2 score.
- FastAPI prediction endpoint with input validation and SHAP feature contributions.
- Monitoring endpoints for drift, anomalies, validation, skew, unified health, and history.
- PSI and Kolmogorov-Smirnov (KS) drift checks for numerical data; category PSI and unseen-category checks for categorical data.
- Isolation Forest anomaly detection.
- Streamlit dashboard for model metrics, monitoring outcomes, and controlled experiments.
- SQLite event storage and reproducible test batches.

## Architecture

```text
Synthetic data -> Model training + MLflow -> Reference profile / feature contract
                                                    |
FastAPI prediction -> Validation -> Prediction -> Anomaly / drift / skew monitoring
                                                    |
                               Unified health score -> Dashboard + SQLite history
```

## Project structure

```text
api/           FastAPI application and request schemas
dashboard/     Streamlit monitoring dashboard
monitoring/    Drift, anomaly, validation, skew, health, and SHAP modules
scripts/       Dataset generation, training, experiments, and test batches
models/        Generated local model artifacts (not committed)
database/      SQLite tracking and monitoring databases
logs/          Generated monitoring batches and controlled results
tests/         Automated tests
```

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/YOUR-USERNAME/AI-Silent-Failure-Detection.git
cd AI-Silent-Failure-Detection
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using `.\.venv\Scripts\python.exe` avoids PowerShell execution-policy issues. You do not need to activate the environment.

### 2. Prepare data, train, and create monitoring artifacts

Run these once after installation:

```powershell
.\.venv\Scripts\python.exe scripts\generate_ecommerce_data.py --rows 12000
.\.venv\Scripts\python.exe scripts\run_mlflow_experiments.py
.\.venv\Scripts\python.exe scripts\select_best_model.py
.\.venv\Scripts\python.exe scripts\create_monitoring_batches.py
.\.venv\Scripts\python.exe scripts\run_failure_experiments.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Run the application

Open separate VS Code terminals in the project folder.

### Terminal 1 — FastAPI service

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload
```

API documentation: http://127.0.0.1:8000/docs

### Terminal 2 — Streamlit dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Dashboard: http://localhost:8501

### Terminal 3 — MLflow UI (optional)

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///C:/Final_Year_Project/AI-Silent-Failure-Detection/database/mlflow_tracking.db" --port 5000
```

MLflow UI: http://127.0.0.1:5000

Stop any running service with `Ctrl + C`.

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Validate a transaction, estimate fraud probability, log an event, and return a SHAP explanation. |
| `POST /monitor/drift` | Compare a 30+ transaction batch with the reference distribution. |
| `POST /monitor/anomaly` | Score a batch for unusual transaction and prediction combinations. |
| `POST /monitor/skew` | Compare serving columns and preprocessing version with the training contract. |
| `POST /monitor/assess` | Run validation, drift, anomaly, skew, and unified health scoring together. |
| `POST /monitor/health` | Demonstrate health scoring with supplied monitoring signals. |
| `GET /monitor/history` | Read stored monitoring events. |
| `GET /health` | Confirm that the API and local model are ready. |

### Example prediction

Use Swagger at `/docs`, select `POST /predict`, choose **Try it out**, and use:

```json
{
  "transaction_timestamp": "2026-11-11T14:30:00Z",
  "customer_age": 31,
  "country": "India",
  "membership_tier": "Basic",
  "customer_tenure_days": 540,
  "product_category": "Electronics",
  "items_count": 2,
  "unit_price": 450,
  "discount_percent": 10,
  "order_value": 810,
  "payment_method": "Credit Card",
  "device_type": "Mobile",
  "shipping_distance_km": 25,
  "delivery_days_estimate": 3,
  "is_weekend": 0,
  "hour_of_day": 14,
  "previous_orders": 18,
  "previous_chargebacks": 0
}
```

## Monitoring demonstrations

After the API is running, use the Swagger interface or the prepared JSON files in `logs/`:

- `normal_batch.json` — baseline transaction batch.
- `drift_batch.json` — shifted transaction distribution for data-drift demonstration.
- `anomaly_batch.json` — 30 valid transactions containing deliberate anomaly cases.
- `controlled_failure_experiments.json` — expected outcomes for normal, drift, anomaly, skew, and combined scenarios.

Example PowerShell request:

```powershell
$body = Get-Content .\logs\drift_batch.json -Raw
Invoke-RestMethod -Uri "http://127.0.0.1:8000/monitor/drift" -Method POST -ContentType "application/json" -Body $body
```

## GitHub hygiene

Do not commit virtual environments, generated model artifacts, raw data, MLflow databases, or secrets. The `.gitignore` file should exclude at least:

```text
.venv/
mlruns/
mlflow.db
*.joblib
.env
```

## Future improvements

- Add a real, governed fraud dataset and evaluate generalisation.
- Add authenticated deployment, role-based dashboard access, and alert integrations.
- Add model promotion gates and retraining policies driven by health-score trends.

