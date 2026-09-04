# AI Silent Failure Detection

A modular MLOps project for detecting and explaining silent failures in machine-learning systems. The planned framework will monitor feature validation, data drift, prediction anomalies, training-serving skew, and model explanations. It will combine those signals into a unified health score with failure classification, severity, and recommended actions.

## Current status

This repository is an initial scaffold. It contains runnable FastAPI and Streamlit placeholders, configuration, and empty monitoring interfaces. Detection logic and model training will be added incrementally.

## Included dataset

`data/raw/ecommerce_transactions.csv` contains 2,000 reproducible synthetic e-commerce transactions. Use `is_fraudulent` as the first model target. See `data/DATA_DICTIONARY.md` for every column and suggested monitoring experiments.

## Planned architecture

```text
Incoming data -> validation -> prediction service -> monitoring signals
                                                   -> health assessment
                                                   -> dashboard / alerts
```

## Quick start

1. Open this folder in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Train the baseline fraud model:

   ```powershell
   .\.venv\Scripts\python.exe scripts\train_model.py
   ```

   This creates `models/fraud_model.joblib`, `models/training_metrics.json`, and
   `models/reference_profile.json`. The model selects a fraud-alert threshold on a
   validation set using F2 score (which prioritizes fraud recall), then reports its
   final results on a separate test set. The reference profile is used later for
   drift and training-serving-skew checks.

5. Run the API:

   ```powershell
   uvicorn api.app:app --reload
   ```

   Visit `http://127.0.0.1:8000/docs` for the API documentation.

   Use **POST `/predict`** in the interactive documentation to submit a transaction.
   It returns the fraud probability, the saved alert threshold, a `fraud_alert` or
   `normal_transaction` decision, confidence, and `validation_status`. Values such
   as an unseen payment method are rejected before inference.

   **POST `/monitor/drift`** accepts 30–2,000 recent transactions and compares
   their numerical feature distributions to the training baseline with PSI and KS
   tests. Re-run training once after pulling these changes to create the saved
   drift reference data.

   Each `/predict` response also contains `anomaly_detected` and `anomaly_score`.
   These come from an Isolation Forest trained on normal feature-and-prediction
   patterns; run training again after pulling these changes to create its artifact.

   **POST `/monitor/skew`** compares a declared serving feature list and
   preprocessing version with the contract saved during training. It reports
   missing, unexpected, reordered, or version-mismatched features.

   **POST `/monitor/health`** is the unified health engine. Provide the latest
   validation, drift, anomaly, and skew findings to receive a 0–100 health score,
   severity level, classified failure types, and recommended actions.

## Drift experiment

After training, run the reproducible experiment below to simulate a shifted
production customer segment. It reports PSI/KS drift and saves the evidence in
`logs/drift_experiment_result.json`.

```powershell
.\.venv\Scripts\python.exe scripts\run_drift_experiment.py
```

6. In another terminal, run the dashboard placeholder:

   ```powershell
   streamlit run dashboard/app.py
   ```

## Project layout

| Folder / file | Purpose |
| --- | --- |
| `api/` | FastAPI prediction and health endpoints |
| `monitoring/` | Interfaces for validation, drift, anomaly, skew, explanation, and health scoring |
| `dashboard/` | Streamlit monitoring dashboard |
| `config/` | Central settings and thresholds |
| `utils/` | Shared logging and helper utilities |
| `data/` | Raw and processed datasets (not committed) |
| `models/` | Trained model artifacts (not committed) |
| `tests/` | Automated tests |
| `main.py` | Local development entry point |

## Incremental implementation order

1. Select and prepare a dataset.
2. Train and save a baseline classification model.
3. Implement request schemas and `/predict`.
4. Add feature validation.
5. Add PSI and KS drift monitoring.
6. Add prediction anomaly and training-serving skew checks.
7. Add SHAP explanations, unified health scoring, dashboard, and alerts.

## Monitoring history and explanations

Flagged fraud alerts and prediction anomalies return top SHAP feature contributions.
Successful predictions and health assessments are stored locally in
`database/monitoring_events.sqlite3` and displayed in the dashboard history.

## MLflow experiments and model registry

Install the updated project dependencies, then run three tracked experiments:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_mlflow_experiments.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///database/mlflow_tracking.db --default-artifact-root .\mlartifacts
```

Open `http://127.0.0.1:5000` to compare runs in **Fraud Detection Experiments v4**. Each run logs parameters,
accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC, the model, training
metrics, and monitoring artifacts. The final run registers `FraudDetectionModel`
and writes its version to `models/model_manifest.json`; the API uses it when
available and falls back to `models/fraud_model.joblib` if registry loading fails.

## Controlled failure experiments

```powershell
.\.venv\Scripts\python.exe scripts\run_failure_experiments.py
```

This produces Normal, Data Drift, Prediction Anomaly, Training-Serving Skew, and
Combined-failure results in `logs/controlled_failure_experiments.json`.

## License

Add a license before publishing the repository.
