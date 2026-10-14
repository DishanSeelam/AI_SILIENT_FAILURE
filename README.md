# AI Silent Failure Detection

Synthetic e-commerce fraud detection with multi-model evaluation and integrated monitoring.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\generate_ecommerce_data.py --rows 12000
.\.venv\Scripts\python.exe scripts\run_mlflow_experiments.py
.\.venv\Scripts\python.exe scripts\export_model_comparison.py
.\.venv\Scripts\python.exe scripts\select_best_model.py
.\.venv\Scripts\python.exe scripts\create_monitoring_batches.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload
```

In separate terminals:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///database/mlflow_tracking.db --port 5000
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open API docs at http://127.0.0.1:8000/docs, MLflow at http://127.0.0.1:5000, and the dashboard at http://localhost:8501.

**Research note:** all transactions are synthetic. Metrics measure this simulated data only and must not be presented as production fraud performance.

## Monitoring demonstrations

```powershell
$body = Get-Content .\logs\drift_batch.json -Raw
Invoke-RestMethod -Uri "http://127.0.0.1:8000/monitor/drift" -Method POST -ContentType "application/json" -Body $body
```

Endpoints: `/predict`, `/monitor/drift`, `/monitor/anomaly`, `/monitor/skew`, `/monitor/health`, `/monitor/assess`, and `/monitor/history`.
