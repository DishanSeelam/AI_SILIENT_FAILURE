"""Streamlit operational dashboard for model and silent-failure monitoring."""
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import METRICS_PATH

st.set_page_config(page_title="AI Silent Failure Detection", layout="wide")
st.title("AI Silent Failure Detection")
st.caption("Synthetic e-commerce fraud model — measurements are for simulated data only.")

metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
if metrics:
    st.subheader("Current local candidate")
    columns = st.columns(5)
    for column, (label, value) in zip(columns, [("Model", metrics["model_type"]), ("F1", metrics["metrics"]["f1"]),
                                                  ("F2", metrics["metrics"]["f2"]), ("Recall", metrics["metrics"]["recall"]),
                                                  ("PR-AUC", metrics["metrics"]["pr_auc"])]):
        column.metric(label, value if label == "Model" else f"{value:.3f}")
else:
    st.warning("No trained candidate found. Run the dataset generator and MLflow experiments.")

comparison_file = ROOT / "logs" / "model_comparison.csv"
if comparison_file.exists():
    st.subheader("MLflow model comparison")
    comparison = pd.read_csv(comparison_file)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(comparison, x="run_name", y=["f1", "f2", "recall"], barmode="group",
                           title="Fraud-model performance (higher is better)"), use_container_width=True)

st.subheader("Controlled Failure Experiments")
failure_file = ROOT / "logs" / "controlled_failure_experiments.json"
if failure_file.exists():
    failures = pd.DataFrame(json.loads(failure_file.read_text()))
    fig = px.bar(failures, x="scenario", y="health_score", color="severity", text="health_score",
                 hover_data=["failure_types", "drift_status", "anomaly_count", "skew_status"], range_y=[0, 105],
                 color_discrete_map={"healthy": "#2ca02c", "warning": "#ffbf00", "critical": "#d62728"})
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(failures, use_container_width=True, hide_index=True)
else:
    st.info("Run `scripts/run_failure_experiments.py` after training to create controlled monitoring readings.")

st.subheader("How to run a real batch assessment")
st.code("POST http://127.0.0.1:8000/monitor/assess\n{ \"transactions\": [ ...at least 10 valid transactions... ] }")
st.markdown("The assessment joins feature validation, numeric/categorical PSI + KS drift, anomaly detection, training-serving skew, and an explainable health score.")
