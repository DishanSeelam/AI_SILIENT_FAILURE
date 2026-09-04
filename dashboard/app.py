"""Interactive Streamlit dashboard for AI Silent Failure Detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from monitoring.health_score import calculate_health_score
from database.event_store import recent_events
from utils.mlflow_tracking import read_model_manifest


METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"
PROFILE_PATH = PROJECT_ROOT / "models" / "reference_profile.json"
DRIFT_RESULT_PATH = PROJECT_ROOT / "logs" / "drift_experiment_result.json"
FAILURE_RESULTS_PATH = PROJECT_ROOT / "logs" / "controlled_failure_experiments.json"


@st.cache_data
def load_json(path: str) -> dict | None:
    """Load a local project JSON artifact when it is available."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def severity_color(severity: str) -> str:
    """Map the health severity to a small, consistent visual label."""
    return {"normal": "green", "warning": "orange", "critical": "red"}[severity]


st.set_page_config(page_title="AI Silent Failure Detection", page_icon="🛡️", layout="wide")
st.title("AI Silent Failure Detection")
st.caption("MLOps monitoring dashboard · e-commerce fraud detection")

metrics = load_json(str(METRICS_PATH))
profile = load_json(str(PROFILE_PATH))
drift_result = load_json(str(DRIFT_RESULT_PATH))
failure_results = load_json(str(FAILURE_RESULTS_PATH))
model_manifest = read_model_manifest()

model_left, model_right, model_third = st.columns(3)
model_left.metric("Active model version", f"v{model_manifest.get('model_version', 'local')}")
model_right.metric("Model lifecycle stage", model_manifest.get("model_stage", "Fallback"))
model_third.caption("MLflow tracks parameters, metrics, artifacts, and registered model versions.")

if metrics is None:
    st.warning("No trained model metrics found. Run `scripts/train_model.py` first.")
else:
    selected = metrics["test_metrics_at_selected_threshold"]
    threshold = metrics["validation_threshold_selection"]["selected_threshold"]
    first, second, third, fourth = st.columns(4)
    first.metric("Fraud recall", f"{selected['recall']:.1%}")
    second.metric("Fraud precision", f"{selected['precision']:.1%}")
    third.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    fourth.metric("Alert threshold", f"{threshold:.2f}")

st.divider()
st.subheader("Unified Health Score")
st.caption("Adjust the latest monitoring findings to assess the current pipeline state.")

left, right = st.columns([1, 1])
with left:
    validation_passed = st.checkbox("Feature validation passed", value=True)
    drift_detected = st.checkbox("Data drift detected", value=False)
    drifted_feature_count = st.number_input(
        "Number of drifted features", min_value=0, max_value=20, value=0, disabled=not drift_detected
    )
with right:
    anomaly_detected = st.checkbox("Prediction anomaly detected", value=False)
    skew_detected = st.checkbox("Training-serving skew detected", value=False)
    st.caption("Health score deductions: validation 30, drift up to 30, anomaly 20, skew 20.")

health = calculate_health_score(
    {
        "validation_passed": validation_passed,
        "drift_detected": drift_detected,
        "drifted_feature_count": int(drifted_feature_count),
        "anomaly_detected": anomaly_detected,
        "skew_detected": skew_detected,
    }
)

score_column, status_column = st.columns([1, 3])
with score_column:
    st.metric("Pipeline health", f"{health['health_score']}/100")
    st.progress(health["health_score"])
with status_column:
    st.markdown(f":{severity_color(health['severity'])}[**{health['severity'].upper()}**]")
    st.write(health["status"].replace("_", " ").title())
    if health["failure_types"]:
        st.write("Detected: " + ", ".join(item.replace("_", " ") for item in health["failure_types"]))
    else:
        st.write("No active silent-failure signals.")

st.subheader("Recommended Actions")
for action in health["recommended_actions"]:
    st.write(f"• {action}")

st.divider()
drift_column, profile_column = st.columns(2)
with drift_column:
    st.subheader("Latest Drift Experiment")
    if drift_result is None:
        st.info("No drift experiment result found. Run `scripts/run_drift_experiment.py` to create one.")
    else:
        if drift_result["status"] == "drift_detected":
            st.error(f"Drift detected in: {', '.join(drift_result['drifted_features'])}")
        else:
            st.success("No drift detected in the latest batch.")
        drift_table = pd.DataFrame.from_dict(drift_result["feature_results"], orient="index")
        st.dataframe(drift_table, use_container_width=True)

with profile_column:
    st.subheader("Training Reference Profile")
    if profile is None:
        st.info("Reference profile will appear after model training.")
    else:
        st.metric("Monitored numerical features", len(profile["numeric_features"]))
        st.metric("Monitored categorical features", len(profile["categorical_features"]))
        st.write("**Categorical values learned during training**")
        for feature, details in profile["categorical_features"].items():
            st.caption(f"{feature}: {', '.join(details['categories'])}")

st.caption("Refresh the page after training or running a monitoring experiment to load the latest artifacts.")

st.divider()
st.subheader("Controlled Failure Experiments")
if failure_results is None:
    st.info("Run `scripts/run_failure_experiments.py` to generate Normal, Drift, Anomaly, Skew, and Combined evidence.")
else:
    failure_rows = [
        {"experiment": name, "expected_failure": value["expected_failure"], "health_score": value["health"]["health_score"], "severity": value["health"]["severity"]}
        for name, value in failure_results["experiments"].items()
    ]
    failure_table = pd.DataFrame(failure_rows)
    failure_table["scenario"] = failure_table["experiment"].map(
        {
            "A_normal": "Normal", "B_data_drift": "Data drift", "C_anomaly": "Anomaly",
            "D_training_serving_skew": "Serving skew", "E_combined": "Combined failures",
        }
    )
    chart = px.bar(
        failure_table, x="scenario", y="health_score", color="severity", text="health_score",
        category_orders={"severity": ["normal", "warning", "critical"]},
        color_discrete_map={"normal": "#2E8B57", "warning": "#E69F00", "critical": "#D1495B"},
        labels={"scenario": "Controlled scenario", "health_score": "Unified health score"},
    )
    chart.update_traces(texttemplate="%{text}/100", textposition="outside", cliponaxis=False)
    chart.update_yaxes(range=[0, 110], dtick=20)
    chart.update_layout(showlegend=True, margin=dict(l=0, r=0, t=15, b=0))
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(failure_table, use_container_width=True)
    st.write("Silent-failure detector metrics", failure_results["silent_failure_detector_metrics"])

st.divider()
st.subheader("Monitoring Event History")
prediction_history, health_history = recent_events()
history_left, history_right = st.columns(2)
with history_left:
    st.caption("Recent prediction events")
    if prediction_history.empty:
        st.info("No prediction events have been logged yet. Use the API /predict endpoint.")
    else:
        st.line_chart(prediction_history.set_index("created_at")["fraud_probability"])
        st.dataframe(prediction_history, use_container_width=True)
with history_right:
    st.caption("Recent unified health assessments")
    if health_history.empty:
        st.info("No health assessments have been logged yet. Use /monitor/health.")
    else:
        st.line_chart(health_history.set_index("created_at")["health_score"])
        st.dataframe(health_history, use_container_width=True)
