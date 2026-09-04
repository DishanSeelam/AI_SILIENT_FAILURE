"""Lightweight SQLite persistence for monitoring and prediction events."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from config.settings import PROJECT_ROOT


DATABASE_PATH = PROJECT_ROOT / "database" / "monitoring_events.sqlite3"


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """Create required tables without deleting existing event history."""
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                decision TEXT NOT NULL,
                anomaly_detected INTEGER NOT NULL,
                anomaly_score REAL NOT NULL,
                explanation_generated INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                health_score INTEGER NOT NULL,
                status TEXT NOT NULL,
                severity TEXT NOT NULL
            );
            """
        )
        for column, definition in [
            ("model_version", "TEXT"), ("model_stage", "TEXT"), ("validation_status", "TEXT"),
            ("drift_status", "TEXT"), ("skew_status", "TEXT"), ("health_score", "INTEGER"),
            ("failure_type", "TEXT"), ("severity", "TEXT"),
        ]:
            existing = {row[1] for row in connection.execute("PRAGMA table_info(prediction_events)")}
            if column not in existing:
                connection.execute(f"ALTER TABLE prediction_events ADD COLUMN {column} {definition}")
        for column, definition in [("validation_status", "TEXT"), ("drift_status", "TEXT"), ("skew_status", "TEXT"), ("anomaly_score", "REAL"), ("failure_type", "TEXT")]:
            existing = {row[1] for row in connection.execute("PRAGMA table_info(health_events)")}
            if column not in existing:
                connection.execute(f"ALTER TABLE health_events ADD COLUMN {column} {definition}")


def log_prediction(result: dict) -> None:
    """Persist non-sensitive monitoring results from a successful prediction."""
    initialize_database()
    with _connection() as connection:
        connection.execute(
            """INSERT INTO prediction_events
               (created_at, fraud_probability, decision, anomaly_detected, anomaly_score, explanation_generated,
                model_version, model_stage, validation_status, drift_status, skew_status, health_score, failure_type, severity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), result["fraud_probability"], result["decision"], int(result["anomaly_detected"]), result["anomaly_score"], int(result["explanation_generated"]), result.get("model_version"), result.get("model_stage"), result.get("validation_status"), None, None, None, "prediction_anomaly" if result["anomaly_detected"] else None, "warning" if result["anomaly_detected"] else "normal"),
        )


def log_health(result: dict, signals: dict) -> None:
    """Persist each explicit unified health assessment."""
    initialize_database()
    with _connection() as connection:
        connection.execute(
            """INSERT INTO health_events (created_at, health_score, status, severity, validation_status, drift_status, skew_status, anomaly_score, failure_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), result["health_score"], result["status"], result["severity"], "passed" if signals["validation_passed"] else "failed", "drift_detected" if signals["drift_detected"] else "stable", "skew_detected" if signals["skew_detected"] else "consistent", -1.0 if signals["anomaly_detected"] else 0.0, ",".join(result["failure_types"]) or None),
        )


def recent_events(limit: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return recent prediction and health history for the Streamlit dashboard."""
    initialize_database()
    with _connection() as connection:
        predictions = pd.read_sql_query("SELECT * FROM prediction_events ORDER BY id DESC LIMIT ?", connection, params=(limit,))
        health = pd.read_sql_query("SELECT * FROM health_events ORDER BY id DESC LIMIT ?", connection, params=(limit,))
    return predictions, health
