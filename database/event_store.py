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


def log_prediction(result: dict) -> None:
    """Persist non-sensitive monitoring results from a successful prediction."""
    initialize_database()
    with _connection() as connection:
        connection.execute(
            """INSERT INTO prediction_events
               (created_at, fraud_probability, decision, anomaly_detected, anomaly_score, explanation_generated)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), result["fraud_probability"], result["decision"], int(result["anomaly_detected"]), result["anomaly_score"], int(result["explanation_generated"])),
        )


def log_health(result: dict) -> None:
    """Persist each explicit unified health assessment."""
    initialize_database()
    with _connection() as connection:
        connection.execute(
            """INSERT INTO health_events (created_at, health_score, status, severity)
               VALUES (?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), result["health_score"], result["status"], result["severity"]),
        )


def recent_events(limit: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return recent prediction and health history for the Streamlit dashboard."""
    initialize_database()
    with _connection() as connection:
        predictions = pd.read_sql_query("SELECT * FROM prediction_events ORDER BY id DESC LIMIT ?", connection, params=(limit,))
        health = pd.read_sql_query("SELECT * FROM health_events ORDER BY id DESC LIMIT ?", connection, params=(limit,))
    return predictions, health
