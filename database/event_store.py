"""Small SQLite audit log for predictions and monitoring assessments."""
import json
import sqlite3
from datetime import datetime, timezone
from config.settings import ROOT_DIR

DB_PATH = ROOT_DIR / "database" / "monitoring_events.sqlite3"


def initialise():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS monitoring_events (
            id INTEGER PRIMARY KEY, timestamp TEXT, event_type TEXT, payload TEXT)""")


def log_event(event_type, payload):
    initialise()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("INSERT INTO monitoring_events(timestamp,event_type,payload) VALUES(?,?,?)",
                           (datetime.now(timezone.utc).isoformat(), event_type, json.dumps(payload)))


def recent(limit=25):
    initialise()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute("SELECT timestamp,event_type,payload FROM monitoring_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"timestamp": row[0], "event_type": row[1], "payload": json.loads(row[2])} for row in rows]
