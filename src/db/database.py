"""SQLite database initialization and query helpers."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "basestation.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS diagnosis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    mode TEXT NOT NULL,
    sample_index INTEGER,
    sample_type TEXT,
    anomaly_type TEXT,
    user_query TEXT,
    has_anomaly INTEGER,
    anomaly_score REAL,
    affected_kpis TEXT,
    root_cause TEXT,
    confidence REAL,
    reasoning TEXT,
    topology_evidence TEXT,
    report_markdown TEXT,
    eval_passed INTEGER,
    latency_ms REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    started_at TEXT DEFAULT (datetime('now')),
    ended_at TEXT,
    interval_sec REAL DEFAULT 15.0,
    total_checks INTEGER DEFAULT 0,
    anomalies_found INTEGER DEFAULT 0,
    reports_generated INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);
"""


async def init_db(db_path: str | Path | None = None):
    """Initialize the SQLite database and create tables if they don't exist."""
    path = str(db_path or DB_PATH)
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA)
        await db.commit()
    return path


async def insert_diagnosis(run_data: dict[str, Any], db_path: str | None = None):
    """Insert a diagnosis run record."""
    path = db_path or str(DB_PATH)
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """INSERT INTO diagnosis_runs
               (run_id, mode, sample_index, sample_type, anomaly_type,
                user_query, has_anomaly, anomaly_score, affected_kpis,
                root_cause, confidence, reasoning, topology_evidence,
                report_markdown, eval_passed, latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_data["run_id"],
                run_data.get("mode", "oneshot"),
                run_data.get("sample_index"),
                run_data.get("sample_type"),
                run_data.get("anomaly_type"),
                run_data.get("user_query"),
                int(run_data.get("has_anomaly", False)),
                run_data.get("anomaly_score"),
                json.dumps(run_data.get("affected_kpis", []), ensure_ascii=False),
                run_data.get("root_cause"),
                run_data.get("confidence"),
                run_data.get("reasoning"),
                json.dumps(run_data.get("topology_evidence", []), ensure_ascii=False),
                run_data.get("report_markdown"),
                int(run_data.get("eval_passed", False)),
                run_data.get("latency_ms"),
            ),
        )
        await db.commit()


async def insert_session(session_data: dict[str, Any], db_path: str | None = None):
    """Insert a monitoring session record."""
    path = db_path or str(DB_PATH)
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """INSERT INTO monitoring_sessions
               (session_id, interval_sec, status)
               VALUES (?,?,?)""",
            (
                session_data["session_id"],
                session_data.get("interval_sec", 15.0),
                "running",
            ),
        )
        await db.commit()


async def update_session(session_id: str, updates: dict, db_path: str | None = None):
    """Update a monitoring session."""
    path = db_path or str(DB_PATH)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [session_id]
    async with aiosqlite.connect(path) as db:
        await db.execute(
            f"UPDATE monitoring_sessions SET {set_clause} WHERE session_id=?",
            values,
        )
        await db.commit()


async def get_history(limit: int = 20, db_path: str | None = None) -> list[dict]:
    """Get recent diagnosis runs."""
    path = db_path or str(DB_PATH)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM diagnosis_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_run(run_id: str, db_path: str | None = None) -> dict | None:
    """Get a single diagnosis run by ID."""
    path = db_path or str(DB_PATH)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM diagnosis_runs WHERE run_id=?",
            (run_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_sessions(limit: int = 10, db_path: str | None = None) -> list[dict]:
    """Get recent monitoring sessions."""
    path = db_path or str(DB_PATH)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM monitoring_sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
