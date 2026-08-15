"""SQLite persistence layer for the 365Dfarms Mulberry AI Streamlit app.

Ported from the Flutter app's `database_helper.dart` schema so the two apps
stay data-compatible. Every table keeps the original `sync_status` column
even though this web build has no cloud sync target yet, for schema parity.

Note for Streamlit Community Cloud deployments: the filesystem is ephemeral
(reset on redeploy / app sleep), so this SQLite file is a demo/local data
store, not durable storage. See README.md for swapping in a hosted DB.
"""
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mulberry_ai.db"
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def save_uploaded_image(image_bytes: bytes, prefix: str = "scan") -> str:
    """Saves image bytes under data/uploads and returns the relative path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{new_id()}.jpg"
    path = UPLOAD_DIR / filename
    path.write_bytes(image_bytes)
    return f"uploads/{filename}"

SCHEMA = {
    "user_profile": """
        CREATE TABLE IF NOT EXISTS user_profile (
            id TEXT PRIMARY KEY, full_name TEXT, mobile_number TEXT, email TEXT,
            password_hash TEXT, preferred_language TEXT, farmer_type TEXT,
            village TEXT, taluka TEXT, district TEXT, state TEXT,
            farm_size_acres REAL, number_of_plots INTEGER,
            mulberry_variety TEXT, irrigation_type TEXT,
            is_organic_preferred INTEGER, is_guest INTEGER,
            photo_training_consent INTEGER, created_at TEXT, sync_status TEXT,
            last_login_at TEXT, is_active INTEGER DEFAULT 1
        )
    """,
    "farm_plot": """
        CREATE TABLE IF NOT EXISTS farm_plot (
            id TEXT PRIMARY KEY, owner_id TEXT, name TEXT, area_acres REAL,
            mulberry_variety TEXT, plantation_date TEXT, irrigation_type TEXT,
            soil_type TEXT, latitude REAL, longitude REAL, notes TEXT,
            photo_paths TEXT, health_score INTEGER, created_at TEXT, sync_status TEXT
        )
    """,
    "scan_record": """
        CREATE TABLE IF NOT EXISTS scan_record (
            id TEXT PRIMARY KEY, owner_id TEXT, plot_id TEXT, scan_type TEXT,
            image_path TEXT, captured_at TEXT, result_label_key TEXT,
            display_name TEXT, confidence REAL, severity_percent REAL, risk_level TEXT,
            model_version TEXT, model_accuracy REAL, used_cloud_fallback INTEGER,
            sync_status TEXT
        )
    """,
    "spray_log": """
        CREATE TABLE IF NOT EXISTS spray_log (
            id TEXT PRIMARY KEY, owner_id TEXT, plot_id TEXT, date TEXT,
            problem_treated TEXT, product_used TEXT, is_organic INTEGER,
            dosage TEXT, water_quantity_litres REAL, area_covered_acres REAL,
            labor_cost REAL, notes TEXT, follow_up_date TEXT, sync_status TEXT
        )
    """,
    "fertilizer_log": """
        CREATE TABLE IF NOT EXISTS fertilizer_log (
            id TEXT PRIMARY KEY, owner_id TEXT, plot_id TEXT, date TEXT,
            fertilizer_name TEXT, quantity_kg REAL, is_organic INTEGER,
            notes TEXT, sync_status TEXT
        )
    """,
    "soil_test": """
        CREATE TABLE IF NOT EXISTS soil_test (
            id TEXT PRIMARY KEY, owner_id TEXT, plot_id TEXT, test_date TEXT,
            ph REAL, ec REAL, nitrogen_level TEXT, phosphorus_level TEXT,
            potassium_level TEXT, organic_carbon_percent REAL, sync_status TEXT
        )
    """,
    "crop_task": """
        CREATE TABLE IF NOT EXISTS crop_task (
            id TEXT PRIMARY KEY, owner_id TEXT, plot_id TEXT, title TEXT,
            category TEXT, due_date TEXT, is_completed INTEGER, notes TEXT,
            sync_status TEXT
        )
    """,
    "production_record": """
        CREATE TABLE IF NOT EXISTS production_record (
            id TEXT PRIMARY KEY, owner_id TEXT, plot_id TEXT, date TEXT,
            leaf_harvest_kg REAL, cocoon_production_kg REAL,
            sale_quantity_kg REAL, sale_rate_per_kg REAL, buyer TEXT,
            expenses REAL, sync_status TEXT
        )
    """,
    "expert_case": """
        CREATE TABLE IF NOT EXISTS expert_case (
            id TEXT PRIMARY KEY, owner_id TEXT, scan_record_id TEXT,
            description TEXT, crop_age_months INTEGER,
            location TEXT, status TEXT, expert_reply TEXT, created_at TEXT,
            sync_status TEXT
        )
    """,
    "notification_item": """
        CREATE TABLE IF NOT EXISTS notification_item (
            id TEXT PRIMARY KEY, owner_id TEXT, type TEXT, title TEXT,
            body TEXT, created_at TEXT, is_read INTEGER, related_record_id TEXT
        )
    """,
}

TABLES = list(SCHEMA.keys())


@st.cache_resource(show_spinner=False)
def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    with conn:
        for ddl in SCHEMA.values():
            conn.execute(ddl)
    _migrate()


def _migrate() -> None:
    """Adds columns introduced after a table's first CREATE TABLE ran, for
    DBs created before this version - CREATE TABLE IF NOT EXISTS is a no-op
    on an already-existing table, so new columns need an explicit ALTER."""
    conn = get_connection()
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(user_profile)")}
    with conn:
        if "last_login_at" not in existing:
            conn.execute("ALTER TABLE user_profile ADD COLUMN last_login_at TEXT")
        if "is_active" not in existing:
            conn.execute("ALTER TABLE user_profile ADD COLUMN is_active INTEGER DEFAULT 1")
    _COLUMN_CACHE.pop("user_profile", None)


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_row(table: str, row: dict) -> str:
    """Inserts `row` into `table`, filling id/created_at/sync_status if absent."""
    row = dict(row)
    row.setdefault("id", new_id())
    if "created_at" in _columns(table):
        row.setdefault("created_at", now_iso())
    if "sync_status" in _columns(table):
        row.setdefault("sync_status", "pending")
    cols = list(row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    conn = get_connection()
    with conn:
        conn.execute(sql, [row[c] for c in cols])
    return row["id"]


def update_row(table: str, row_id: str, fields: dict) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    sql = f"UPDATE {table} SET {set_clause} WHERE id = ?"
    conn = get_connection()
    with conn:
        conn.execute(sql, [*fields.values(), row_id])


def delete_row(table: str, row_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))


def fetch_all(table: str, where: str = "", params: tuple = (), order_by: str = "") -> list[dict]:
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    conn = get_connection()
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def fetch_one(table: str, where: str, params: tuple = ()) -> dict | None:
    rows = fetch_all(table, where, params)
    return rows[0] if rows else None


def count(table: str, where: str = "", params: tuple = ()) -> int:
    sql = f"SELECT COUNT(*) AS c FROM {table}"
    if where:
        sql += f" WHERE {where}"
    conn = get_connection()
    cur = conn.execute(sql, params)
    return cur.fetchone()["c"]


_COLUMN_CACHE: dict[str, list[str]] = {}


def _columns(table: str) -> list[str]:
    if table not in _COLUMN_CACHE:
        conn = get_connection()
        cur = conn.execute(f"PRAGMA table_info({table})")
        _COLUMN_CACHE[table] = [r["name"] for r in cur.fetchall()]
    return _COLUMN_CACHE[table]


def wipe_all_tables() -> None:
    conn = get_connection()
    with conn:
        for table in TABLES:
            conn.execute(f"DELETE FROM {table}")
