"""Persistence layer for the 365Dfarms Mulberry AI Streamlit app.

Dual-backend, same principle as ai/weather.py and ai/chatbot.py's
fallbacks: with no [supabase] secret configured, this runs on a local
SQLite file (zero setup, good for dev/demo - but ephemeral on Streamlit
Community Cloud, wiped on redeploy/sleep). Once `db_url` is set in
secrets, every call here transparently goes to a Supabase/Postgres
database instead - durable, and the only file that needed to change (see
README.md "Migrating to Supabase (Postgres)").

Schema ported from the Flutter app's `database_helper.dart` so the two
apps stay data-compatible. Every table keeps the original `sync_status`
column even though this web build has no cloud sync target yet, for
schema parity.
"""
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mulberry_ai.db"
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"

# Streamlit runs each script execution on its own thread, and st.cache_resource
# shares ONE connection across all of them. For SQLite that flag only disables
# Python's same-thread safety check - it does not make the connection itself
# safe for concurrent use (this genuinely caused "cannot commit - no
# transaction is active" once, from a rerun racing session_cookie.py's retry-
# rerun). Postgres connections have the same single-connection-shared-across-
# threads hazard. This lock serializes all access to prevent either.
_db_lock = threading.Lock()


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
            body TEXT, created_at TEXT, is_read INTEGER, related_record_id TEXT,
            category TEXT, priority TEXT, status TEXT,
            action_taken TEXT, outcome TEXT, resolved_at TEXT
        )
    """,
    "user_session": """
        CREATE TABLE IF NOT EXISTS user_session (
            id TEXT PRIMARY KEY, user_id TEXT, created_at TEXT, expires_at TEXT
        )
    """,
    "silkworm_batch": """
        CREATE TABLE IF NOT EXISTS silkworm_batch (
            id TEXT PRIMARY KEY, owner_id TEXT, plot_id TEXT, batch_name TEXT,
            instar_stage TEXT, mortality_percent REAL, last_fed_at TEXT,
            created_at TEXT, updated_at TEXT, notes TEXT
        )
    """,
}

TABLES = list(SCHEMA.keys())


def _postgres_url() -> str | None:
    cfg = st.secrets.get("supabase", {})
    return cfg.get("db_url") or os.environ.get("SUPABASE_DB_URL")


def _backend() -> str:
    return "postgres" if _postgres_url() else "sqlite"


def active_backend() -> str:
    """'postgres' (Supabase) or 'sqlite' (local file) - whichever is live."""
    return _backend()


@st.cache_resource(show_spinner=False)
def get_connection():
    if _backend() == "postgres":
        import psycopg2
        import psycopg2.extras

        return psycopg2.connect(_postgres_url(), cursor_factory=psycopg2.extras.RealDictCursor)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _exec(conn, sql: str, params=()):
    """Runs sql/params on either backend and returns a cursor - sqlite3's
    Connection.execute() is a convenience method Postgres connections
    don't have, and Postgres uses %s placeholders instead of sqlite's ?,
    so this is the one seam every call in this module goes through."""
    if _backend() == "postgres":
        cur = conn.cursor()
        cur.execute(sql.replace("?", "%s"), params)
        return cur
    return conn.execute(sql, params)


def init_db() -> None:
    conn = get_connection()
    with _db_lock, conn:
        for ddl in SCHEMA.values():
            _exec(conn, ddl)
    _migrate()
    with _db_lock, conn:
        _exec(conn, "DELETE FROM user_session WHERE expires_at < ?", (now_iso(),))


def _live_columns(conn, table: str) -> set[str]:
    """Uncached column lookup used only during _migrate(), before
    _COLUMN_CACHE has necessarily been populated."""
    if _backend() == "postgres":
        cur = conn.cursor()
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table,))
        return {r["column_name"] for r in cur.fetchall()}
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def _migrate() -> None:
    """Adds columns introduced after a table's first CREATE TABLE ran, for
    DBs created before this version - CREATE TABLE IF NOT EXISTS is a no-op
    on an already-existing table, so new columns need an explicit ALTER."""
    conn = get_connection()
    with _db_lock:
        existing_profile = _live_columns(conn, "user_profile")
        existing_notification = _live_columns(conn, "notification_item")
        with conn:
            if "last_login_at" not in existing_profile:
                _exec(conn, "ALTER TABLE user_profile ADD COLUMN last_login_at TEXT")
            if "is_active" not in existing_profile:
                _exec(conn, "ALTER TABLE user_profile ADD COLUMN is_active INTEGER DEFAULT 1")
            for col, coltype in (
                ("category", "TEXT"), ("priority", "TEXT"), ("status", "TEXT"),
                ("action_taken", "TEXT"), ("outcome", "TEXT"), ("resolved_at", "TEXT"),
            ):
                if col not in existing_notification:
                    _exec(conn, f"ALTER TABLE notification_item ADD COLUMN {col} {coltype}")
    _COLUMN_CACHE.pop("user_profile", None)
    _COLUMN_CACHE.pop("notification_item", None)


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
    with _db_lock, conn:
        _exec(conn, sql, [row[c] for c in cols])
    return row["id"]


def update_row(table: str, row_id: str, fields: dict) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    sql = f"UPDATE {table} SET {set_clause} WHERE id = ?"
    conn = get_connection()
    with _db_lock, conn:
        _exec(conn, sql, [*fields.values(), row_id])


def delete_row(table: str, row_id: str) -> None:
    conn = get_connection()
    with _db_lock, conn:
        _exec(conn, f"DELETE FROM {table} WHERE id = ?", (row_id,))


def fetch_all(table: str, where: str = "", params: tuple = (), order_by: str = "") -> list[dict]:
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    conn = get_connection()
    with _db_lock:
        cur = _exec(conn, sql, params)
        return [dict(r) for r in cur.fetchall()]


def fetch_one(table: str, where: str, params: tuple = ()) -> dict | None:
    rows = fetch_all(table, where, params)
    return rows[0] if rows else None


def count(table: str, where: str = "", params: tuple = ()) -> int:
    sql = f"SELECT COUNT(*) AS c FROM {table}"
    if where:
        sql += f" WHERE {where}"
    conn = get_connection()
    with _db_lock:
        cur = _exec(conn, sql, params)
        return cur.fetchone()["c"]


_COLUMN_CACHE: dict[str, list[str]] = {}


def _columns(table: str) -> list[str]:
    if table not in _COLUMN_CACHE:
        conn = get_connection()
        with _db_lock:
            _COLUMN_CACHE[table] = sorted(_live_columns(conn, table))
    return _COLUMN_CACHE[table]


def execute_transaction(statements: list[tuple[str, tuple]]) -> None:
    """Runs multiple (sql, params) statements as one locked transaction -
    for multi-table operations like cascading deletes that need to go
    through the same lock as every other write (see _db_lock docstring)."""
    conn = get_connection()
    with _db_lock, conn:
        for sql, params in statements:
            _exec(conn, sql, params)


def wipe_all_tables() -> None:
    conn = get_connection()
    with _db_lock, conn:
        for table in TABLES:
            _exec(conn, f"DELETE FROM {table}")
