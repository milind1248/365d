"""Small shared UI/formatting helpers used across app_pages/*."""
from __future__ import annotations

import datetime as dt

from core import db

RISK_COLORS = {"low": "green", "medium": "orange", "high": "red", "severe": "red"}
RISK_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "severe": "Severe"}

PRIORITY_COLORS = {"critical": "red", "high": "orange", "medium": "blue", "low": "gray"}
PRIORITY_ICONS = {"critical": "🔴", "high": "🟠", "medium": "🔵", "low": "⚪"}
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def risk_badge(risk: str) -> str:
    color = RISK_COLORS.get(risk, "gray")
    label = RISK_LABELS.get(risk, risk.title())
    return f":{color}[**{label}**]"


def priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, "gray")
    icon = PRIORITY_ICONS.get(priority, "⚪")
    label = (priority or "low").title()
    return f"{icon} :{color}[**{label}**]"


def confidence_badge(tier: str) -> str:
    mapping = {"likely": (":green[Likely]"), "possible": (":orange[Possible]"), "uncertain": (":red[Uncertain]")}
    return mapping.get(tier, tier)


def live_data_caption(source: str, fetched_at: dt.datetime, source_name: str = "Open-Meteo") -> str:
    """Renders a small, honest freshness indicator for weather-driven widgets:
    shows the real current date/time and whether the data behind it is a
    live API response or the offline mock fallback - never just implies
    "live" without checking."""
    stamp = fetched_at.strftime("%d %b %Y, %H:%M")
    if source == "live":
        return f":green[●] Live · {source_name} · as of {stamp}"
    return f":orange[●] Offline fallback (mock forecast) · as of {stamp}"


def plot_options(owner_id: str) -> dict[str, str]:
    """Returns {plot_id: display_name} for the owner's plots, newest first."""
    plots = db.fetch_all("farm_plot", "owner_id = ?", (owner_id,), order_by="created_at DESC")
    return {p["id"]: f"{p['name']} ({p['area_acres']} acres)" for p in plots}


def today_iso() -> str:
    return dt.date.today().isoformat()


def parse_date(value: str, default: dt.date | None = None) -> dt.date:
    if not value:
        return default or dt.date.today()
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return default or dt.date.today()


def add_notification(
    owner_id: str, ntype: str, title: str, body: str, related_record_id: str = "",
    category: str = "general", priority: str = "low",
) -> None:
    db.insert_row(
        "notification_item",
        {
            "owner_id": owner_id,
            "type": ntype,
            "title": title,
            "body": body,
            "created_at": db.now_iso(),
            "is_read": 0,
            "related_record_id": related_record_id,
            "category": category,
            "priority": priority,
            "status": "open",
        },
    )
