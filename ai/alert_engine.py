"""Rule-based Alert Engine: turns scan results, weather risk, and silkworm
rearing data into categorized, prioritized notifications the farmer can
act on and mark resolved.

Transparent thresholds (same philosophy as ai/disease_risk.py - no black
box, an agronomist can review/tune every number here directly), not a
learned model. See core/helpers.py's add_notification() for the underlying
insert and app_pages/notifications.py for the UI that consumes
category/priority/status.
"""
from __future__ import annotations

import datetime as dt

from ai.disease_risk import DiseaseRiskAssessment
from core import db
from core.helpers import add_notification

RISK_TO_PRIORITY = {"severe": "critical", "high": "high", "medium": "medium", "low": "low"}

# Typical instar durations (days) - standard sericulture reference ranges,
# used only to flag "a transition is likely approaching", not as a precise
# prediction (actual duration varies with temperature/humidity/breed).
INSTAR_DURATIONS_DAYS = {"1st": 4, "2nd": 3, "3rd": 4, "4th": 5, "5th": 7}
INSTAR_ORDER = ["1st", "2nd", "3rd", "4th", "5th", "Spinning"]

FEEDING_INTERVAL_HOURS = 7  # ~3x/day typical feeding schedule
MORTALITY_THRESHOLD_EARLY = 3.0  # % - 1st-3rd instar (poster's own example thresholds)
MORTALITY_THRESHOLD_LATE = 2.0  # % - 4th instar onward


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse(ts: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def generate_disease_alert(owner_id: str, label_name: str, risk_level: str, severity_percent: float, source_label: str) -> None:
    """Called after any scan (mulberry or silkworm) with a non-healthy
    result. Low-risk scans aren't alert-worthy on their own - they're just
    visible in scan history."""
    priority = RISK_TO_PRIORITY.get(risk_level, "low")
    if priority == "low":
        return
    add_notification(
        owner_id, "scan_alert", f"{priority.title()} risk: {label_name}",
        f"A {source_label} scan detected {label_name} at {severity_percent:.0f}% severity.",
        category="disease", priority=priority,
    )


def generate_climate_alerts(owner_id: str, assessment: DiseaseRiskAssessment) -> None:
    """Once-per-day (dedup'd) weather-risk alert, sourced from the existing
    ai/disease_risk.py rules - no new weather logic, just surfaces "high"
    results as a real actionable alert instead of only a dashboard widget."""
    today = dt.date.today().isoformat()
    dedup_marker = f"climate::{today}"
    if db.fetch_one("notification_item", "owner_id = ? AND related_record_id = ?", (owner_id, dedup_marker)):
        return

    risky = []
    if assessment.powdery_mildew_risk == "high":
        risky.append("Powdery mildew risk is high today")
    if assessment.leaf_rust_risk == "high":
        risky.append("Leaf rust risk is high today")
    if assessment.fungal_disease_risk == "high":
        risky.append("Fungal disease risk is high today")
    if assessment.pest_outbreak_risk == "high":
        risky.append("Pest outbreak risk is high today")
    if assessment.heat_stress_warning:
        risky.append("Heat stress warning: temperature ≥ 38°C today")
    if not risky:
        return

    add_notification(
        owner_id, "climate_alert", "High disease/pest risk today",
        " · ".join(risky) + f". {assessment.best_spray_window_note}",
        related_record_id=dedup_marker, category="climate", priority="high",
    )


def generate_feeding_reminder(owner_id: str, batch: dict) -> None:
    if not batch.get("last_fed_at"):
        return
    hours_since = (_now() - _parse(batch["last_fed_at"])).total_seconds() / 3600
    if hours_since < FEEDING_INTERVAL_HOURS:
        return
    today = dt.date.today().isoformat()
    dedup_marker = f"feeding::{batch['id']}::{today}"
    if db.fetch_one("notification_item", "related_record_id = ?", (dedup_marker,)):
        return
    priority = "high" if hours_since >= FEEDING_INTERVAL_HOURS * 2 else "medium"
    add_notification(
        owner_id, "feeding_reminder", f"Feeding due — {batch.get('batch_name') or 'batch'}",
        f"Last fed {hours_since:.0f}h ago ({batch.get('instar_stage', '')} instar). Provide fresh mulberry leaves.",
        related_record_id=dedup_marker, category="feeding", priority=priority,
    )


def generate_instar_reminder(owner_id: str, batch: dict) -> None:
    stage = batch.get("instar_stage")
    if stage not in INSTAR_DURATIONS_DAYS or not batch.get("updated_at"):
        return
    days_in_stage = (_now() - _parse(batch["updated_at"])).total_seconds() / 86400
    typical_days = INSTAR_DURATIONS_DAYS[stage]
    if days_in_stage < typical_days - 1:
        return  # not close to a transition yet

    today = dt.date.today().isoformat()
    dedup_marker = f"instar::{batch['id']}::{today}"
    if db.fetch_one("notification_item", "related_record_id = ?", (dedup_marker,)):
        return

    next_stage = INSTAR_ORDER[INSTAR_ORDER.index(stage) + 1] if stage in INSTAR_ORDER[:-1] else "spinning"
    add_notification(
        owner_id, "instar_reminder", f"Molting expected soon — {batch.get('batch_name') or 'batch'}",
        f"{stage} instar typically lasts ~{typical_days} days. Prepare for transition to {next_stage} and bed cleaning.",
        related_record_id=dedup_marker, category="instar", priority="medium",
    )


def generate_mortality_alert(owner_id: str, batch: dict) -> None:
    mortality = batch.get("mortality_percent")
    if mortality is None:
        return
    stage = batch.get("instar_stage") or ""
    threshold = MORTALITY_THRESHOLD_EARLY if stage in ("1st", "2nd", "3rd") else MORTALITY_THRESHOLD_LATE
    if mortality < threshold:
        return

    # Don't re-alert while an earlier mortality alert for this batch is
    # still open - resolving/skipping it (Notifications page) clears the
    # way for a fresh one on the next check.
    dedup_marker = f"mortality::{batch['id']}"
    existing_open = db.fetch_one(
        "notification_item",
        "related_record_id = ? AND (status IS NULL OR status = 'open')",
        (dedup_marker,),
    )
    if existing_open:
        return

    priority = "critical" if mortality >= threshold * 2 else "high"
    add_notification(
        owner_id, "mortality_alert", f"High mortality — {batch.get('batch_name') or 'batch'}",
        f"Mortality at {mortality:.1f}% ({stage or 'unknown'} instar, threshold {threshold}%). "
        f"Check disease, feed quality and environment.",
        related_record_id=dedup_marker, category="mortality", priority=priority,
    )


def run_batch_checks(owner_id: str, batch: dict) -> None:
    """Runs all rearing-data-driven checks for one batch - call after
    logging/updating a batch, or once per dashboard/tracking-page load."""
    generate_feeding_reminder(owner_id, batch)
    generate_instar_reminder(owner_id, batch)
    generate_mortality_alert(owner_id, batch)
