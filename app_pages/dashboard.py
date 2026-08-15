import datetime as dt

import streamlit as st

from ai import alert_engine
from ai.disease_risk import assess_day
from ai.weather import get_forecast_with_meta
from core import auth, db
from core.helpers import live_data_caption, priority_badge, risk_badge

user = auth.require_login()
owner_id = user["id"]

st.title("🏠 Dashboard")
st.caption(f"Welcome back, {user['full_name'] or 'Farmer'}. Here's how your mulberry crop looks today.")

plots = db.fetch_all("farm_plot", "owner_id = ?", (owner_id,))
tasks = db.fetch_all("crop_task", "owner_id = ? AND is_completed = 0", (owner_id,))
scans = db.fetch_all("scan_record", "owner_id = ?", (owner_id,), order_by="captured_at DESC")
expert_cases_open = db.count("expert_case", "owner_id = ? AND status != 'closed'", (owner_id,))

for batch in db.fetch_all("silkworm_batch", "owner_id = ?", (owner_id,)):
    alert_engine.run_batch_checks(owner_id, batch)

open_alerts = db.fetch_all("notification_item", "owner_id = ? AND (status IS NULL OR status = 'open')", (owner_id,))
if open_alerts:
    critical = sum(1 for a in open_alerts if a.get("priority") == "critical")
    high = sum(1 for a in open_alerts if a.get("priority") == "high")
    medium = sum(1 for a in open_alerts if a.get("priority") == "medium")
    low = sum(1 for a in open_alerts if (a.get("priority") or "low") == "low")
    with st.container(border=True):
        st.markdown(f"**🔔 Today's alerts** — {len(open_alerts)} open")
        s1, s2, s3, s4, s5 = st.columns([1, 1, 1, 1, 1.4])
        s1.markdown(f"{priority_badge('critical')} {critical}")
        s2.markdown(f"{priority_badge('high')} {high}")
        s3.markdown(f"{priority_badge('medium')} {medium}")
        s4.markdown(f"{priority_badge('low')} {low}")
        s5.page_link("app_pages/notifications.py", label="View all alerts", icon=":material/notifications:")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Farm plots", len(plots))
col2.metric("Pending tasks", len(tasks))
col3.metric("Scans logged", len(scans))
col4.metric("Open expert cases", expert_cases_open)

st.divider()

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.subheader("Recent scans")
    if not scans:
        st.info("No scans yet. Head to **Scan Disease** to run your first AI leaf check.")
    else:
        for scan in scans[:5]:
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.markdown(f"**{scan['display_name'] or scan['result_label_key']}**")
                c1.caption(scan["captured_at"][:16].replace("T", " "))
                c2.metric("Confidence", f"{scan['confidence'] * 100:.0f}%", label_visibility="collapsed")
                c3.markdown(risk_badge(scan["risk_level"] or "low"))

    st.subheader("Upcoming tasks")
    if not tasks:
        st.info("No pending tasks. Add one in **Crop Calendar**.")
    else:
        for task in sorted(tasks, key=lambda t: t["due_date"] or "9999")[:5]:
            overdue = task["due_date"] and task["due_date"] < dt.date.today().isoformat()
            marker = "🔴" if overdue else "🟢"
            st.markdown(f"{marker} **{task['title']}** — due {task['due_date'] or 'no date'} ({task['category']})")

with right:
    st.subheader("Today's disease risk")
    if plots:
        lat = plots[0]["latitude"] or 19.07
        lon = plots[0]["longitude"] or 74.74
    else:
        lat, lon = 19.07, 74.74

    fc = get_forecast_with_meta(lat, lon)
    assessment = assess_day(fc.records[0])
    alert_engine.generate_climate_alerts(owner_id, assessment)
    with st.container(border=True):
        st.caption(live_data_caption(fc.source, fc.fetched_at))
        st.markdown(f"Overall risk today: {risk_badge(assessment.overall_risk)}")
        st.write(f"🍄 Powdery mildew: {risk_badge(assessment.powdery_mildew_risk)}")
        st.write(f"🟠 Leaf rust: {risk_badge(assessment.leaf_rust_risk)}")
        st.write(f"🦠 Fungal disease: {risk_badge(assessment.fungal_disease_risk)}")
        st.write(f"🐛 Pest outbreak: {risk_badge(assessment.pest_outbreak_risk)}")
        st.caption(assessment.best_spray_window_note)
        if assessment.heat_stress_warning:
            st.warning("Heat stress warning: temperature ≥ 38°C today.")

    st.subheader("Quick actions")
    b1, b2 = st.columns(2)
    b1.page_link("app_pages/scan_disease.py", label="Scan a leaf", icon=":material/photo_camera:")
    b2.page_link("app_pages/my_farm.py", label="Manage plots", icon=":material/grass:")
    b1.page_link("app_pages/logs.py", label="Log spray/fertilizer", icon=":material/science:")
    b2.page_link("app_pages/expert_help.py", label="Ask an expert", icon=":material/support_agent:")
