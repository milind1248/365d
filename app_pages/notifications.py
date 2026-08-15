import streamlit as st

from core import auth, db
from core.helpers import priority_badge, PRIORITY_ORDER

user = auth.require_login()
owner_id = user["id"]

st.title("🔔 Alerts & Notifications")

TYPE_ICONS = {
    "scan_alert": "🔬", "expert_case": "🧑‍🌾", "task_reminder": "📅",
    "climate_alert": "🌦️", "feeding_reminder": "🍃", "instar_reminder": "🦋", "mortality_alert": "💀",
}
CATEGORY_LABELS = {
    "disease": "Disease", "climate": "Climate", "feeding": "Feeding", "instar": "Instar",
    "mortality": "Mortality", "pest": "Pest", "general": "General",
}

all_notifications = db.fetch_all("notification_item", "owner_id = ?", (owner_id,), order_by="created_at DESC")

if not all_notifications:
    st.info("No notifications yet. High-risk scans, weather risk, silkworm rearing reminders and expert case updates will appear here.")
    st.stop()

open_alerts = [n for n in all_notifications if (n.get("status") or "open") == "open"]
unread = [n for n in all_notifications if not n["is_read"]]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Open alerts", len(open_alerts))
c2.metric("Critical/High", sum(1 for n in open_alerts if (n.get("priority") or "low") in ("critical", "high")))
c3.metric("Unread", len(unread))
c4.metric("Total", len(all_notifications))

if unread and st.button(f"Mark all {len(unread)} as read", icon=":material/done_all:"):
    for n in unread:
        db.update_row("notification_item", n["id"], {"is_read": 1})
    st.rerun()

st.divider()

f1, f2 = st.columns(2)
categories_present = sorted({n.get("category") or "general" for n in all_notifications})
category_filter = f1.multiselect(
    "Filter by category", categories_present,
    default=categories_present, format_func=lambda c: CATEGORY_LABELS.get(c, c.title()),
)
status_filter = f2.selectbox("Status", ["Open", "Completed", "Skipped", "All"])

filtered = [n for n in all_notifications if (n.get("category") or "general") in category_filter]
if status_filter != "All":
    target = status_filter.lower()
    filtered = [n for n in filtered if (n.get("status") or "open") == target]

filtered.sort(key=lambda n: PRIORITY_ORDER.get(n.get("priority") or "low", 3))

st.caption(f"{len(filtered)} of {len(all_notifications)} alerts")

for n in filtered:
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        marker = "" if n["is_read"] else "🔵 "
        icon = TYPE_ICONS.get(n["type"], "🔔")
        c1.markdown(f"{marker}{icon} **{n['title']}**  {priority_badge(n.get('priority') or 'low')}")
        c1.caption(n["body"])
        c1.caption(
            f"{n['created_at'][:16].replace('T', ' ')} · {CATEGORY_LABELS.get(n.get('category') or 'general', 'General')}"
        )

        status = n.get("status") or "open"
        if status == "open":
            outcome_key = f"outcome_{n['id']}"
            outcome = c1.text_input("Outcome (optional)", key=outcome_key, label_visibility="collapsed", placeholder="Optional note on what you did...")
            b1, b2 = c1.columns(2)
            if b1.button("Mark completed", key=f"complete_{n['id']}", icon=":material/check_circle:", width="stretch"):
                db.update_row("notification_item", n["id"], {"status": "completed", "outcome": outcome, "resolved_at": db.now_iso(), "is_read": 1})
                st.rerun()
            if b2.button("Mark skipped", key=f"skip_{n['id']}", icon=":material/cancel:", width="stretch"):
                db.update_row("notification_item", n["id"], {"status": "skipped", "outcome": outcome, "resolved_at": db.now_iso(), "is_read": 1})
                st.rerun()
        else:
            status_icon = "✅" if status == "completed" else "⏭️"
            c1.caption(f"{status_icon} {status.title()}" + (f" — {n['outcome']}" if n.get("outcome") else "") + (f" · {n['resolved_at'][:16].replace('T', ' ')}" if n.get("resolved_at") else ""))

        if c2.button("Delete", key=f"del_notif_{n['id']}", icon=":material/delete:"):
            db.delete_row("notification_item", n["id"])
            st.rerun()
