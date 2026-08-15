import streamlit as st

from core import auth, db

user = auth.require_login()
owner_id = user["id"]

st.title("🔔 Notifications")

notifications = db.fetch_all("notification_item", "owner_id = ?", (owner_id,), order_by="created_at DESC")

if not notifications:
    st.info("No notifications yet. High-risk scans and expert case updates will appear here.")
else:
    unread = [n for n in notifications if not n["is_read"]]
    if unread and st.button(f"Mark all {len(unread)} as read", icon=":material/done_all:"):
        for n in unread:
            db.update_row("notification_item", n["id"], {"is_read": 1})
        st.rerun()

    type_icons = {"scan_alert": "🔬", "expert_case": "🧑‍🌾", "task_reminder": "📅"}
    for n in notifications:
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            marker = "" if n["is_read"] else "🔵 "
            c1.markdown(f"{marker}{type_icons.get(n['type'], '🔔')} **{n['title']}**")
            c1.caption(n["body"])
            c1.caption(n["created_at"][:16].replace("T", " "))
            if c2.button("Delete", key=f"del_notif_{n['id']}", icon=":material/delete:"):
                db.delete_row("notification_item", n["id"])
                st.rerun()
