import pandas as pd
import streamlit as st

from core import db

# Temporary hardcoded admin password - not tied to any user_profile account,
# this is a separate, simple gate for the admin/user-management view only.
# TODO: replace with a real admin account + hashed password before any
# public deployment - see core/auth.py for the pattern to follow.
ADMIN_PASSWORD = "365D"
SESSION_ADMIN_KEY = "is_admin"

CASCADE_TABLES = [
    "farm_plot", "scan_record", "spray_log", "fertilizer_log", "soil_test",
    "crop_task", "production_record", "expert_case", "notification_item",
]

st.title("🔐 User Management")

if not st.session_state.get(SESSION_ADMIN_KEY):
    st.caption("Admin access only.")
    with st.form("admin_login_form"):
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        if password == ADMIN_PASSWORD:
            st.session_state[SESSION_ADMIN_KEY] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

top = st.columns([5, 1])
top[0].caption("All registered accounts, including guest sessions.")
if top[1].button("Log out", icon=":material/logout:"):
    st.session_state.pop(SESSION_ADMIN_KEY, None)
    st.rerun()

users = db.fetch_all("user_profile", order_by="created_at DESC")

if not users:
    st.info("No users yet.")
    st.stop()

active_count = sum(1 for u in users if u.get("is_active") != 0)
guest_count = sum(1 for u in users if u.get("is_guest"))
c1, c2, c3 = st.columns(3)
c1.metric("Total accounts", len(users))
c2.metric("Active", active_count)
c3.metric("Guest sessions", guest_count)

st.divider()

st.subheader("Overview")
df = pd.DataFrame([
    {
        "Name": u["full_name"] or "—",
        "Contact": u["mobile_number"] or u["email"] or "—",
        "Type": "Guest" if u["is_guest"] else "Registered",
        "Status": "Active" if u.get("is_active") != 0 else "Revoked",
        "Created": (u["created_at"] or "")[:16].replace("T", " "),
        "Last login": (u["last_login_at"] or "—")[:16].replace("T", " ") if u.get("last_login_at") else "—",
    }
    for u in users
])
st.dataframe(df, width="stretch", hide_index=True)

st.divider()
st.subheader("Manage accounts")

for u in users:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
        c1.markdown(f"**{u['full_name'] or 'Unnamed'}**" + ("  🟢" if u.get("is_active") != 0 else "  🔴 revoked"))
        c1.caption(u["mobile_number"] or u["email"] or ("Guest session" if u["is_guest"] else "No contact info"))
        c1.caption(
            f"Created: {(u['created_at'] or '—')[:16].replace('T', ' ')}"
            f" · Last login: {(u['last_login_at'] or '—')[:16].replace('T', ' ') if u.get('last_login_at') else '—'}"
        )

        plot_count = db.count("farm_plot", "owner_id = ?", (u["id"],))
        scan_count = db.count("scan_record", "owner_id = ?", (u["id"],))
        c2.metric("Plots", plot_count)
        c3.metric("Scans", scan_count)

        if u.get("is_active") != 0:
            if c4.button("Revoke", key=f"revoke_{u['id']}", icon=":material/block:", width="stretch"):
                db.update_row("user_profile", u["id"], {"is_active": 0})
                st.rerun()
        else:
            if c4.button("Restore", key=f"restore_{u['id']}", icon=":material/check_circle:", width="stretch"):
                db.update_row("user_profile", u["id"], {"is_active": 1})
                st.rerun()

        confirm_key = f"confirm_delete_{u['id']}"
        del_col, confirm_col = st.columns([1, 3])
        if del_col.button("Delete user", key=f"del_{u['id']}", icon=":material/delete_forever:"):
            st.session_state[confirm_key] = True
        if st.session_state.get(confirm_key):
            confirm_col.warning(f"Permanently delete {u['full_name'] or 'this user'} and all their data?")
            yes_col, no_col = confirm_col.columns(2)
            if yes_col.button("Yes, delete", key=f"yes_{u['id']}", type="primary"):
                statements = [(f"DELETE FROM {table} WHERE owner_id = ?", (u["id"],)) for table in CASCADE_TABLES]
                statements.append(("DELETE FROM user_session WHERE user_id = ?", (u["id"],)))
                statements.append(("DELETE FROM user_profile WHERE id = ?", (u["id"],)))
                db.execute_transaction(statements)
                st.session_state.pop(confirm_key, None)
                st.success("User deleted.")
                st.rerun()
            if no_col.button("Cancel", key=f"no_{u['id']}"):
                st.session_state.pop(confirm_key, None)
                st.rerun()
