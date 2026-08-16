import streamlit as st

from core import auth, db, email_notify
from core.helpers import add_notification, plot_options

user = auth.require_login()
owner_id = user["id"]

st.title("🧑‍🌾 Expert Help")
st.caption("Submit a case for a sericulture/agriculture expert to review. Cases from Scan Disease are pre-linked.")

scans = db.fetch_all("scan_record", "owner_id = ?", (owner_id,), order_by="captured_at DESC")

with st.expander("➕ Submit a new case", expanded=True):
    with st.form("expert_form", clear_on_submit=True):
        scan_options = {"": "— not linked to a scan —"} | {s["id"]: f"{s['display_name']} ({s['captured_at'][:10]})" for s in scans}
        scan_id = st.selectbox("Related scan (optional)", options=list(scan_options.keys()), format_func=lambda k: scan_options[k])
        description = st.text_area("Describe the problem*", placeholder="What symptoms are you seeing? When did it start?")
        c1, c2 = st.columns(2)
        crop_age = c1.number_input("Crop age (months)", min_value=0, step=1)
        location = c2.text_input("Village/location")
        submitted = st.form_submit_button("Submit to expert", type="primary")
    if submitted:
        if not description.strip():
            st.error("Please describe the problem.")
        else:
            db.insert_row(
                "expert_case",
                {
                    "owner_id": owner_id, "scan_record_id": scan_id or None,
                    "description": description.strip(), "crop_age_months": crop_age,
                    "location": location, "status": "pending", "expert_reply": "",
                },
            )
            add_notification(owner_id, "expert_case", "Case submitted", "Your case was submitted and is pending expert review.")
            email_notify.notify_new_expert_case(user["full_name"], user.get("mobile_number"), description.strip(), location, crop_age)
            st.success("Case submitted. You'll see the expert's reply here once reviewed.")
            st.rerun()

st.divider()
st.subheader("My cases")

cases = db.fetch_all("expert_case", "owner_id = ?", (owner_id,), order_by="created_at DESC")
if not cases:
    st.info("No cases submitted yet.")
else:
    status_icons = {"pending": "🟡", "in_review": "🔵", "closed": "🟢"}
    for case in cases:
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"{status_icons.get(case['status'], '⚪')} **{case['description'][:80]}**")
            c1.caption(f"Submitted {case['created_at'][:16].replace('T', ' ')} · Status: {case['status'].replace('_', ' ').title()}")
            if case["expert_reply"]:
                c1.success(f"Expert reply: {case['expert_reply']}")
            else:
                c1.caption("Awaiting expert reply. Typical response time: 24-48 hours.")
            if c2.button("Delete", key=f"del_case_{case['id']}", icon=":material/delete:"):
                db.delete_row("expert_case", case["id"])
                st.rerun()
