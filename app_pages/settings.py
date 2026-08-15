import streamlit as st

from core import auth, db, session_cookie

user = auth.require_login()

st.title("⚙️ Settings & Profile")

tab_profile, tab_app, tab_data = st.tabs(["Profile", "App info", "Data"])

with tab_profile:
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        full_name = c1.text_input("Full name", value=user["full_name"] or "")
        mobile = c2.text_input("Mobile number", value=user["mobile_number"] or "")
        c3, c4 = st.columns(2)
        village = c3.text_input("Village", value=user["village"] or "")
        district = c4.text_input("District", value=user["district"] or "")
        c5, c6 = st.columns(2)
        farmer_type = c5.selectbox("Farmer type", ["", "Sericulture", "Mixed", "Other"], index=0)
        lang = c6.selectbox(
            "Preferred language", ["en", "mr", "hi"],
            index=["en", "mr", "hi"].index(user["preferred_language"]) if user["preferred_language"] in ["en", "mr", "hi"] else 0,
            format_func=lambda k: {"en": "English", "mr": "मराठी", "hi": "हिंदी"}[k],
        )
        c7, c8 = st.columns(2)
        farm_size = c7.number_input("Total farm size (acres)", min_value=0.0, value=float(user["farm_size_acres"] or 0), step=0.1)
        variety = c8.text_input("Mulberry variety", value=user["mulberry_variety"] or "")
        organic_pref = st.checkbox("Prefer organic treatments", value=bool(user["is_organic_preferred"]))
        submitted = st.form_submit_button("Save profile", type="primary")
    if submitted:
        db.update_row(
            "user_profile", user["id"],
            {
                "full_name": full_name, "mobile_number": mobile, "village": village,
                "district": district, "preferred_language": lang, "farm_size_acres": farm_size,
                "mulberry_variety": variety, "is_organic_preferred": int(organic_pref),
            },
        )
        st.success("Profile updated.")
        st.rerun()

with tab_app:
    st.markdown("**365Dfarms Mulberry AI** — Go Grow Green")
    st.caption("Web edition, ported from the Flutter mobile app to Streamlit for browser access and deployment.")
    st.write(
        "Offline-first mulberry crop health & advisory app for sericulture and mulberry farmers: "
        "AI-assisted disease/pest/deficiency scanning, farm records, weather-based spray guidance and "
        "expert help."
    )
    st.caption("This build stores data in a local SQLite database on the server. On Streamlit Community "
               "Cloud that storage is ephemeral — see README.md for persistent-storage options.")

with tab_data:
    st.warning("This permanently deletes all your plots, scans, logs and records.")
    confirm = st.checkbox("I understand this cannot be undone")
    if st.button("Delete all my data", disabled=not confirm, icon=":material/delete_forever:"):
        tables = ["farm_plot", "scan_record", "spray_log", "fertilizer_log", "soil_test", "crop_task", "production_record", "expert_case", "notification_item"]
        db.execute_transaction([(f"DELETE FROM {t} WHERE owner_id = ?", (user["id"],)) for t in tables])
        st.success("All your data has been deleted.")
        st.rerun()

    st.divider()
    if st.button("Log out", icon=":material/logout:"):
        session_cookie.forget(session_cookie.get_shared_manager())
        auth.logout()
        st.rerun()
