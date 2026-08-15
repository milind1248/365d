import streamlit as st

from core import auth, db

st.set_page_config(page_title="365Dfarms Mulberry AI", page_icon="🌱", layout="wide")
db.init_db()

user = auth.current_user()

if not user:
    col_logo, col_form = st.columns([1, 1.4], gap="large")
    with col_logo:
        st.image("assets/images/logo.png", width=140)
        st.title("365Dfarms Mulberry AI")
        st.caption("Go Grow Green — mulberry crop health & advisory, in your browser.")
        st.markdown(
            "AI-assisted disease/pest/deficiency scanning, farm records, weather-based "
            "spray guidance and expert help for sericulture and mulberry farmers."
        )

    with col_form:
        tab_login, tab_register, tab_guest = st.tabs(["Log in", "Create account", "Guest"])

        with tab_login:
            with st.form("login_form"):
                identifier = st.text_input("Mobile number or email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary", width="stretch")
            if submitted:
                ok, message = auth.login(identifier, password)
                if ok:
                    st.rerun()
                else:
                    st.error(message)

        with tab_register:
            with st.form("register_form"):
                full_name = st.text_input("Full name")
                mobile = st.text_input("Mobile number")
                email = st.text_input("Email (optional)")
                lang = st.selectbox("Preferred language", ["en", "mr", "hi"], format_func=lambda k: {"en": "English", "mr": "मराठी", "hi": "हिंदी"}[k])
                password = st.text_input("Password", type="password", key="reg_pw")
                submitted = st.form_submit_button("Create account", type="primary", width="stretch")
            if submitted:
                ok, message = auth.register(full_name, mobile, email, password, lang)
                if ok:
                    st.rerun()
                else:
                    st.error(message)

        with tab_guest:
            st.write("Explore the app without creating an account. Guest data stays on this server only.")
            if st.button("Continue as guest", width="stretch"):
                auth.continue_as_guest()
                st.rerun()

    st.stop()

pages = {
    "": [
        st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/home:", default=True),
    ],
    "Farm": [
        st.Page("app_pages/my_farm.py", title="My Farm", icon=":material/grass:"),
        st.Page("app_pages/scan_disease.py", title="Scan Disease", icon=":material/photo_camera:"),
        st.Page("app_pages/ai_model_report.py", title="AI Model Report", icon=":material/analytics:"),
        st.Page("app_pages/disease_library.py", title="Disease Library", icon=":material/menu_book:"),
        st.Page("app_pages/crop_calendar.py", title="Crop Calendar", icon=":material/event:"),
    ],
    "Records": [
        st.Page("app_pages/logs.py", title="Spray, Fertilizer & Soil Logs", icon=":material/science:"),
        st.Page("app_pages/production.py", title="Production", icon=":material/inventory_2:"),
    ],
    "Advisory": [
        st.Page("app_pages/weather.py", title="Weather & Disease Risk", icon=":material/cloud:"),
        st.Page("app_pages/expert_help.py", title="Expert Help", icon=":material/support_agent:"),
        st.Page("app_pages/chatbot.py", title="Chatbot", icon=":material/chat:"),
    ],
    "Account": [
        st.Page("app_pages/notifications.py", title="Notifications", icon=":material/notifications:"),
        st.Page("app_pages/settings.py", title="Settings", icon=":material/settings:"),
        st.Page("app_pages/admin.py", title="User Management", icon=":material/admin_panel_settings:"),
    ],
}

st.logo("assets/images/logo.png", size="large")

with st.sidebar:
    st.markdown(f"**{user['full_name'] or 'Farmer'}**")
    st.caption(user.get("village") or ("Guest session" if user.get("is_guest") else "Mulberry farmer"))
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        auth.logout()
        st.rerun()
    st.divider()

page = st.navigation(pages, position="sidebar")
page.run()
