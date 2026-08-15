"""Session-based auth backed by the local user_profile table.

Mirrors the Flutter app's auth_repository.dart intent (email/mobile + local
account, plus a "continue as guest" path) but simplified for a single-process
Streamlit deployment: no Supabase, salted-hash passwords stored locally.
"""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import streamlit as st

from core import db

SESSION_USER_KEY = "auth_user"

# "Remember me" cookie lifetime - see core/session_cookie.py, which issues
# and restores the long-lived token this creates/validates. Long duration
# per user request; a single knob to shorten later if needed.
REMEMBER_ME_DAYS = 30


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000)
    return f"{salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(_hash_password(password, salt), stored)


def register(full_name: str, mobile_number: str, email: str, password: str, preferred_language: str = "en") -> tuple[bool, str]:
    if not full_name.strip():
        return False, "Full name is required."
    if not mobile_number.strip() and not email.strip():
        return False, "Enter a mobile number or email."
    existing = db.fetch_one("user_profile", "mobile_number = ? OR (email = ? AND email != '')", (mobile_number, email))
    if existing:
        return False, "An account with this mobile number or email already exists."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    user_id = db.insert_row(
        "user_profile",
        {
            "full_name": full_name.strip(),
            "mobile_number": mobile_number.strip(),
            "email": email.strip(),
            "password_hash": _hash_password(password),
            "preferred_language": preferred_language,
            "farmer_type": "",
            "village": "", "taluka": "", "district": "", "state": "",
            "farm_size_acres": 0.0, "number_of_plots": 0,
            "mulberry_variety": "", "irrigation_type": "",
            "is_organic_preferred": 0, "is_guest": 0,
            "photo_training_consent": 0,
            "last_login_at": db.now_iso(), "is_active": 1,
        },
    )
    st.session_state[SESSION_USER_KEY] = user_id
    return True, "Account created."


def login(identifier: str, password: str) -> tuple[bool, str]:
    user = db.fetch_one("user_profile", "mobile_number = ? OR email = ?", (identifier.strip(), identifier.strip()))
    if not user or not user.get("password_hash"):
        return False, "No account found for that mobile number or email."
    if not _verify_password(password, user["password_hash"]):
        return False, "Incorrect password."
    if user.get("is_active") == 0:
        return False, "This account's access has been revoked. Contact the app administrator."
    db.update_row("user_profile", user["id"], {"last_login_at": db.now_iso()})
    st.session_state[SESSION_USER_KEY] = user["id"]
    return True, "Welcome back."


def continue_as_guest() -> None:
    user_id = db.insert_row(
        "user_profile",
        {
            "full_name": "Guest Farmer",
            "mobile_number": "", "email": "", "password_hash": "",
            "preferred_language": "en", "farmer_type": "",
            "village": "", "taluka": "", "district": "", "state": "",
            "farm_size_acres": 0.0, "number_of_plots": 0,
            "mulberry_variety": "", "irrigation_type": "",
            "is_organic_preferred": 0, "is_guest": 1,
            "photo_training_consent": 0,
            "last_login_at": db.now_iso(), "is_active": 1,
        },
    )
    st.session_state[SESSION_USER_KEY] = user_id


def logout() -> None:
    st.session_state.pop(SESSION_USER_KEY, None)


def current_user() -> dict | None:
    user_id = st.session_state.get(SESSION_USER_KEY)
    if not user_id:
        return None
    return db.fetch_one("user_profile", "id = ?", (user_id,))


def create_session_token(user_id: str, days: int = REMEMBER_ME_DAYS) -> str:
    """Issues a long-lived, random session token for a "remember me" cookie
    (see core/session_cookie.py) - separate from the short-lived in-memory
    st.session_state, which is lost on every full page reload."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    db.insert_row("user_session", {"id": token, "user_id": user_id, "expires_at": expires_at})
    return token


def validate_session_token(token: str) -> dict | None:
    """Returns the token's user if it exists, hasn't expired, and that
    account is still active - None otherwise (also purges expired/orphaned
    tokens so they don't linger)."""
    row = db.fetch_one("user_session", "id = ?", (token,))
    if not row:
        return None
    if row["expires_at"] < db.now_iso():
        db.delete_row("user_session", token)
        return None
    user = db.fetch_one("user_profile", "id = ?", (row["user_id"],))
    if not user or user.get("is_active") == 0:
        db.delete_row("user_session", token)
        return None
    return user


def revoke_session_token(token: str) -> None:
    db.delete_row("user_session", token)


def require_login() -> dict:
    """Call at the top of any protected page. Stops the script if not logged
    in, or if an admin has revoked this account's access mid-session.

    Note: streamlit_app.py is the entrypoint script, not a page registered
    via st.Page/st.navigation, so st.page_link can't target it directly -
    a plain rerun is enough, since the entrypoint re-executes from the top
    and renders the login screen itself once current_user() is None."""
    user = current_user()
    if not user:
        st.warning("Please log in first.")
        if st.button("Go to login", icon=":material/login:"):
            st.rerun()
        st.stop()
    if user.get("is_active") == 0:
        logout()
        st.warning("This account's access has been revoked. Contact the app administrator.")
        if st.button("Go to login", icon=":material/login:"):
            st.rerun()
        st.stop()
    return user
