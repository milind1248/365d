"""Long-lived "remember me" login, layered on top of core/auth.py's
in-memory st.session_state.

Streamlit's session state lives only for the current browser tab's
WebSocket connection - a full page reload (or reopening the app later)
loses it, forcing a fresh login every time. That's too short for a farmer
who wants to open the app once and stay logged in. This module stores a
random, DB-backed session token (core.auth.create_session_token) in a
browser cookie with a long expiry (core.auth.REMEMBER_ME_DAYS, 30 days by
default) via extra_streamlit_components' CookieManager, and restores the
login from that cookie on future visits.

Three timing/API quirks of that component drive the design below (all
confirmed by testing, not theoretical):
  - CookieManager's __init__ itself issues a component call ("getAll"), so
    it counts as a widget - wrapping its construction in @st.cache_resource
    trips Streamlit's "widget created inside a cached function" policy
    (CachedWidgetWarning). It must be a plain, uncached construction.
  - It must be constructed with the SAME key exactly ONCE per script run.
    A second, differently-keyed instance is a component the browser has
    never seen before, so its .cookies dict is still the {} default on
    this first render - and this library's .delete() does an unguarded
    `del self.cookies[x]`, raising KeyError instead of a clean no-op. Fixed
    by constructing it once in streamlit_app.py's top level
    (get_manager()) and having any sub-page that also needs cookie access
    in the same run read that same instance back via get_shared_manager()
    instead of constructing its own.
  - .get() can return None on the very first script run after a fresh page
    load even when a valid cookie exists, because the component hasn't
    delivered its value back yet - fixed with a single, guarded retry
    rerun in restore_session() (never more than one, so a genuinely
    logged-out visitor doesn't loop).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import extra_streamlit_components as stx
import streamlit as st

from core import auth

COOKIE_NAME = "mulberry_session_token"
_PENDING_REMEMBER_KEY = "_pending_remember_user_id"
_COOKIE_RETRY_KEY = "_cookie_restore_retried"
_SHARED_MANAGER_KEY = "_shared_cookie_manager"


def get_manager() -> stx.CookieManager:
    """Call exactly once per script run, from streamlit_app.py's top level
    (before page.run()). Stores the instance in st.session_state so any
    sub-page executed later in this same run can fetch the identical
    object via get_shared_manager() instead of constructing a second,
    cold one - see module docstring."""
    cm = stx.CookieManager(key="mulberry_cookie_manager")
    st.session_state[_SHARED_MANAGER_KEY] = cm
    return cm


def get_shared_manager() -> stx.CookieManager:
    """Call from a sub-page that needs cookie access (e.g. its own logout
    button) - returns the instance streamlit_app.py's top level already
    constructed earlier in this same run via get_manager()."""
    cm = st.session_state.get(_SHARED_MANAGER_KEY)
    if cm is None:
        raise RuntimeError(
            "get_shared_manager() called before get_manager() ran in this "
            "script run - call session_cookie.get_manager() once at the top "
            "of streamlit_app.py first."
        )
    return cm


def restore_session(cm: stx.CookieManager) -> None:
    """Call once near the top of streamlit_app.py, before the login gate.
    Auto-logs-in from a valid remember-me cookie if the in-memory session
    is empty (e.g. after a page reload or reopening the browser)."""
    if auth.current_user():
        return

    token = cm.get(COOKIE_NAME)
    if not token:
        # Might be a real "no cookie" or just the component not having
        # delivered its value back yet on this very first run - retry once.
        if not st.session_state.get(_COOKIE_RETRY_KEY):
            st.session_state[_COOKIE_RETRY_KEY] = True
            st.rerun()
        return

    user = auth.validate_session_token(token)
    if user:
        st.session_state[auth.SESSION_USER_KEY] = user["id"]


def mark_for_remember(user_id: str) -> None:
    """Call right after a successful login/register/guest, before the
    st.rerun() that shows the dashboard. Defers the actual cookie write to
    apply_pending_remember() on the next render - see module docstring."""
    st.session_state[_PENDING_REMEMBER_KEY] = user_id


def apply_pending_remember(cm: stx.CookieManager) -> None:
    """Call once on every authenticated render (after the login gate).
    Issues the long-lived cookie queued by mark_for_remember(), if any."""
    user_id = st.session_state.pop(_PENDING_REMEMBER_KEY, None)
    if not user_id:
        return
    token = auth.create_session_token(user_id)
    expires_at = datetime.now() + timedelta(days=auth.REMEMBER_ME_DAYS)
    cm.set(COOKIE_NAME, token, expires_at=expires_at, key="mulberry_set_cookie")


def forget(cm: stx.CookieManager) -> None:
    """Call on logout: revokes the DB-side token and clears the cookie."""
    token = cm.get(COOKIE_NAME)
    if token:
        auth.revoke_session_token(token)
    try:
        cm.delete(COOKIE_NAME, key="mulberry_delete_cookie")
    except KeyError:
        pass  # already absent from the component's local cache - nothing to clear
