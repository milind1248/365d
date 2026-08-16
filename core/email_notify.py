"""Outbound admin notification emails (new signup, new expert case) sent to
the operator inbox - not farmer-facing, so there's no offline fallback to
build (unlike ai/chatbot.py's Groq fallback): if it's not configured, the
call is a silent no-op and the app keeps working exactly as before.

Uses Gmail SMTP with an App Password (see README.md "Email notifications
setup") - not the account's real login password, which Gmail no longer
accepts for SMTP auth at all.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

import streamlit as st

DEFAULT_TO_ADDR = "365dfarmsai@gmail.com"
DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587

CONTACT_PHONE = "+91-9767589994"

WEBSITE_FEATURES = [
    "AI Scan Disease - photograph a mulberry leaf and get an instant disease/pest/deficiency diagnosis",
    "AI Scan Silkworm - photograph a silkworm larva and get an instant health check (healthy / Grasserie / signs of illness)",
    "Silkworm Rearing tracker - log instar stage, feeding and mortality; get reminders when action is due",
    "Weather & Disease Risk - daily mildew/rust/fungal/pest risk and best-spray-window guidance for your location",
    "My Farm - manage your plots and their location",
    "Spray, Fertilizer & Soil Logs and Production records - keep all your farm records in one place",
    "Crop Calendar - track upcoming farm tasks",
    "Expert Help - submit a case and get guidance from a sericulture/agriculture expert",
    "Chatbot - ask farming questions any time, in English/Marathi/Hindi",
    "Alerts & Notifications - a prioritized feed of everything that needs your attention",
]


def _config() -> dict:
    cfg = st.secrets.get("email", {})
    return {
        "smtp_host": cfg.get("smtp_host") or os.environ.get("SMTP_HOST", DEFAULT_SMTP_HOST),
        "smtp_port": int(cfg.get("smtp_port") or os.environ.get("SMTP_PORT", DEFAULT_SMTP_PORT)),
        "smtp_user": cfg.get("smtp_user") or os.environ.get("SMTP_USER"),
        "smtp_password": cfg.get("smtp_password") or os.environ.get("SMTP_PASSWORD"),
        "to_addr": cfg.get("notify_to") or os.environ.get("EMAIL_NOTIFY_TO", DEFAULT_TO_ADDR),
    }


def is_configured() -> bool:
    cfg = _config()
    return bool(cfg["smtp_user"] and cfg["smtp_password"])


def send_notification_email(subject: str, body: str, to_addr: str | None = None) -> bool:
    """Best-effort: never raises. Returns False (and logs to the Streamlit
    server console) if email isn't configured or the send fails, so a
    signup/case-submit action never breaks on an SMTP problem.

    Defaults to the admin inbox (cfg["to_addr"]) - pass to_addr to send
    somewhere else instead, e.g. the farmer's own address for a welcome
    email."""
    cfg = _config()
    if not cfg["smtp_user"] or not cfg["smtp_password"]:
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["smtp_user"]
    msg["To"] = to_addr or cfg["to_addr"]
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=10) as server:
            server.starttls(context=context)
            server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort notification, never break the caller's flow
        print(f"[email_notify] send failed: {exc}")
        return False


def notify_new_signup(full_name: str, mobile_number: str, email: str) -> None:
    body = (
        f"A new farmer account was created on 365Dfarms Mulberry AI.\n\n"
        f"Name: {full_name or '-'}\n"
        f"Mobile: {mobile_number or '-'}\n"
        f"Email: {email or '-'}\n"
    )
    send_notification_email("365Dfarms: New signup", body)


def send_welcome_email(full_name: str, to_email: str) -> None:
    """Sent to the farmer's own email (only if they gave one at signup -
    mobile-only accounts simply don't get this, same as any other
    email-only feature in the app)."""
    if not to_email or not to_email.strip():
        return

    bullets = "\n".join(f"  - {feature}" for feature in WEBSITE_FEATURES)
    body = (
        f"Namaste {full_name or 'Farmer'},\n\n"
        f"Welcome to 365Dfarms Mulberry AI! Your account has been created "
        f"successfully.\n\n"
        f"Here's what you can do on the website:\n\n"
        f"{bullets}\n\n"
        f"Need help or have questions? Call us at {CONTACT_PHONE} or use "
        f"the Expert Help / Chatbot pages inside the app.\n\n"
        f"Go Grow Green!\n"
        f"Team 365Dfarms"
    )
    send_notification_email("Welcome to 365Dfarms Mulberry AI!", body, to_addr=to_email.strip())


def notify_new_expert_case(farmer_name: str, farmer_mobile: str, description: str, location: str, crop_age_months) -> None:
    body = (
        f"A farmer submitted a new case on 365Dfarms Mulberry AI.\n\n"
        f"Farmer: {farmer_name or '-'}\n"
        f"Mobile: {farmer_mobile or '-'}\n"
        f"Location: {location or '-'}\n"
        f"Crop age: {crop_age_months if crop_age_months else '-'} months\n\n"
        f"Description:\n{description}\n"
    )
    send_notification_email("365Dfarms: New expert case submitted", body)
