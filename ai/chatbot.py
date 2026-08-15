"""Advisory chatbot: Groq-hosted LLM (llama-3.3-70b-versatile) as the primary
answer engine, grounded with matching entries from the offline knowledge base
so chemical/dosage advice stays consistent with the app's vetted content.

Falls back to the original local keyword-match engine (ported from
features/chatbot/chatbot_engine.dart) whenever Groq isn't configured or the
call fails for any reason (no key, network, rate limit) - the chatbot always
answers something, same "must run end-to-end without external services"
principle the rest of this app follows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

from ai.classifier import AI_ADVISORY_DISCLAIMER
from ai.disease_risk import assess_day
from ai.weather import get_forecast

KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent / "data" / "diseases_knowledge_base.json"

CHEMICAL_SAFETY_DISCLAIMER = (
    "Consult your local agriculture officer before using any chemical treatment. "
    "Follow label dosage, safety interval and re-entry interval."
)

DEFAULT_LAT, DEFAULT_LON = 19.07, 74.74  # Maharashtra sericulture belt, same default as the mobile app

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT_SECONDS = 15
MAX_HISTORY_TURNS = 6  # user+assistant pairs kept for conversational context

SYSTEM_PROMPT = """You are the advisory chatbot inside "365Dfarms Mulberry AI", a web app for \
mulberry and sericulture farmers in India. Answer clearly, practically, and briefly (usually \
under 120 words) in a farmer-friendly tone - avoid jargon, or explain it simply if unavoidable.

Rules:
- If reference information from the app's knowledge base is provided below, ground your answer \
in it and do not contradict it - it reflects vetted local agricultural guidance.
- Always end any chemical-treatment advice with a reminder to consult the local agriculture \
officer and follow label dosage, safety interval and re-entry interval.
- If the question describes a severe, worsening, or unclear problem, recommend the farmer use \
the app's "Submit to Expert" feature for a personal reply.
- You are advisory only, not a substitute for professional agronomic advice.
- Answer in the same language the farmer used (English, Hindi, or Marathi)."""


@st.cache_data(show_spinner=False)
def _knowledge_base() -> list[dict]:
    return json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))


def _mentions_spray(text: str) -> bool:
    return any(w in text for w in ("spray", "फवारणी", "छिड़काव"))


def _mentions_weather(text: str) -> bool:
    return any(w in text for w in ("rain", "पाऊस", "बारिश", "tomorrow"))


def _spray_weather_answer() -> str:
    forecast = get_forecast(DEFAULT_LAT, DEFAULT_LON)
    assessment = assess_day(forecast[0])
    if assessment.spray_recommended:
        return f"Today looks suitable for spraying - {assessment.best_spray_window_note}."
    return f"Spraying is not recommended today - {assessment.best_spray_window_note}."


def _matching_kb_entries(normalized: str, limit: int = 2) -> list[dict]:
    matches = []
    for entry in _knowledge_base():
        matchers = [
            entry.get("name_en", "").lower(),
            entry.get("name_mr", "").lower(),
            entry.get("name_hi", "").lower(),
            *entry.get("symptoms", "").lower().replace(",", " ").replace(".", " ").split(),
        ]
        if any(len(m) > 3 and m in normalized for m in matchers):
            matches.append(entry)
        if len(matches) >= limit:
            break
    return matches


def _offline_answer(normalized: str) -> str:
    matches = _matching_kb_entries(normalized, limit=1)
    if matches:
        entry = matches[0]
        return (
            f"**{entry.get('name_en')}** ({entry.get('name_mr')})\n\n"
            f"**Symptoms:** {entry.get('symptoms')}\n\n"
            f"**Organic control:** {entry.get('organic_control')}\n\n"
            f"{CHEMICAL_SAFETY_DISCLAIMER}"
        )
    return (
        "I could not find a confident answer for that yet. Please try describing the symptom "
        'differently, or use "Submit to Expert" in the Expert Help section for a personal reply.\n\n'
        f"{AI_ADVISORY_DISCLAIMER}"
    )


def _groq_api_key() -> str | None:
    try:
        key = st.secrets.get("llm", {}).get("groq_api_key")
    except Exception:
        key = None
    return key or os.environ.get("GROQ_API_KEY")


def _groq_answer(question: str, normalized: str, history: list[dict], api_key: str) -> str:
    reference = ""
    matches = _matching_kb_entries(normalized, limit=2)
    if matches:
        blocks = []
        for e in matches:
            blocks.append(
                f"- {e.get('name_en')} ({e.get('category')}): symptoms: {e.get('symptoms')} | "
                f"organic control: {e.get('organic_control')} | chemical control: {e.get('chemical_control')} | "
                f"safety interval: {e.get('safety_interval_days')} days | "
                f"escalate to expert if: {e.get('expert_escalation_condition')}"
            )
        reference = "\n\nReference information from the app's knowledge base:\n" + "\n".join(blocks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + reference}]
    messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": question})

    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.4, "max_tokens": 500},
        timeout=GROQ_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def answer(question: str, history: list[dict] | None = None) -> tuple[str, str]:
    """Returns (answer_text, source) where source is "groq" or "offline"."""
    normalized = question.lower()

    if _mentions_spray(normalized) and _mentions_weather(normalized):
        return _spray_weather_answer(), "rule"

    api_key = _groq_api_key()
    if api_key:
        try:
            return _groq_answer(question, normalized, history or [], api_key), "groq"
        except Exception:
            pass  # fall through to offline engine below

    return _offline_answer(normalized), "offline"
