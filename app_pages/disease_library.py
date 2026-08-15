import json
from pathlib import Path

import streamlit as st

from core import auth

auth.require_login()

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "diseases_knowledge_base.json"
entries = json.loads(KB_PATH.read_text(encoding="utf-8"))

st.title("📖 Disease Library")
st.caption("Offline advisory knowledge base — 25 mulberry disease, pest, deficiency and stress conditions.")

search = st.text_input("Search by name or symptom", placeholder="e.g. yellow, powdery, mealybug")
categories = sorted({e["category"] for e in entries})
category_filter = st.multiselect("Filter by category", categories, default=categories)

filtered = [
    e for e in entries
    if e["category"] in category_filter
    and (not search or search.lower() in e["name_en"].lower() or search.lower() in e["symptoms"].lower())
]

st.caption(f"{len(filtered)} of {len(entries)} entries")

for entry in filtered:
    with st.expander(f"{entry['name_en']} — {entry['name_mr']}"):
        st.caption(f"Category: {entry['category'].title()}")
        st.write(f"**Symptoms:** {entry['symptoms']}")
        st.write(f"**Cause:** {entry['cause']}")
        st.write(f"**Favorable weather:** {entry['favorable_weather']}")
        st.write(f"**Organic control:** {entry['organic_control']}")
        st.write(f"**Chemical control:** {entry['chemical_control']}")
        st.write(f"**Cultural practice:** {entry['cultural_practice']}")
        st.write(f"**Prevention:** {entry['prevention']}")
        if entry["category"] != "healthy":
            st.caption(
                f"Safety interval: {entry['safety_interval_days']} days · "
                f"Re-entry: {entry['re_entry_interval_hours']} hours"
            )
            st.warning(f"Escalate to expert if: {entry['expert_escalation_condition']}")
