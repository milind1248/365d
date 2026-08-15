import datetime as dt

import streamlit as st

from ai.chatbot import answer
from core import auth

user = auth.require_login()

st.title("💬 Advisory Chatbot")
st.caption(
    "Ask about symptoms, treatments or spray timing. Grounded in the app's knowledge base, with an "
    "offline fallback if the AI service is unavailable."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Namaste! Ask me about a mulberry disease, pest or 'can I spray tomorrow?'",
            "time": dt.datetime.now().strftime("%d %b, %H:%M"),
        }
    ]

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["time"])

if prompt := st.chat_input("Type your question..."):
    now = dt.datetime.now().strftime("%d %b, %H:%M")
    st.session_state.chat_history.append({"role": "user", "content": prompt, "time": now})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(now)

    history_for_llm = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[:-1]
        if m["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, _source = answer(prompt, history=history_for_llm)
        reply_time = dt.datetime.now().strftime("%d %b, %H:%M")
        st.markdown(reply)
        st.caption(reply_time)
    st.session_state.chat_history.append({"role": "assistant", "content": reply, "time": reply_time})

with st.sidebar:
    st.caption("Try asking:")
    for example in ["Powdery mildew treatment", "Can I spray tomorrow?", "Yellow leaves nitrogen", "Whitefly control"]:
        st.caption(f"• {example}")
