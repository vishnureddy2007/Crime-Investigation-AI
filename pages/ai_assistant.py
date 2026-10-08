"""
AI Assistant Chat page.

Provides interactive Q&A regarding case evidence, verified weapons, severity scores,
and AI narratives. Answers simple database/fact questions instantly from DB,
and uses cached AI analysis for complex reasoning.
"""

from __future__ import annotations

import streamlit as st
from config import DATABASE_PATH
from database.repository import list_cases, get_latest_analysis, get_latest_summary
from pages._layout import empty_state, render_page_header
from services.chat_assistant import ChatAssistant
from models.schemas import EvidenceAnalysis


def render() -> None:
    render_page_header(
        "💬",
        title="AI Assistant",
        subtitle="Ask questions regarding evidence, weapon verifications, and investigation analysis.",
    )

    cases = list_cases(DATABASE_PATH)
    if not cases:
        empty_state(
            "No active cases found.",
            "Process evidence in the Investigation workspace first to chat with the AI Assistant.",
            action_label="New Investigation",
            action_target="New Investigation",
        )
        return

    # Case Selector
    case_map = {f"Case #{c['case_id']:03d} ({c['source_name']})": c for c in cases}
    selected_label = st.selectbox("Select Case Context", list(case_map.keys()))
    case_id = case_map[selected_label]["case_id"]

    db_analysis = get_latest_analysis(DATABASE_PATH, case_id)
    analysis_obj = None
    if db_analysis and db_analysis.get("payload_json"):
        import json
        try:
            payload = json.loads(db_analysis["payload_json"])
            analysis_obj = EvidenceAnalysis.from_dict(payload)
        except Exception:
            pass

    # Chat history state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.markdown("---")

    # Render previous messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.caption(f"Source: `{msg['source']}`")

    # User Chat Input
    prompt = st.chat_input("Ask a question about this investigation...")
    if prompt:
        # Display user message
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Reply
        assistant = ChatAssistant(analysis=analysis_obj)
        reply = assistant.reply(prompt, analysis=analysis_obj)

        with st.chat_message("assistant"):
            st.markdown(reply.answer)
            st.caption(f"Source: `{reply.source}` | Model: `{reply.model_name}`")

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": reply.answer,
            "source": reply.source,
        })
