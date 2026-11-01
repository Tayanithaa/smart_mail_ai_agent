from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from ai_agent import GmailAgent, OllamaClient
from smart_mail_ai_agent.gmail_service import GmailService, GmailSetupError

load_dotenv()
st.set_page_config(page_title="Smart Gmail AI Agent", page_icon="📬", layout="wide")

st.title("📬 Smart Gmail AI Agent")
st.caption("A read-only AI assistant that summarizes today's Gmail and ranks important messages.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    ollama_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    max_emails = st.slider("Emails to inspect", min_value=5, max_value=100, value=40, step=5)
    st.caption("Gmail access is read-only. No email is sent, modified, or deleted.")

@st.cache_resource
def get_ollama_client(base_url: str, model: str) -> OllamaClient:
    return OllamaClient(base_url=base_url, model=model)


def connect_gmail() -> GmailService:
    return GmailService(credentials_path="credentials.json", token_path="token.json")


def display_email(email: dict[str, Any], index: int | None = None) -> None:
    prefix = f"{index}. " if index is not None else ""
    with st.expander(f"{prefix}{email.get('subject') or '(No subject)'} — {email.get('sender') or 'Unknown sender'}"):
        st.write(f"**Received:** {email.get('received_at', 'Unknown time')}")
        st.write(f"**Importance score:** {email.get('importance_score', 'Not scored')}/100")
        if email.get("importance_reason"):
            st.write(f"**Why it matters:** {email['importance_reason']}")
        if email.get("summary"):
            st.write(f"**Summary:** {email['summary']}")
        if email.get("action_items"):
            st.write("**Possible action items:**")
            for action in email["action_items"]:
                st.markdown(f"- {action}")
        if email.get("snippet"):
            st.caption(email["snippet"])
        if email.get("gmail_url"):
            st.link_button("Open in Gmail", email["gmail_url"])

if "gmail_service" not in st.session_state:
    st.session_state.gmail_service = None
if "emails" not in st.session_state:
    st.session_state.emails = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🔐 Connect Gmail", type="primary", use_container_width=True):
        try:
            service = connect_gmail()
            st.session_state.gmail_service = service
            st.success("Gmail connected successfully.")
        except GmailSetupError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Gmail connection failed: {exc}")
with col2:
    st.info("First run: place your Google OAuth Desktop app file at `credentials.json`, then click Connect Gmail. A browser window will ask you to authorize read-only access.")

if st.session_state.gmail_service is not None:
    st.success("Gmail is connected for this session.")
    run = st.button("✨ Analyze today's emails", type="primary", use_container_width=True)
    if run:
        try:
            with st.spinner("Fetching today's messages..."):
                emails = st.session_state.gmail_service.fetch_today_emails(limit=max_emails, local_date=datetime.now(ZoneInfo("Asia/Kolkata")).date())
            if not emails:
                st.session_state.emails = []
                st.session_state.analysis = {"daily_summary": "No emails were found for today.", "top_emails": [], "action_items": []}
            else:
                with st.spinner(f"Analyzing {len(emails)} emails with the agent..."):
                    agent = GmailAgent(get_ollama_client(ollama_url, model_name))
                    analysis = agent.analyze(emails)
                st.session_state.emails = emails
                st.session_state.analysis = analysis
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption("Check that Ollama is running, the selected model is installed, and Gmail authorization completed.")

analysis = st.session_state.analysis
if analysis is not None:
    st.divider()
    emails = st.session_state.emails
    m1, m2, m3 = st.columns(3)
    m1.metric("Emails received today", len(emails))
    m2.metric("Important emails", sum(1 for e in emails if int(e.get("importance_score", 0)) >= 60))
    m3.metric("Possible action items", len(analysis.get("action_items", [])))
    st.subheader("Today's summary")
    st.write(analysis.get("daily_summary", "No summary available."))

    st.subheader("Top 5 important emails")
    top_emails = analysis.get("top_emails", [])
    if top_emails:
        by_id = {str(e.get("id")): e for e in emails}
        for i, item in enumerate(top_emails[:5], start=1):
            original = by_id.get(str(item.get("id")), item)
            merged = {**original, **item}
            display_email(merged, i)
    else:
        st.write("No top emails were selected.")

    st.subheader("Suggested action items")
    actions = analysis.get("action_items", [])
    if actions:
        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.write("No clear action items detected.")

    with st.expander("All emails analyzed"):
        for email in emails:
            display_email(email)

    st.download_button(
        "Download analysis as JSON",
        data=__import__("json").dumps({"analysis": analysis, "emails": emails}, indent=2, ensure_ascii=False),
        file_name=f"gmail_analysis_{datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat()}.json",
        mime="application/json",
    )

st.divider()
st.caption("Privacy: this app requests Gmail read-only access. When Ollama is used, email text is sent to your local Ollama service. Treat email content as untrusted input; the model is instructed not to follow instructions embedded inside emails.")
