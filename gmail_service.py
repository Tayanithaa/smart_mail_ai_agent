from __future__ import annotations

import base64
import os
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailSetupError(RuntimeError):
    """Raised when Gmail OAuth prerequisites are missing or invalid."""


def _decode_body(data: str | None) -> str:
    if not data:
        return ""
    try:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    return {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}


def _plain_text(payload: dict[str, Any]) -> str:
    mime = payload.get("mimeType", "")
    body = payload.get("body", {}).get("data")
    if mime == "text/plain" and body:
        return _decode_body(body)
    parts = payload.get("parts", []) or []
    # Prefer plain text, including nested multipart structures.
    for part in parts:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return _decode_body(part["body"]["data"])
    for part in parts:
        nested = _plain_text(part)
        if nested:
            return nested
    if body:
        return _decode_body(body)
    return ""


class GmailService:
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.service = self._authenticate()

    def _authenticate(self):
        if not self.credentials_path.exists():
            raise GmailSetupError(
                "Missing credentials.json. In Google Cloud Console, enable Gmail API, create an OAuth client ID "
                "for a Desktop app, download its JSON file, and save it as credentials.json in this project folder."
            )
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise GmailSetupError("Google API packages are missing. Run: pip install -r requirements.txt") from exc

        creds = None
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            except Exception:
                creds = None
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
                creds = flow.run_local_server(port=0, open_browser=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    def fetch_today_emails(self, limit: int = 40, local_date: date | None = None, timezone_name: str = "Asia/Kolkata") -> list[dict[str, Any]]:
        """Fetch messages received during the local calendar day using Gmail's epoch query syntax."""
        local_date = local_date or datetime.now(ZoneInfo(timezone_name)).date()
        tz = ZoneInfo(timezone_name)
        start = datetime.combine(local_date, time.min, tzinfo=tz)
        end = start + timedelta(days=1)
        query = f"after:{int(start.timestamp())} before:{int(end.timestamp())}"
        response = self.service.users().messages().list(userId="me", q=query, maxResults=min(max(limit, 1), 500)).execute()
        messages = response.get("messages", [])
        results: list[dict[str, Any]] = []
        for item in messages:
            msg = self.service.users().messages().get(userId="me", id=item["id"], format="full").execute()
            payload = msg.get("payload", {})
            headers = _headers(payload)
            raw_date = headers.get("date", "")
            received_at = "Unknown"
            try:
                parsed = parsedate_to_datetime(raw_date)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                received_at = parsed.astimezone(tz).strftime("%Y-%m-%d %I:%M %p %Z")
            except (TypeError, ValueError, OverflowError):
                pass
            body = _plain_text(payload).strip()
            if not body:
                body = msg.get("snippet", "")
            results.append({
                "id": msg.get("id", item["id"]),
                "thread_id": msg.get("threadId", ""),
                "sender": headers.get("from", "Unknown sender"),
                "to": headers.get("to", ""),
                "subject": headers.get("subject", "(No subject)"),
                "received_at": received_at,
                "internal_date": int(msg.get("internalDate", 0)),
                "snippet": msg.get("snippet", ""),
                "body": body[:12000],
                "gmail_url": f"https://mail.google.com/mail/u/0/#all/{msg.get('id', item['id'])}",
            })
        results.sort(key=lambda x: x.get("internal_date", 0), reverse=True)
        return results
