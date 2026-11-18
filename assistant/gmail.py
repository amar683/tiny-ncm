"""Gmail, read-only. The assistant can list and read your emails; it can never send, delete or change anything.

One-time setup (about 5 minutes, free):
  1. https://console.cloud.google.com -> new project -> "APIs & Services" -> enable "Gmail API".
  2. "OAuth consent screen": External, add your own address as a test user.
  3. "Credentials" -> "Create credentials" -> "OAuth client ID" -> "Desktop app" -> download the JSON
     and save it as ~/.tiny-ncm/credentials.json
  4. python chat.py --connect-gmail   (opens your browser once; the token is saved in ~/.tiny-ncm/token.json)
Everything stays on this computer; to disconnect, delete ~/.tiny-ncm/token.json.
"""
import base64
import html
import re
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
HOME = Path.home() / ".tiny-ncm"
CREDS, TOKEN = HOME / "credentials.json", HOME / "token.json"


def connect(interactive=False):
    """Returns a Gmail object, or None if not set up. interactive=True opens the browser sign-in."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES) if TOKEN.exists() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN.write_text(creds.to_json())
    if not (creds and creds.valid):
        if not interactive:
            return None
        if not CREDS.exists():
            raise SystemExit(f"Missing {CREDS}. See the setup steps at the top of assistant/gmail.py.")
        from google_auth_oauthlib.flow import InstalledAppFlow
        creds = InstalledAppFlow.from_client_secrets_file(str(CREDS), SCOPES).run_local_server(port=0)
        HOME.mkdir(exist_ok=True)
        TOKEN.write_text(creds.to_json())
        TOKEN.chmod(0o600)
    from googleapiclient.discovery import build
    return Gmail(build("gmail", "v1", credentials=creds, cache_discovery=False))


def _header(msg, name):
    return next((h["value"] for h in msg["payload"].get("headers", []) if h["name"].lower() == name), "")


def _sender(s):
    m = re.match(r'\s*"?([^"<]+?)"?\s*<', s)
    return m.group(1).strip() if m else s


def _body(payload):
    """Plain text of a message (prefers text/plain, falls back to stripped HTML)."""
    parts, stack = {}, [payload]
    while stack:
        p = stack.pop()
        stack += p.get("parts", [])
        data = p.get("body", {}).get("data")
        if data and p.get("mimeType") in ("text/plain", "text/html"):
            parts.setdefault(p["mimeType"], base64.urlsafe_b64decode(data).decode("utf-8", "replace"))
    text = parts.get("text/plain")
    if text is None and "text/html" in parts:
        text = html.unescape(re.sub(r"<(script|style).*?</\1>|<[^>]+>", " ", parts["text/html"], flags=re.S))
    return " ".join((text or "").split())


class Gmail:
    def __init__(self, service):
        self.s = service.users()

    def list(self, query="", n=5):
        ids = self.s.messages().list(userId="me", q=query, maxResults=max(1, min(int(n), 10))).execute().get("messages", [])
        out = []
        for m in ids:
            msg = self.s.messages().get(userId="me", id=m["id"], format="full").execute()
            out.append({"id": m["id"], "from": _sender(_header(msg, "from")), "subject": _header(msg, "subject"),
                        "snippet": _body(msg["payload"])[:500] or html.unescape(msg.get("snippet", ""))})
        return out

    def read(self, id):
        msg = self.s.messages().get(userId="me", id=id, format="full").execute()
        return {"from": _sender(_header(msg, "from")), "subject": _header(msg, "subject"), "body": _body(msg["payload"])[:2000]}
