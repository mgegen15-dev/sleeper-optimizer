"""Send via Gmail SMTP (SSL 465). Credentials come only from environment variables
(GMAIL_USER, GMAIL_APP_PASSWORD) — never from files in the repo."""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


def send(cfg: dict, subject: str, text: str, html: str) -> None:
    ec = cfg["email"]
    user = os.environ.get(ec["user_env"])
    pwd = os.environ.get(ec["password_env"])
    to = os.environ.get("EMAIL_TO") or ec.get("to") or user
    if not user or not pwd:
        raise RuntimeError(f"missing env {ec['user_env']} / {ec['password_env']}")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL(ec["smtp_host"], ec["smtp_port"], context=ssl.create_default_context(), timeout=30) as s:
        s.login(user, pwd)
        s.send_message(msg)
