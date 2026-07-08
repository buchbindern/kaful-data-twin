"""Opaque server-side session tokens."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

SESSION_TTL = timedelta(days=14)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def session_expiry(now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)) + SESSION_TTL
