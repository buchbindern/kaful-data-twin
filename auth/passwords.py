"""Password hashing (bcrypt). Never store or log plaintext."""
from __future__ import annotations

import bcrypt

# bcrypt truncates at 72 bytes; we reject longer to avoid silent truncation surprises.
_MAX_BYTES = 72


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    if len(pw) > _MAX_BYTES:
        raise ValueError("password too long")
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
