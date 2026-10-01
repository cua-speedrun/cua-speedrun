"""Encryption for user-supplied runtime values at rest.

Users paste their own Modal token pair and reusable or evaluation-specific
environment variables. Each value is Fernet-encrypted with a key derived from
CS_SECRET_KEY before it touches the database, so a copied platform.db does not
expose plaintext. This is same-host symmetric encryption: it protects the
database file, not a fully compromised server. The web tier sees a value only
inside the request that saves it; only the worker decrypts stored values,
immediately before a run.
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken


def _fernet() -> Fernet:
    secret = os.environ.get("CS_SECRET_KEY", "dev-only-not-secret")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt(token: str) -> str | None:
    """None when the value cannot be decrypted (CS_SECRET_KEY changed)."""
    try:
        return _fernet().decrypt(token.encode()).decode()
    except (InvalidToken, ValueError):
        return None
