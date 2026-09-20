"""Password hashing, session-cookie signing, and the FastAPI auth
dependencies (`get_current_user`, `require_admin`) every protected
router uses.

Session tokens themselves are opaque random values stored server-side in
Redis (see app/services/auth.py) — nothing about a user's identity is
encoded in the cookie. `sign_token`/`unsign_token` HMAC-sign the cookie
*value* with SECRET_KEY purely so a tampered/garbage cookie is rejected
locally (cheap, no I/O) before it ever reaches Redis or the database —
the security boundary that actually matters is still "does this token
exist as a live session in Redis," not the signature.
"""

import hashlib
import hmac
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from db.models import User
from db.session import get_db

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Argon2id hash (argon2-cffi's default profile) — the current
    OWASP-recommended algorithm for password storage. Never called with,
    and never returns, the raw password.
    """
    return _hasher.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Constant-time-safe verification (argon2-cffi handles this
    internally). Returns False on any mismatch or malformed hash rather
    than raising, so callers have one boolean to branch on.
    """
    try:
        return _hasher.verify(hashed_password, password)
    except VerifyMismatchError:
        return False
    except Exception:
        # A hash from a different/older scheme, corrupted data, etc —
        # fail closed, never treat an unreadable hash as a match.
        return False


def sign_token(token: str) -> str:
    settings = get_settings()
    signature = hmac.new(settings.secret_key.encode(), token.encode(), hashlib.sha256).hexdigest()
    return f"{token}.{signature}"


def unsign_token(signed_value: str) -> str | None:
    settings = get_settings()
    token, _, signature = signed_value.rpartition(".")
    if not token or not signature:
        return None
    expected = hmac.new(settings.secret_key.encode(), token.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    return token


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Required by every protected route. Re-reads the user from the
    database on every call (not just the Redis-cached session payload)
    so that deactivating a user or revoking admin takes effect
    immediately — not only after their session expires — at the cost of
    one indexed primary-key lookup per request.
    """
    from app.services.auth import get_session_store  # local import: avoids a circular import

    settings = get_settings()
    raw_cookie = request.cookies.get(settings.session_cookie_name)
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated.",
        headers={"WWW-Authenticate": "Cookie"},
    )
    if not raw_cookie:
        raise unauthorized

    token = unsign_token(raw_cookie)
    if token is None:
        raise unauthorized

    session_data = get_session_store().get(token)
    if session_data is None:
        raise unauthorized

    try:
        user_id = uuid.UUID(session_data["user_id"])
    except (KeyError, ValueError, TypeError) as exc:
        raise unauthorized from exc

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise unauthorized

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required.",
        )
    return current_user
