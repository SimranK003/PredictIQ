"""Session storage and login-attempt rate limiting, both backed by the
Redis instance the stack already runs for Celery — no new infrastructure
for auth. Session data is small (user id/email/is_admin) and TTL'd by
Redis itself, so "secure expiration" falls out of Redis's own key
expiry rather than custom sweep logic.
"""

import json
import secrets
from functools import lru_cache

import redis

from app.core.config import get_settings

_SESSION_KEY_PREFIX = "auth:session:"
_LOGIN_ATTEMPTS_KEY_PREFIX = "auth:login_attempts:"


class SessionStore:
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int):
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    def create(self, *, user_id: str, email: str, is_admin: bool) -> str:
        token = secrets.token_urlsafe(32)
        payload = json.dumps({"user_id": user_id, "email": email, "is_admin": is_admin})
        self._redis.setex(f"{_SESSION_KEY_PREFIX}{token}", self._ttl_seconds, payload)
        return token

    def get(self, token: str) -> dict | None:
        raw = self._redis.get(f"{_SESSION_KEY_PREFIX}{token}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def delete(self, token: str) -> None:
        self._redis.delete(f"{_SESSION_KEY_PREFIX}{token}")


@lru_cache
def get_session_store() -> SessionStore:
    settings = get_settings()
    client = redis.from_url(settings.redis_url, decode_responses=True)
    return SessionStore(client, settings.session_ttl_seconds)


def _rate_limit_key(identifier: str) -> str:
    return f"{_LOGIN_ATTEMPTS_KEY_PREFIX}{identifier}"


def is_login_rate_limited(identifier: str) -> bool:
    settings = get_settings()
    client = get_session_store()._redis  # same connection, no need for a second client
    raw = client.get(_rate_limit_key(identifier))
    return raw is not None and int(raw) >= settings.login_rate_limit_max_attempts


def record_failed_login(identifier: str) -> None:
    settings = get_settings()
    client = get_session_store()._redis
    key = _rate_limit_key(identifier)
    # Fixed window: the first failure in a window starts the TTL: `NX`
    # (Redis 7+) only sets the expiry if the key doesn't already have
    # one, so later increments in the same window don't keep pushing
    # the window back forever.
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, settings.login_rate_limit_window_seconds, nx=True)
    pipe.execute()


def clear_login_rate_limit(identifier: str) -> None:
    client = get_session_store()._redis
    client.delete(_rate_limit_key(identifier))
