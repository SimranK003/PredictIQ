"""Auth API integration tests: login/logout/me, session invalidation,
protected-endpoint enforcement, admin-only authorization, and login rate
limiting — all against the real app, real Postgres, and real Redis (the
session store and rate limiter are Redis-backed, see app/services/auth.py).
"""

import uuid

from app.core.config import get_settings
from app.services.auth import get_session_store


def _redis_client():
    return get_session_store()._redis


def test_login_with_correct_credentials_succeeds(anonymous_client, make_user):
    user, password = make_user(email="alice@test.local", password="a-real-password-1")
    resp = anonymous_client.post(
        "/auth/login", json={"email": user.email, "password": password}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == user.email
    assert body["is_admin"] is False
    assert "password" not in body
    assert "hashed_password" not in body
    # The session cookie must be set and marked HttpOnly.
    set_cookie = resp.headers.get("set-cookie", "")
    assert get_settings().session_cookie_name in set_cookie
    assert "httponly" in set_cookie.lower()


def test_login_with_wrong_password_is_rejected(anonymous_client, make_user):
    user, _ = make_user(email="bob@test.local", password="the-real-password")
    resp = anonymous_client.post(
        "/auth/login", json={"email": user.email, "password": "wrong-password"}
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Incorrect email or password."


def test_login_with_nonexistent_email_is_rejected_with_the_same_message(anonymous_client):
    resp = anonymous_client.post(
        "/auth/login",
        json={"email": "nobody-like-this-exists@test.local", "password": "whatever123"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Incorrect email or password."


def test_login_wrong_password_and_login_nonexistent_email_return_identical_error_bodies(
    anonymous_client, make_user
):
    """Guards against user enumeration via a differing error message."""
    user, _ = make_user(email="carol@test.local", password="the-real-password")

    wrong_password_resp = anonymous_client.post(
        "/auth/login", json={"email": user.email, "password": "not-the-real-password"}
    )
    nonexistent_resp = anonymous_client.post(
        "/auth/login",
        json={"email": "definitely-not-registered@test.local", "password": "whatever123"},
    )
    assert wrong_password_resp.status_code == nonexistent_resp.status_code == 401
    assert wrong_password_resp.json()["detail"] == nonexistent_resp.json()["detail"]


def test_login_for_a_deactivated_user_is_rejected(anonymous_client, make_user):
    user, password = make_user(
        email="deactivated@test.local", password="a-real-password-1", is_admin=False
    )
    user.is_active = False
    resp = anonymous_client.post(
        "/auth/login", json={"email": user.email, "password": password}
    )
    assert resp.status_code == 401


def test_password_is_never_echoed_back_in_a_validation_error(anonymous_client):
    # password as the wrong type trips Pydantic validation and, without
    # the redaction in app/main.py's RequestValidationError handler,
    # would otherwise echo it back in the 422 body.
    secret_value = "super-secret-attempted-password-should-not-leak"
    resp = anonymous_client.post(
        "/auth/login",
        json={"email": "someone@test.local", "password": {"nested": secret_value}},
    )
    assert resp.status_code == 422
    assert secret_value not in resp.text


def test_me_without_a_session_is_401(anonymous_client):
    resp = anonymous_client.get("/auth/me")
    assert resp.status_code == 401


def test_me_with_a_valid_session_returns_the_current_user(anonymous_client, make_user):
    user, password = make_user(email="dave@test.local", password="a-real-password-1")
    anonymous_client.post("/auth/login", json={"email": user.email, "password": password})
    resp = anonymous_client.get("/auth/me")
    assert resp.status_code == 200
    assert resp.json()["email"] == user.email


def test_logout_invalidates_the_session(anonymous_client, make_user):
    user, password = make_user(email="erin@test.local", password="a-real-password-1")
    anonymous_client.post("/auth/login", json={"email": user.email, "password": password})
    assert anonymous_client.get("/auth/me").status_code == 200

    logout_resp = anonymous_client.post("/auth/logout")
    assert logout_resp.status_code == 204

    # Same cookie jar, same (now-deleted) session token: must be rejected.
    resp = anonymous_client.get("/auth/me")
    assert resp.status_code == 401


def test_logout_actually_deletes_the_redis_session_key(anonymous_client, make_user):
    user, password = make_user(email="frank@test.local", password="a-real-password-1")
    login_resp = anonymous_client.post(
        "/auth/login", json={"email": user.email, "password": password}
    )
    session_cookie = login_resp.cookies.get(get_settings().session_cookie_name)
    assert session_cookie is not None

    from app.core.security import unsign_token

    token = unsign_token(session_cookie)
    assert get_session_store().get(token) is not None

    anonymous_client.post("/auth/logout")
    assert get_session_store().get(token) is None


def test_expired_or_forged_session_cookie_is_rejected(anonymous_client):
    settings = get_settings()
    anonymous_client.cookies.set(settings.session_cookie_name, "forged.notarealsignature")
    resp = anonymous_client.get("/auth/me")
    assert resp.status_code == 401


# --- Backend-enforced protection: unauthenticated + non-admin rejection ---


def test_unauthenticated_request_to_a_protected_endpoint_is_401(anonymous_client):
    for method, path, kwargs in [
        ("get", "/models", {}),
        ("get", "/datasets", {}),
        ("get", "/jobs", {}),
        ("get", "/predictions", {}),
        ("get", "/monitoring/summary", {}),
        ("post", "/predict", {"json": {}}),
    ]:
        resp = getattr(anonymous_client, method)(path, **kwargs)
        assert resp.status_code == 401, (
            f"{method.upper()} {path} should require auth, got {resp.status_code}"
        )


def test_health_endpoint_stays_public(anonymous_client):
    assert anonymous_client.get("/health").status_code in (200, 503)


def test_metrics_endpoint_stays_public(anonymous_client):
    assert anonymous_client.get("/metrics").status_code == 200


def test_non_admin_authenticated_user_can_read_models(non_admin_client, make_model_version):
    from db.models import ModelStage

    make_model_version(stage=ModelStage.PRODUCTION)
    resp = non_admin_client.get("/models")
    assert resp.status_code == 200


def test_non_admin_user_cannot_promote_a_model(
    non_admin_client, make_model_version, disable_artifact_check
):
    candidate = make_model_version()
    resp = non_admin_client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    # 403 (not the auth layer at all) proves this is an authorization
    # rejection, not a promotion-policy one — require_admin runs before
    # promote_model is ever called.
    assert resp.status_code == 403


def test_non_admin_user_cannot_rollback(non_admin_client):
    resp = non_admin_client.post("/models/rollback")
    assert resp.status_code == 403


def test_admin_user_can_promote_a_model(client, make_model_version, disable_artifact_check):
    candidate = make_model_version()
    resp = client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    assert resp.status_code == 200


def test_unauthenticated_user_cannot_promote_a_model(anonymous_client, make_model_version):
    candidate = make_model_version()
    resp = anonymous_client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    assert resp.status_code == 401


def test_promotion_audit_trail_records_the_acting_admins_email(
    client, db_session, make_model_version, disable_artifact_check
):
    from db.models import LifecycleAction, ModelLifecycleEvent

    candidate = make_model_version()
    resp = client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    assert resp.status_code == 200

    event = (
        db_session.query(ModelLifecycleEvent)
        .filter(
            ModelLifecycleEvent.model_version_id == candidate.id,
            ModelLifecycleEvent.action == LifecycleAction.PROMOTED,
        )
        .one()
    )
    assert event.triggered_by == "admin@test.local"


# --- Rate limiting ---


def test_repeated_failed_logins_are_rate_limited(anonymous_client, make_user):
    email = f"ratelimited-{uuid.uuid4().hex[:8]}@test.local"
    user, password = make_user(email=email, password="the-real-one")
    settings = get_settings()

    try:
        for _ in range(settings.login_rate_limit_max_attempts):
            resp = anonymous_client.post(
                "/auth/login", json={"email": user.email, "password": "wrong"}
            )
            assert resp.status_code == 401

        blocked_resp = anonymous_client.post(
            "/auth/login", json={"email": user.email, "password": "wrong"}
        )
        assert blocked_resp.status_code == 429

        # Even the *correct* password is blocked while rate-limited —
        # the limiter gates on attempt volume, not just failures.
        still_blocked = anonymous_client.post(
            "/auth/login", json={"email": user.email, "password": password}
        )
        assert still_blocked.status_code == 429
    finally:
        from app.services.auth import _rate_limit_key

        client_ip = "testclient"
        _redis_client().delete(_rate_limit_key(f"{user.email}:{client_ip}"))


def test_successful_login_clears_the_rate_limit_counter(anonymous_client, make_user):
    email = f"resets-{uuid.uuid4().hex[:8]}@test.local"
    user, password = make_user(email=email, password="the-real-one")

    anonymous_client.post("/auth/login", json={"email": user.email, "password": "wrong"})
    anonymous_client.post("/auth/login", json={"email": user.email, "password": "wrong"})
    ok_resp = anonymous_client.post(
        "/auth/login", json={"email": user.email, "password": password}
    )
    assert ok_resp.status_code == 200

    from app.services.auth import _rate_limit_key

    assert _redis_client().get(_rate_limit_key(f"{user.email}:testclient")) is None


# --- X-Forwarded-For trust (Phase 10 — see app/core/config.py's
# trust_proxy_headers and app/routers/auth.py's _client_identifier) ---


def test_x_forwarded_for_is_ignored_by_default(anonymous_client, make_user):
    """Without TRUST_PROXY_HEADERS, two "different" clients (per a
    spoofed X-Forwarded-For) must still share one rate-limit bucket —
    proves the header is ignored, not honored, when there's no trusted
    proxy in front (the default, matching native dev/Docker Compose).
    """
    from app.services.auth import _rate_limit_key

    email = f"noproxytrust-{uuid.uuid4().hex[:8]}@test.local"
    user, _ = make_user(email=email, password="the-real-one")
    settings = get_settings()

    try:
        for i in range(settings.login_rate_limit_max_attempts):
            resp = anonymous_client.post(
                "/auth/login",
                json={"email": user.email, "password": "wrong"},
                headers={"X-Forwarded-For": f"203.0.113.{i}"},
            )
            assert resp.status_code == 401

        # A "new" forwarded IP doesn't grant a fresh bucket — real
        # client.host (the httpx test transport's own fixed peer
        # address) is what's actually being keyed on.
        blocked_resp = anonymous_client.post(
            "/auth/login",
            json={"email": user.email, "password": "wrong"},
            headers={"X-Forwarded-For": "203.0.113.250"},
        )
        assert blocked_resp.status_code == 429
    finally:
        _redis_client().delete(_rate_limit_key(f"{user.email}:testclient"))


def test_x_forwarded_for_is_honored_when_trust_proxy_headers_is_enabled(
    anonymous_client, make_user, monkeypatch
):
    """With TRUST_PROXY_HEADERS=true (render.yaml's production setting),
    each distinct X-Forwarded-For value gets its own rate-limit bucket —
    proves per-real-client limiting actually works behind a trusted
    proxy, instead of every user sharing one bucket keyed on the
    proxy's own address.
    """
    from app.services.auth import _rate_limit_key

    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    get_settings.cache_clear()

    email = f"proxytrust-{uuid.uuid4().hex[:8]}@test.local"
    user, password = make_user(email=email, password="the-real-one")
    settings = get_settings()

    try:
        for _ in range(settings.login_rate_limit_max_attempts):
            resp = anonymous_client.post(
                "/auth/login",
                json={"email": user.email, "password": "wrong"},
                headers={"X-Forwarded-For": "198.51.100.10"},
            )
            assert resp.status_code == 401

        blocked = anonymous_client.post(
            "/auth/login",
            json={"email": user.email, "password": "wrong"},
            headers={"X-Forwarded-For": "198.51.100.10"},
        )
        assert blocked.status_code == 429

        # A different forwarded client IP is a genuinely separate
        # bucket and can still log in.
        other_client_resp = anonymous_client.post(
            "/auth/login",
            json={"email": user.email, "password": password},
            headers={"X-Forwarded-For": "198.51.100.20"},
        )
        assert other_client_resp.status_code == 200
    finally:
        _redis_client().delete(_rate_limit_key(f"{user.email}:198.51.100.10"))
        _redis_client().delete(_rate_limit_key(f"{user.email}:198.51.100.20"))
        monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
        get_settings.cache_clear()
