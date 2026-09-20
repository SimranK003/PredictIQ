"""Login/logout/session-identity endpoints.

Deliberately no self-registration endpoint here — users are provisioned
via `backend/scripts/create_admin.py` (see README). That's a scope
decision, not an oversight: this phase asks for login/logout and
admin-vs-user authorization, not open account creation.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import (
    get_current_user,
    hash_password,
    sign_token,
    unsign_token,
    verify_password,
)
from app.schemas.auth import LoginRequest, UserOut
from app.services.auth import (
    clear_login_rate_limit,
    get_session_store,
    is_login_rate_limited,
    record_failed_login,
)
from db.models import User
from db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

# Generic on purpose: never reveal whether the email exists, only that
# the (email, password) pair was invalid.
_INVALID_CREDENTIALS_DETAIL = "Incorrect email or password."


def _client_identifier(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    request.client.host is only the real peer address when nothing sits
    between the client and this process — true for native dev and
    Docker Compose, false for Render (Phase 10 production deployment),
    where every request is proxied through Render's own edge and
    request.client.host would just be that proxy's address for every
    client, collapsing the rate limiter to one shared bucket for
    everyone. settings.trust_proxy_headers (see app/core/config.py) is
    the explicit opt-in that switches to X-Forwarded-For's left-most
    entry instead (the original client — each proxy in the chain
    appends its own address to the right of it) — off by default
    because trusting that header without an actual trusted proxy in
    front would let any client forge it to dodge the limiter entirely.
    """
    settings = get_settings()
    if settings.trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# A real Argon2 hash of an unrelated fixed string, computed once at
# import time — used below so a login attempt against a nonexistent
# email still does real hashing work, instead of short-circuiting in a
# way that would make "no such user" measurably faster than "wrong
# password" (a timing side-channel for account enumeration).
_DUMMY_HASH = hash_password("dummy-constant-time-comparison-value")


@router.post("/login", response_model=UserOut)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> User:
    settings = get_settings()
    email = payload.email.strip().lower()
    identifier = f"{email}:{_client_identifier(request)}"

    if is_login_rate_limited(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again later.",
        )

    user = db.query(User).filter(User.email == email).first()
    # Always run verify_password, even when no user was found, so a
    # nonexistent-email request takes about as long as a wrong-password
    # one — avoids a cheap timing signal for account enumeration. The
    # dummy hash below is a real Argon2 hash of an unrelated fixed
    # string, not a shortcut that skips hashing work entirely.
    password_ok = verify_password(
        payload.password,
        user.hashed_password if user else _DUMMY_HASH,
    )

    if user is None or not user.is_active or not password_ok:
        record_failed_login(identifier)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=_INVALID_CREDENTIALS_DETAIL
        )

    clear_login_rate_limit(identifier)
    user.last_login_at = datetime.now(UTC)
    db.commit()
    db.refresh(user)

    token = get_session_store().create(
        user_id=str(user.id), email=user.email, is_admin=user.is_admin
    )
    response.set_cookie(
        key=settings.session_cookie_name,
        value=sign_token(token),
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
        path="/",
    )
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response) -> Response:
    settings = get_settings()
    raw_cookie = request.cookies.get(settings.session_cookie_name)
    if raw_cookie:
        token = unsign_token(raw_cookie)
        if token:
            get_session_store().delete(token)
    response.delete_cookie(settings.session_cookie_name, path="/")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
