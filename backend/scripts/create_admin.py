"""Create (or update) a PredictIQ user — the only way to provision an
account, since there is no self-registration endpoint (see
app/routers/auth.py). Run once after migrations to bootstrap the first
administrator; safe to re-run for additional users or to reset a
password.

    cd backend && python -m scripts.create_admin --email you@example.com --admin

Password is never accepted as a bare CLI argument here (it would land in
shell history / process listings) — it's always prompted for via
getpass, which doesn't echo to the terminal.

Docker:
    docker compose run --rm backend python -m scripts.create_admin --email you@example.com --admin
"""

import argparse
import getpass
import sys

from app.core.security import hash_password
from db.models import User
from db.session import SessionLocal

_MIN_PASSWORD_LENGTH = 8


def _prompt_for_password() -> str:
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords did not match.", file=sys.stderr)
        raise SystemExit(1)
    return password


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--email", required=True, help="Login email (case-insensitive, unique).")
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Grant administrator privileges (default: regular user).",
    )
    args = parser.parse_args()

    email = args.email.strip().lower()
    password = _prompt_for_password()
    if len(password) < _MIN_PASSWORD_LENGTH:
        print(f"Password must be at least {_MIN_PASSWORD_LENGTH} characters.", file=sys.stderr)
        raise SystemExit(1)

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        hashed = hash_password(password)
        if user is not None:
            user.hashed_password = hashed
            user.is_admin = args.admin or user.is_admin
            user.is_active = True
            db.commit()
            print(f"Updated existing user {email} (admin={user.is_admin}).")
        else:
            user = User(email=email, hashed_password=hashed, is_admin=args.admin)
            db.add(user)
            db.commit()
            print(f"Created user {email} (admin={args.admin}).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
