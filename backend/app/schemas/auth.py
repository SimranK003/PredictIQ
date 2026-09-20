"""Pydantic models for the auth API. Never include hashed_password or
any password field in a response schema.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=256)


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    is_admin: bool
    created_at: datetime
    last_login_at: datetime | None

    model_config = {"from_attributes": True}
