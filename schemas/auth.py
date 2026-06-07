from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional
import re


class SignupRequest(BaseModel):
    name:     str
    email:    str
    password: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required")
        return v

    @field_validator("email")
    @classmethod
    def email_valid(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email:    str
    password: str

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.strip().lower()


class UserOut(BaseModel):
    id:            int
    name:          str
    email:         str
    auth_provider: str = "email"
    avatar_url:    Optional[str] = None
    created_at:    datetime

    class Config:
        from_attributes = True


class GoogleAuthRequest(BaseModel):
    access_token: str   # Google OAuth2 access token obtained from frontend


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserOut
