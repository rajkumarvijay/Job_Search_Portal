import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from jose import JWTError

from db.database import get_db
from db.models import User
from schemas.auth import SignupRequest, LoginRequest, GoogleAuthRequest, TokenResponse, UserOut
from services.auth_service import hash_password, verify_password, create_access_token, decode_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)


# ── Dependency: resolve JWT → User ────────────────────────────────────────────
async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db:    AsyncSession                  = Depends(get_db),
) -> User:
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = decode_token(creds.credentials)
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account not found")
    return user


# ── POST /api/v1/auth/signup ──────────────────────────────────────────────────
@router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(payload: SignupRequest, db: AsyncSession = Depends(get_db)):
    # Duplicate email check
    existing = (await db.execute(
        select(User).where(User.email == payload.email)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user = User(
        name            = payload.name,
        email           = payload.email,
        hashed_password = hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info(f"New user registered: {user.email}")

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user=UserOut.model_validate(user),
    )


# ── POST /api/v1/auth/login ───────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.email == payload.email, User.is_active == True)
    )
    user = result.scalar_one_or_none()

    if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token(user.id, user.email)
    logger.info(f"User logged in: {user.email}")
    return TokenResponse(
        access_token=token,
        user=UserOut.model_validate(user),
    )


# ── POST /api/v1/auth/google ──────────────────────────────────────────────────
@router.post("/google", response_model=TokenResponse)
async def google_auth(payload: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    """
    Verify a Google OAuth2 access-token obtained by the frontend via
    @react-oauth/google, fetch user info from Google's userinfo endpoint,
    then find-or-create the local user account and return a JWT.
    """
    # 1. Fetch user profile from Google
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {payload.access_token}"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid Google token. Please try again.")

    g = resp.json()
    google_id  = g.get("sub", "")
    email      = g.get("email", "").lower()
    name       = g.get("name") or email.split("@")[0]
    avatar_url = g.get("picture")

    if not email:
        raise HTTPException(status_code=400, detail="Google did not return an email address.")

    # 2. Find existing user by Google ID or email
    result = await db.execute(
        select(User).where(
            or_(User.google_id == google_id, User.email == email),
            User.is_active == True,
        )
    )
    user = result.scalar_one_or_none()

    if user:
        # Link Google ID to an existing email-only account if needed
        if not user.google_id:
            user.google_id     = google_id
            user.auth_provider = "google"
        if avatar_url:
            user.avatar_url = avatar_url
        await db.commit()
        await db.refresh(user)
    else:
        # Create brand-new Google account (no password)
        user = User(
            name          = name,
            email         = email,
            google_id     = google_id,
            hashed_password = None,
            auth_provider = "google",
            avatar_url    = avatar_url,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info(f"New Google user: {email}")

    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


# ── GET /api/v1/auth/me ───────────────────────────────────────────────────────
@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)
