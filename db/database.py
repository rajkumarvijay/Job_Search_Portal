from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
import os
import ssl
import sys

# ── Read DATABASE_URL from environment ──────────────────────────────────────
_raw_url = os.getenv("DATABASE_URL", "")

if not _raw_url:
    print("ERROR: DATABASE_URL environment variable is not set.", file=sys.stderr)
    print("Set it in Railway → your backend service → Variables tab.", file=sys.stderr)
    sys.exit(1)

# Railway / some providers give postgresql:// or postgres://
# SQLAlchemy async requires postgresql+asyncpg://
if _raw_url.startswith("postgresql://"):
    DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgres://"):
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgresql+asyncpg://"):
    DATABASE_URL = _raw_url  # already correct
else:
    print(f"ERROR: Unsupported DATABASE_URL scheme: {_raw_url[:30]}...", file=sys.stderr)
    print("Expected postgresql:// or postgres:// URL.", file=sys.stderr)
    sys.exit(1)

# ── SSL context — Railway PostgreSQL requires SSL ────────────────────────────
# Use a permissive SSL context that does not verify the server certificate.
# This is safe for Railway's internal private networking.
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# ── Create async engine ──────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,       # drops stale connections automatically
    pool_recycle=1800,        # recycle connections every 30 minutes
    connect_args={"ssl": _ssl_ctx},  # required by Railway PostgreSQL
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
