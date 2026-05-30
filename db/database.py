from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
import os

# Railway provides postgresql:// — SQLAlchemy async requires postgresql+asyncpg://
# This handles both formats automatically.
_raw_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./job_portal.db")

if _raw_url.startswith("postgresql://"):
    DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgres://"):
    # Some providers use postgres:// (non-standard)
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = _raw_url  # sqlite+aiosqlite:// for local dev

# PostgreSQL needs pool settings; SQLite does not support them
_is_postgres = DATABASE_URL.startswith("postgresql")
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    **({
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
    } if _is_postgres else {
        "connect_args": {"check_same_thread": False},
    })
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
