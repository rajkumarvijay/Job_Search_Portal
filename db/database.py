from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import os
import sys

# ── Read DATABASE_URL from environment ──────────────────────────────────────
_raw_url = os.getenv("DATABASE_URL", "")

if not _raw_url:
    print("ERROR: DATABASE_URL environment variable is not set.", file=sys.stderr)
    print("Set it in Railway → backend service → Variables → Add Reference → Postgres → DATABASE_URL", file=sys.stderr)
    sys.exit(1)

# ── Fix scheme: Railway gives postgresql:// or postgres://
#    asyncpg requires  postgresql+asyncpg://
# ─────────────────────────────────────────────────────────
if _raw_url.startswith("postgresql+asyncpg://"):
    _fixed = _raw_url
elif _raw_url.startswith("postgresql://"):
    _fixed = "postgresql+asyncpg://" + _raw_url[len("postgresql://"):]
elif _raw_url.startswith("postgres://"):
    _fixed = "postgresql+asyncpg://" + _raw_url[len("postgres://"):]
else:
    print(f"ERROR: Unrecognised DATABASE_URL scheme: {_raw_url[:40]}", file=sys.stderr)
    sys.exit(1)

# ── Strip ?sslmode from the URL ──────────────────────────────────────────────
# Railway's public proxy terminates TLS at the NETWORK layer.
# asyncpg must NOT attempt a PostgreSQL-level SSL upgrade (SSLRequest packet)
# because the proxy resets the connection when it receives one.
# We strip sslmode from the URL and pass ssl=False to connect_args so asyncpg
# connects over plain TCP — the proxy's TLS handles encryption transparently.
_parsed = urlparse(_fixed)
_qs = parse_qs(_parsed.query, keep_blank_values=True)
_qs.pop("sslmode", None)        # remove sslmode=require / verify-full / etc.
_qs.pop("sslrootcert", None)    # remove any cert path
DATABASE_URL = urlunparse(_parsed._replace(query=urlencode(_qs, doseq=True)))

# ── Create async engine ──────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,     # validates connections before use
    pool_recycle=1800,      # recycle connections every 30 min
    connect_args={
        "ssl": False,       # disable asyncpg SSL — Railway proxy handles TLS
    },
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
