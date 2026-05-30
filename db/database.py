from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import os
import sys

# ── Resolve DATABASE_URL ─────────────────────────────────────────────────────
# Prefer the private Railway internal URL (postgres.railway.internal) — it
# connects directly inside Railway's private network without SSL or proxy.
# Fall back to DATABASE_URL (public proxy) if private URL is not set.
_raw_url = (
    os.getenv("DATABASE_PRIVATE_URL")   # Railway internal — best option
    or os.getenv("DATABASE_URL", "")    # Public proxy fallback
)

if not _raw_url:
    print("ERROR: No database URL found.", file=sys.stderr)
    print("In Railway → backend service → Variables → Add Reference → Postgres → DATABASE_PRIVATE_URL", file=sys.stderr)
    sys.exit(1)

# ── Fix scheme for asyncpg ───────────────────────────────────────────────────
if _raw_url.startswith("postgresql+asyncpg://"):
    _fixed = _raw_url
elif _raw_url.startswith("postgresql://"):
    _fixed = "postgresql+asyncpg://" + _raw_url[len("postgresql://"):]
elif _raw_url.startswith("postgres://"):
    _fixed = "postgresql+asyncpg://" + _raw_url[len("postgres://"):]
else:
    print(f"ERROR: Unrecognised DATABASE_URL scheme: {_raw_url[:40]}", file=sys.stderr)
    sys.exit(1)

# ── Strip ?sslmode from URL query string ─────────────────────────────────────
_parsed = urlparse(_fixed)
_qs = parse_qs(_parsed.query, keep_blank_values=True)
_qs.pop("sslmode", None)
_qs.pop("sslrootcert", None)
DATABASE_URL = urlunparse(_parsed._replace(query=urlencode(_qs, doseq=True)))

# ── Detect connection type ───────────────────────────────────────────────────
# Private Railway URL (railway.internal) — plain TCP, no SSL required
# Public proxy URL (rlwy.net)            — SSL required at protocol level
_is_private = "railway.internal" in DATABASE_URL
_use_ssl = not _is_private  # only enable SSL for public proxy connections

print(
    f"DB: {'private internal' if _is_private else 'public proxy'} "
    f"| ssl={'disabled' if not _use_ssl else 'enabled'}",
    file=sys.stderr,
)

# ── SSL context for public proxy (fallback) ──────────────────────────────────
_connect_args: dict = {}
if _use_ssl:
    import ssl as _ssl
    _ssl_ctx = _ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = _ssl.CERT_NONE
    _connect_args["ssl"] = _ssl_ctx
else:
    _connect_args["ssl"] = False   # private network — no SSL needed

# ── Create engine ────────────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
