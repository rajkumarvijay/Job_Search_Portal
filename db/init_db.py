from sqlalchemy import text
from .database import engine, Base
from . import models  # noqa: F401 — ensures models are registered


async def init_db():
    async with engine.begin() as conn:
        # Enable pgvector extension (no-op if already installed)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create all tables that don't exist yet
        await conn.run_sync(Base.metadata.create_all)

        # ── Live migration for the `users` table ──────────────────────────────
        # These statements are safe to run on both fresh and existing deployments:
        #   • ADD COLUMN IF NOT EXISTS   — no-op if already present
        #   • DROP NOT NULL              — no-op if already nullable
        try:
            await conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id    VARCHAR(128) UNIQUE"
            ))
            await conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(32)  NOT NULL DEFAULT 'email'"
            ))
            await conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url   TEXT"
            ))
            await conn.execute(text(
                "ALTER TABLE users ALTER COLUMN hashed_password DROP NOT NULL"
            ))
        except Exception:
            # Table may not exist yet (first run) — create_all handled it above
            pass
