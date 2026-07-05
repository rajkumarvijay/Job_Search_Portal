from sqlalchemy import text
from .database import engine, Base
from . import models  # noqa: F401 — ensures models are registered

EXPECTED_EMBEDDING_DIM = 384


async def init_db():
    async with engine.begin() as conn:
        # Enable pgvector extension (no-op if already installed)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # If job_embeddings exists with wrong vector dimension, drop and recreate.
        # This handles the migration from 768-dim (all-mpnet-base-v2) to
        # 384-dim (all-MiniLM-L6-v2) which uses 5x less RAM on free-tier hosting.
        try:
            result = await conn.execute(text("""
                SELECT atttypmod
                FROM pg_attribute
                JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
                WHERE pg_class.relname = 'job_embeddings'
                  AND pg_attribute.attname = 'embedding'
                  AND pg_attribute.attnum > 0
            """))
            row = result.fetchone()
            if row is not None and row[0] != EXPECTED_EMBEDDING_DIM:
                await conn.execute(text("DROP TABLE IF EXISTS job_embeddings CASCADE"))
        except Exception:
            pass

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
