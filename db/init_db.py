from .database import engine, Base
from . import models  # noqa: F401 — ensures models are registered


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
