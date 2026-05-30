import hashlib
import json
import logging
from datetime import datetime, timedelta
from cachetools import TTLCache
from sqlalchemy import select
from db.database import AsyncSessionLocal
from db.models import TrendingCache

logger = logging.getLogger(__name__)

_memory_cache: TTLCache = TTLCache(maxsize=256, ttl=300)  # 5-minute in-memory cache


def _make_key(*parts) -> str:
    raw = ":".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()


def get_from_memory(key: str):
    return _memory_cache.get(key)


def set_in_memory(key: str, value):
    _memory_cache[key] = value


async def get_from_db(cache_key: str):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(TrendingCache).where(TrendingCache.cache_key == cache_key)
        )
        row = result.scalar_one_or_none()
        if row and row.expires_at > datetime.utcnow():
            return json.loads(row.data)
    return None


async def set_in_db(cache_key: str, data, ttl_hours: int = 6):
    async with AsyncSessionLocal() as db:
        expires = datetime.utcnow() + timedelta(hours=ttl_hours)
        existing = await db.execute(
            select(TrendingCache).where(TrendingCache.cache_key == cache_key)
        )
        row = existing.scalar_one_or_none()
        if row:
            row.data = json.dumps(data)
            row.expires_at = expires
            row.updated_at = datetime.utcnow()
        else:
            db.add(TrendingCache(
                cache_key=cache_key,
                data=json.dumps(data),
                expires_at=expires,
            ))
        await db.commit()


def make_search_key(query: str, location: str, platforms: list, results: int) -> str:
    return _make_key(query.lower().strip(), location.lower().strip(), sorted(platforms), results)
