"""
Semantic job search via pgvector + Google text-embedding-004.

Flow:
  store_job_embedding(job)  — called after each job is scraped
  semantic_search(query)    — called from the /jobs/semantic-search endpoint
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from sqlalchemy import text, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.models import JobEmbedding

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)

EMBEDDING_DIM = 768   # Google text-embedding-004


# ── Lazy genai initialisation (same pattern as gemini_service) ────────────────

_api_key: str = ""

def _get_api_key() -> str:
    global _api_key
    if _api_key:
        return _api_key
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set")
    _api_key = key
    return key


# ── Core embedding call using the SDK (handles URL/auth correctly) ────────────

_EMBED_MODELS = [
    "text-embedding-004",
    "gemini-embedding-exp-03-07",
    "embedding-001",
]
_VERSIONS = ["v1", "v1beta"]

def _embed_sync(content: str) -> list[float]:
    import httpx
    api_key = _get_api_key()
    last_err = None

    for version in _VERSIONS:
        for model in _EMBED_MODELS:
            for auth_style in ["header", "param"]:
                try:
                    url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:embedContent"
                    kwargs: dict = {
                        "json": {
                            "model": f"models/{model}",
                            "content": {"parts": [{"text": content[:8000]}]},
                        },
                        "timeout": 20,
                    }
                    if auth_style == "header":
                        kwargs["headers"] = {"x-goog-api-key": api_key}
                    else:
                        kwargs["params"] = {"key": api_key}

                    resp = httpx.post(url, **kwargs)
                    if resp.status_code == 200:
                        logger.info(f"[embedding] Working combo: {version}/{model} auth={auth_style}")
                        return resp.json()["embedding"]["values"]
                    last_err = f"{resp.status_code} {resp.text[:100]}"
                except Exception as e:
                    last_err = str(e)[:100]

    raise RuntimeError(f"All embedding combos failed. Last: {last_err}")


async def _embed(text_content: str) -> list[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _embed_sync, text_content)


def _job_text(job: dict) -> str:
    """Build a rich text representation of a job for embedding."""
    parts = [
        f"Job Title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
        f"Type: {job.get('job_type', '')}",
        f"Remote: {'Yes' if job.get('is_remote') else 'No'}",
        f"Description: {(job.get('description') or '')[:3000]}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())


# ── Public API ────────────────────────────────────────────────────────────────

async def store_job_embedding(job: dict) -> bool:
    """
    Embed a job dict and upsert into job_embeddings.
    Returns True on success, False on failure (non-fatal).
    """
    job_id = job.get("job_id") or job.get("id")
    if not job_id:
        return False
    try:
        vector = await _embed(_job_text(job))
        async with AsyncSessionLocal() as db:
            await db.execute(
                delete(JobEmbedding).where(JobEmbedding.job_id == str(job_id))
            )
            db.add(JobEmbedding(
                job_id      = str(job_id),
                title       = str(job.get("title") or "")[:512],
                company     = str(job.get("company") or "")[:256],
                location    = str(job.get("location") or "")[:256],
                description = str(job.get("description") or "")[:10000],
                job_url     = str(job.get("job_url") or ""),
                platform    = str(job.get("platform") or "")[:64],
                date_posted = str(job.get("date_posted") or "")[:32],
                is_remote   = bool(job.get("is_remote", False)),
                embedding   = vector,
            ))
            await db.commit()
        logger.info(f"[embedding] Stored embedding for job {job_id}")
        return True
    except Exception as e:
        logger.error(f"[embedding] Failed to embed job {job_id}: {e}", exc_info=True)
        return False


async def semantic_search(
    query: str,
    db: AsyncSession,
    limit: int = 20,
    location: Optional[str] = None,
) -> list[dict]:
    """
    Find jobs whose embeddings are closest to the query embedding.
    Returns a list of dicts with job fields + similarity score (0–1).
    """
    try:
        query_vec = await _embed(query)
    except Exception as e:
        logger.error(f"[embedding] Could not embed query: {e}")
        return []

    vec_str = "[" + ",".join(str(round(v, 6)) for v in query_vec) + "]"

    sql = text("""
        SELECT
            job_id, title, company, location, description,
            job_url, platform, date_posted, is_remote,
            ROUND((1 - (embedding <=> :vec))::numeric, 3) AS similarity
        FROM job_embeddings
        WHERE 1 - (embedding <=> :vec) > 0.2
        ORDER BY embedding <=> :vec
        LIMIT :lim
    """)

    try:
        rows = (await db.execute(sql, {"vec": vec_str, "lim": limit})).fetchall()
        logger.info(f"[embedding] semantic_search '{query[:40]}' → {len(rows)} rows")
    except Exception as e:
        logger.error(f"[embedding] pgvector query failed: {e}", exc_info=True)
        return []

    results = []
    for r in rows:
        results.append({
            "job_id":      r.job_id,
            "title":       r.title,
            "company":     r.company,
            "location":    r.location,
            "description": r.description,
            "job_url":     r.job_url,
            "platform":    r.platform,
            "date_posted": r.date_posted,
            "is_remote":   r.is_remote,
            "similarity":  float(r.similarity),
        })
    return results


async def get_similar_jobs(job_id: str, db: AsyncSession, limit: int = 5) -> list[dict]:
    """Return jobs similar to a given job_id — used for 'Similar Jobs' on job cards."""
    sql = text("""
        SELECT
            e2.job_id, e2.title, e2.company, e2.location,
            e2.job_url, e2.platform,
            ROUND((1 - (e1.embedding <=> e2.embedding))::numeric, 3) AS similarity
        FROM job_embeddings e1
        JOIN job_embeddings e2 ON e2.job_id != e1.job_id
        WHERE e1.job_id = :jid
          AND 1 - (e1.embedding <=> e2.embedding) > 0.5
        ORDER BY e1.embedding <=> e2.embedding
        LIMIT :lim
    """)
    try:
        rows = (await db.execute(sql, {"jid": job_id, "lim": limit})).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        logger.warning(f"[embedding] similar_jobs failed for {job_id}: {e}")
        return []
