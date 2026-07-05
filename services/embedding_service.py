"""
Semantic job search via pgvector + HuggingFace Inference API.

Model: sentence-transformers/all-mpnet-base-v2  (768 dims — matches schema)
API:   https://api-inference.huggingface.co/models/<model>
Auth:  Bearer token from env var HUGGINGFACE_API_TOKEN

Flow:
  store_job_embedding(job)  — called after each job is scraped
  semantic_search(query)    — called from /jobs/semantic-search endpoint
  get_similar_jobs(job_id)  — called from /jobs/similar/{job_id} endpoint
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
from sqlalchemy import text, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.models import JobEmbedding

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)

EMBEDDING_DIM = 768
HF_MODEL     = "sentence-transformers/all-mpnet-base-v2"
HF_API_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# HF cold-start: model may be loading on first request — retry up to 3×
_MAX_RETRIES = 3
_RETRY_DELAY = 20   # seconds to wait if HF returns 503 "model is loading"


def _get_hf_token() -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    if not token:
        raise ValueError("HUGGINGFACE_API_TOKEN is not set in environment variables")
    return token


# ── Core embedding call (sync, runs in thread pool) ───────────────────────────

def _embed_sync(content: str) -> list[float]:
    """
    Call the HF Inference API and return a 768-dim embedding vector.
    Retries on 503 (model loading) with a delay.
    """
    token = _get_hf_token()
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": content[:8000]}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = httpx.post(HF_API_URL, json=payload, headers=headers, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                # API returns list[list[float]] — one vector per input string
                vector = data[0] if isinstance(data[0], list) else data
                logger.info(f"[embedding] HF API ok — dims={len(vector)}")
                return vector

            if resp.status_code == 503:
                # Model is cold-starting on HF infrastructure
                logger.warning(
                    f"[embedding] HF model loading (attempt {attempt}/{_MAX_RETRIES}), "
                    f"waiting {_RETRY_DELAY}s..."
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    continue

            raise RuntimeError(
                f"HF Inference API returned {resp.status_code}: {resp.text[:200]}"
            )

        except httpx.TimeoutException:
            logger.warning(f"[embedding] HF API timeout (attempt {attempt})")
            if attempt == _MAX_RETRIES:
                raise RuntimeError("HF Inference API timed out after all retries")

    raise RuntimeError("HF Inference API failed after all retries")


async def _embed(text_content: str) -> list[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _embed_sync, text_content)


# ── Job text serialisation ────────────────────────────────────────────────────

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

    return [
        {
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
        }
        for r in rows
    ]


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
