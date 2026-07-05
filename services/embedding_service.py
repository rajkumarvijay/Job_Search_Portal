"""
Semantic job search via pgvector + sentence-transformers (local model).

Model: sentence-transformers/all-mpnet-base-v2  (768 dims — matches schema)
Runs entirely inside the container — no external API calls needed.

Flow:
  store_job_embedding(job)  — called after each job is scraped
  semantic_search(query)    — called from /jobs/semantic-search endpoint
  get_similar_jobs(job_id)  — called from /jobs/similar/{job_id} endpoint
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from sqlalchemy import text, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.models import JobEmbedding

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)

EMBEDDING_DIM = 384
HF_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"

# ── Lazy model load (singleton) ───────────────────────────────────────────────

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[embedding] Loading model {HF_MODEL} ...")
        _model = SentenceTransformer(HF_MODEL)
        logger.info("[embedding] Model loaded.")
    return _model


# ── Core embedding call (sync, runs in thread pool) ───────────────────────────

def _embed_sync(content: str) -> list[float]:
    return _get_model().encode(content[:8000], convert_to_numpy=True).tolist()


async def _embed(text_content: str) -> list[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _embed_sync, text_content)


# ── Job text serialisation ────────────────────────────────────────────────────

def _job_text(job: dict) -> str:
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
