import logging
from fastapi import APIRouter, Query, Header, Depends
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from db.database import get_db
from db.models import PostedJob
from services.job_fetcher import fetch_jobs, ALL_PLATFORMS
from services.gemini_service import search_jobs_ai
from services.cache_service import get_from_memory, set_in_memory, make_search_key
from services.embedding_service import semantic_search, get_similar_jobs
from schemas.job import JobResult, SearchResponse
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])

RESULTS_PER_SITE = 8   # per portal — 8 × 6 portals = up to 48 results total


def _gemini_to_job_result(j: dict) -> Optional[JobResult]:
    """Convert a Gemini AI job dict to a JobResult schema object."""
    try:
        return JobResult(
            job_id          = j.get("job_id", ""),
            title           = j.get("title", "Untitled"),
            company         = j.get("company", "Unknown"),
            location        = j.get("location", ""),
            min_salary      = j.get("min_salary"),
            max_salary      = j.get("max_salary"),
            salary_currency = j.get("salary_currency", "INR"),
            salary_interval = None,
            job_url         = j.get("job_url"),
            platform        = j.get("platform", "ai"),
            description     = j.get("description"),
            date_posted     = j.get("date_posted"),
            job_type        = j.get("job_type"),
            is_remote       = bool(j.get("is_remote", False)),
        )
    except Exception as e:
        logger.warning(f"Could not convert Gemini job: {e}")
        return None


def _posted_to_job_result(p: PostedJob) -> JobResult:
    """Convert a PostedJob DB row into a JobResult schema object."""
    is_remote = (p.work_mode or "").lower() == "remote"
    apply = p.apply_url or f"mailto:{p.contact_email}"
    return JobResult(
        job_id          = p.job_id,
        title           = p.title,
        company         = p.company,
        location        = p.location,
        min_salary      = p.min_salary,
        max_salary      = p.max_salary,
        salary_currency = p.salary_currency or "INR",
        salary_interval = "yearly",
        job_url         = apply,
        platform        = "portal",
        description     = p.description,
        date_posted     = p.posted_at.strftime("%Y-%m-%d") if p.posted_at else None,
        job_type        = p.job_type,
        is_remote       = is_remote,
    )


@router.get("/search", response_model=SearchResponse)
async def search_jobs(
    q:               str = Query(..., min_length=1),
    location:        str = Query("India"),
    platforms:       str = Query("all"),
    results_per_site:int = Query(RESULTS_PER_SITE, ge=1, le=25),
    page:            int = Query(1, ge=1),
    x_session_id: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    platform_list = (
        ALL_PLATFORMS
        if platforms.lower() == "all"
        else [p.strip() for p in platforms.split(",") if p.strip()]
    )

    cache_key = make_search_key(q, location, platform_list, results_per_site)
    cached = get_from_memory(cache_key)
    if cached:
        jobs = cached["jobs"]
        return SearchResponse(
            query=q, location=location,
            total=len(jobs), page=page,
            per_page=results_per_site * len(platform_list),
            jobs=jobs, platforms_searched=platform_list, cached=True,
        )

    # ── Run jobspy + Gemini AI search in parallel ─────────────────────────────
    jobspy_task = fetch_jobs(q, location, platform_list, results_per_site)
    gemini_task = search_jobs_ai(q, location, results_wanted=20)

    jobspy_results, gemini_results = await asyncio.gather(
        jobspy_task, gemini_task,
        return_exceptions=True,
    )

    seen_ids: set[str] = set()
    all_jobs: list[JobResult] = []

    # Add jobspy results first
    if isinstance(jobspy_results, list):
        for job in jobspy_results:
            if job.job_id not in seen_ids:
                seen_ids.add(job.job_id)
                all_jobs.append(job)
    else:
        logger.warning(f"jobspy failed: {jobspy_results}")

    # Merge Gemini AI results (deduplicated)
    if isinstance(gemini_results, list):
        for j in gemini_results:
            job = _gemini_to_job_result(j)
            if job and job.job_id and job.job_id not in seen_ids:
                seen_ids.add(job.job_id)
                all_jobs.append(job)
    else:
        logger.warning(f"Gemini search failed: {gemini_results}")

    # ── Merge posted jobs from DB ─────────────────────────────────────────────
    try:
        kw = f"%{q}%"
        stmt = select(PostedJob).where(
            PostedJob.is_active == True,
            or_(
                PostedJob.title.ilike(kw),
                PostedJob.description.ilike(kw),
                PostedJob.skills.ilike(kw),
                PostedJob.company.ilike(kw),
            )
        ).limit(10)
        posted_rows = (await db.execute(stmt)).scalars().all()
        for p in posted_rows:
            if p.job_id not in seen_ids:
                seen_ids.add(p.job_id)
                all_jobs.insert(0, _posted_to_job_result(p))   # surface at top
    except Exception as e:
        logger.warning(f"Could not fetch posted jobs: {e}")

    logger.info(f"Total combined jobs: {len(all_jobs)} (jobspy + Gemini AI + portal posts)")

    set_in_memory(cache_key, {"jobs": all_jobs})

    return SearchResponse(
        query=q, location=location,
        total=len(all_jobs), page=page,
        per_page=results_per_site * len(platform_list),
        jobs=all_jobs,
        platforms_searched=platform_list + (["ai"] if isinstance(gemini_results, list) and gemini_results else []),
        cached=False,
    )


# ── Semantic Search ───────────────────────────────────────────────────────────

@router.get("/semantic-search")
async def semantic_search_jobs(
    q:        str = Query(..., min_length=2, description="Natural-language query"),
    location: str = Query("India"),
    limit:    int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Meaning-based job search using pgvector cosine similarity.
    Works with vague queries like "remote python work" or "finance role in south india".
    """
    results = await semantic_search(q, db, limit=limit, location=location)

    jobs = [
        JobResult(
            job_id      = r["job_id"],
            title       = r["title"],
            company     = r["company"],
            location    = r.get("location") or "",
            description = r.get("description"),
            job_url     = r.get("job_url"),
            platform    = r.get("platform") or "semantic",
            date_posted = r.get("date_posted"),
            is_remote   = r.get("is_remote", False),
            salary_currency = "INR",
        )
        for r in results
    ]

    return {
        "query":    q,
        "location": location,
        "total":    len(jobs),
        "jobs":     jobs,
        "mode":     "semantic",
    }


@router.get("/similar/{job_id}")
async def similar_jobs(
    job_id: str,
    limit:  int = Query(5, ge=1, le=10),
    db: AsyncSession = Depends(get_db),
):
    """Return jobs semantically similar to a given job — powers 'Similar Jobs' cards."""
    results = await get_similar_jobs(job_id, db, limit=limit)
    return {"job_id": job_id, "similar": results, "total": len(results)}


@router.post("/seed-embeddings")
async def seed_embeddings(db: AsyncSession = Depends(get_db)):
    """
    Scrape 10 common Indian job queries and embed all results.
    Call this once after first deploy to populate the vector index.
    Returns count of jobs embedded.
    """
    from services.embedding_service import store_job_embedding
    from services.job_fetcher import fetch_jobs as _fetch

    SEED_QUERIES = [
        "software engineer", "data analyst", "product manager",
        "frontend developer", "backend developer", "full stack developer",
        "python developer", "react developer", "devops engineer",
        "data scientist",
    ]

    total_embedded = 0
    for query in SEED_QUERIES:
        try:
            jobs = await _fetch(query, "India", None, 5)
            for job in jobs:
                ok = await store_job_embedding(job.model_dump())
                if ok:
                    total_embedded += 1
        except Exception as e:
            logger.warning(f"[seed] Failed for '{query}': {e}")

    return {
        "message": f"Seeded {total_embedded} job embeddings",
        "total_embedded": total_embedded,
        "queries_run": len(SEED_QUERIES),
    }


@router.get("/test-embedding")
async def test_embedding():
    """Debug endpoint — tests the local sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        vector = model.encode("software engineer python django").tolist()
        return {"status": "ok", "model": "all-mpnet-base-v2", "dims": len(vector), "sample": vector[:5]}
    except Exception as e:
        return {"status": "error", "error": str(e)[:300]}
