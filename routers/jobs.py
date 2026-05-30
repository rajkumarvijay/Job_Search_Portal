from fastapi import APIRouter, Query, Header
from typing import Optional
from services.job_fetcher import fetch_jobs, ALL_PLATFORMS
from services.cache_service import get_from_memory, set_in_memory, make_search_key
from schemas.job import SearchResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])

RESULTS_PER_SITE = 8   # per portal — 8 × 6 portals = up to 48 results total


@router.get("/search", response_model=SearchResponse)
async def search_jobs(
    q: str = Query(..., min_length=1, description="Search keywords"),
    location: str = Query("India", description="Job location"),
    platforms: str = Query("all", description="Comma-separated platforms or 'all'"),
    results_per_site: int = Query(RESULTS_PER_SITE, ge=1, le=25),
    page: int = Query(1, ge=1),
    x_session_id: Optional[str] = Header(None),
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
            query=q,
            location=location,
            total=len(jobs),
            page=page,
            per_page=results_per_site * len(platform_list),
            jobs=jobs,
            platforms_searched=platform_list,
            cached=True,
        )

    jobs = await fetch_jobs(q, location, platform_list, results_per_site)

    set_in_memory(cache_key, {"jobs": jobs})

    return SearchResponse(
        query=q,
        location=location,
        total=len(jobs),
        page=page,
        per_page=results_per_site * len(platform_list),
        jobs=jobs,
        platforms_searched=platform_list,
        cached=False,
    )
