import asyncio
import logging
from services.job_fetcher import fetch_jobs, ALL_PLATFORMS
from services.cache_service import get_from_db, set_in_db, get_from_memory, set_in_memory
from schemas.trending import TrendingRole, SalaryBand, TrendingKeyword, StatsResponse

logger = logging.getLogger(__name__)

CURATED_ROLES = [
    {"role": "Software Engineer", "top_skill": "Python", "icon": "code"},
    {"role": "Data Analyst", "top_skill": "SQL", "icon": "bar-chart"},
    {"role": "Product Manager", "top_skill": "Agile", "icon": "layers"},
    {"role": "DevOps Engineer", "top_skill": "Kubernetes", "icon": "server"},
    {"role": "Frontend Developer", "top_skill": "React", "icon": "monitor"},
    {"role": "Backend Developer", "top_skill": "Node.js", "icon": "database"},
    {"role": "Data Scientist", "top_skill": "ML", "icon": "cpu"},
    {"role": "Cloud Architect", "top_skill": "AWS", "icon": "cloud"},
    {"role": "UI/UX Designer", "top_skill": "Figma", "icon": "pen-tool"},
    {"role": "Business Analyst", "top_skill": "Excel", "icon": "trending-up"},
    {"role": "QA Engineer", "top_skill": "Selenium", "icon": "check-circle"},
    {"role": "Mobile Developer", "top_skill": "Flutter", "icon": "smartphone"},
]

SALARY_BANDS = [
    SalaryBand(role="Software Engineer", fresher="3–6", mid="8–18", senior="20–45"),
    SalaryBand(role="Data Analyst", fresher="3–5", mid="7–14", senior="15–30"),
    SalaryBand(role="Product Manager", fresher="5–8", mid="12–22", senior="25–50"),
    SalaryBand(role="DevOps Engineer", fresher="4–7", mid="10–20", senior="22–40"),
    SalaryBand(role="Frontend Developer", fresher="3–6", mid="8–16", senior="18–35"),
    SalaryBand(role="Backend Developer", fresher="3–6", mid="8–18", senior="20–40"),
    SalaryBand(role="Data Scientist", fresher="5–8", mid="12–22", senior="25–50"),
    SalaryBand(role="Cloud Architect", fresher="6–10", mid="15–25", senior="28–55"),
    SalaryBand(role="UI/UX Designer", fresher="3–5", mid="6–14", senior="15–28"),
    SalaryBand(role="Business Analyst", fresher="3–5", mid="7–14", senior="15–28"),
    SalaryBand(role="QA Engineer", fresher="2–4", mid="6–12", senior="13–25"),
    SalaryBand(role="Mobile Developer", fresher="3–6", mid="8–16", senior="18–35"),
]

CURATED_KEYWORDS = [
    "Python", "React", "Machine Learning", "AWS", "SQL", "Node.js",
    "Docker", "Kubernetes", "Data Science", "Product Management",
    "DevOps", "TypeScript", "Java", "Golang", "Figma", "Agile",
    "Deep Learning", "NLP", "Microservices", "CI/CD",
]

FALLBACK_COUNTS = {r["role"]: 1500 + i * 300 for i, r in enumerate(CURATED_ROLES)}


async def get_trending_roles() -> list[TrendingRole]:
    key = "trending_roles_v1"
    cached = get_from_memory(key)
    if cached:
        return [TrendingRole(**r) for r in cached]

    db_cached = await get_from_db(key)
    if db_cached:
        set_in_memory(key, db_cached)
        return [TrendingRole(**r) for r in db_cached]

    roles = await _compute_trending_roles()
    data = [r.model_dump() for r in roles]
    set_in_memory(key, data)
    await set_in_db(key, data, ttl_hours=6)
    return roles


async def _compute_trending_roles() -> list[TrendingRole]:
    results = []

    async def count_role(meta: dict) -> TrendingRole:
        try:
            jobs = await asyncio.wait_for(
                fetch_jobs(meta["role"], "India", ALL_PLATFORMS, results_per_site=5),
                timeout=30,
            )
            count = len(jobs) * 80 + FALLBACK_COUNTS.get(meta["role"], 1000)
        except Exception:
            count = FALLBACK_COUNTS.get(meta["role"], 1200)

        band = next((s for s in SALARY_BANDS if s.role == meta["role"]), None)
        return TrendingRole(
            role=meta["role"],
            count=count,
            top_skill=meta["top_skill"],
            avg_salary_lpa=band.mid if band else "8–18",
            icon=meta["icon"],
        )

    tasks = [count_role(m) for m in CURATED_ROLES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, TrendingRole)]


async def get_salary_bands() -> list[SalaryBand]:
    return SALARY_BANDS


async def get_trending_keywords() -> list[TrendingKeyword]:
    return [
        TrendingKeyword(keyword=kw, count=10000 - i * 300, trend="up" if i < 10 else "stable")
        for i, kw in enumerate(CURATED_KEYWORDS)
    ]


async def get_stats() -> StatsResponse:
    key = "portal_stats_v1"
    cached = get_from_memory(key)
    if cached:
        return StatsResponse(**cached)

    db_cached = await get_from_db(key)
    if db_cached:
        set_in_memory(key, db_cached)
        return StatsResponse(**db_cached)

    stats = StatsResponse(
        total_active_jobs=485000,
        top_salary_lpa="55 LPA",
        platform_count=6,
        cities_covered=45,
    )
    data = stats.model_dump()
    set_in_memory(key, data)
    await set_in_db(key, data, ttl_hours=6)
    return stats


async def refresh_trending_cache():
    logger.info("Refreshing trending cache...")
    for key in ["trending_roles_v1", "portal_stats_v1"]:
        try:
            if key in _get_memory_cache():
                del _get_memory_cache()[key]
        except Exception:
            pass
    await get_trending_roles()
    await get_stats()
    logger.info("Trending cache refreshed.")


def _get_memory_cache():
    from services.cache_service import _memory_cache
    return _memory_cache
