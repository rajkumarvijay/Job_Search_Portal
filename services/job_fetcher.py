import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import pandas as pd
from jobspy import scrape_jobs
from schemas.job import JobResult
from services.naukri_custom import fetch_naukri
from services.glassdoor_custom import fetch_glassdoor
from services.fallback_fetcher import fetch_remotive, fetch_arbeitnow
from services.embedding_service import store_job_embedding

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=8)

ALL_PLATFORMS = ["linkedin", "indeed", "glassdoor", "naukri", "ziprecruiter", "google"]

# Portals handled by jobspy directly
JOBSPY_PLATFORMS = {"linkedin", "indeed"}

# Per-portal timeout
PORTAL_TIMEOUT = {
    "linkedin":     45,
    "indeed":       50,
    "glassdoor":    40,
    "naukri":       40,
    "ziprecruiter": 35,
    "google":       35,
}

# jobspy kwargs per portal
JOBSPY_KWARGS = {
    "linkedin": {"linkedin_fetch_description": False},
    "indeed":   {"country_indeed": "India"},
}


def _generate_job_id(url: str, title: str, company: str) -> str:
    raw = f"{url}{title}{company}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _normalize_salary(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_dataframe(df: pd.DataFrame) -> list[JobResult]:
    jobs = []
    for _, row in df.iterrows():
        try:
            url      = str(row.get("job_url",  "") or "")
            title    = str(row.get("title",    "") or "").strip() or "Untitled"
            company  = str(row.get("company",  "") or "").strip() or "Unknown"
            platform = str(row.get("site",     "") or "").strip()

            min_sal  = _normalize_salary(row.get("min_amount"))
            max_sal  = _normalize_salary(row.get("max_amount"))
            currency = str(row.get("currency", "INR") or "INR")
            interval = str(row.get("interval", "")    or "")

            # Convert USD → INR LPA
            if currency.upper() in ("USD", "$") and min_sal:
                min_sal  = round(min_sal * 83 / 100000, 2)
                max_sal  = round(max_sal * 83 / 100000, 2) if max_sal else None
                currency = "INR"
                interval = "yearly"

            desc = str(row.get("description", "") or "")
            if len(desc) > 2000:
                desc = desc[:2000] + "..."

            date_val = row.get("date_posted")
            date_str = str(date_val) if date_val and str(date_val) != "NaT" else None

            jobs.append(JobResult(
                job_id          = _generate_job_id(url, title, company),
                title           = title,
                company         = company,
                location        = str(row.get("location", "") or "India").strip(),
                min_salary      = min_sal,
                max_salary      = max_sal,
                salary_currency = currency,
                salary_interval = interval or None,
                job_url         = url or None,
                platform        = platform,
                description     = desc or None,
                date_posted     = date_str,
                job_type        = str(row.get("job_type", "") or "").strip() or None,
                is_remote       = bool(row.get("is_remote", False)),
            ))
        except Exception as e:
            logger.warning(f"Skipping malformed row: {e}")
    return jobs


def _scrape_jobspy(portal: str, query: str, location: str, results: int) -> list[JobResult]:
    """Scrape a portal using jobspy. Returns [] on any failure."""
    extra_kwargs = JOBSPY_KWARGS.get(portal, {})
    try:
        df = scrape_jobs(
            site_name      = [portal],
            search_term    = query,
            location       = location,
            results_wanted = results,
            hours_old      = 168,
            **extra_kwargs,
        )
        if df is None or df.empty:
            logger.info(f"[{portal}] 0 results (empty)")
            return []
        jobs = _normalize_dataframe(df)
        logger.info(f"[{portal}] ✓ {len(jobs)} jobs (jobspy)")
        return jobs
    except Exception as e:
        logger.warning(f"[{portal}] ✗ jobspy error: {type(e).__name__}: {e}")
        return []


def _scrape_one_portal(portal: str, query: str, location: str, results: int) -> list[JobResult]:
    """Route each portal to its working scraper. Never raises."""
    try:
        if portal == "naukri":
            jobs = fetch_naukri(query, location, results)
            # If custom scraper fails, try jobspy as fallback
            if not jobs:
                logger.info(f"[naukri] custom failed, trying jobspy fallback")
                jobs = _scrape_jobspy("naukri", query, location, results)
            return jobs

        if portal == "glassdoor":
            jobs = fetch_glassdoor(query, location, results)
            if not jobs:
                logger.info(f"[glassdoor] custom failed, trying jobspy fallback")
                jobs = _scrape_jobspy("glassdoor", query, location, results)
            return jobs

        if portal == "ziprecruiter":
            # ZipRecruiter is US-only and blocked — go straight to Arbeitnow fallback
            jobs = fetch_arbeitnow(query, results, platform_label="ziprecruiter")
            return jobs

        if portal == "google":
            # Google Jobs scraper is blocked; go straight to Remotive fallback
            jobs = fetch_remotive(query, results, platform_label="google")
            return jobs

        # LinkedIn and Indeed via jobspy (working)
        return _scrape_jobspy(portal, query, location, results)

    except Exception as e:
        logger.warning(f"[{portal}] ✗ unexpected error: {type(e).__name__}: {e}")
        return []


async def _scrape_portal_async(portal: str, query: str, location: str, results: int) -> list[JobResult]:
    """Run portal scrape in executor with per-portal timeout."""
    loop    = asyncio.get_event_loop()
    timeout = PORTAL_TIMEOUT.get(portal, 45)
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, _scrape_one_portal, portal, query, location, results),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[{portal}] ✗ timed out after {timeout}s")
        return []
    except Exception as e:
        logger.warning(f"[{portal}] ✗ async error: {e}")
        return []


async def fetch_jobs(
    query:            str,
    location:         str = "India",
    platforms:        Optional[list[str]] = None,
    results_per_site: int = 10,
) -> list[JobResult]:
    target = [p for p in (platforms or ALL_PLATFORMS) if p in ALL_PLATFORMS] or ALL_PLATFORMS

    logger.info(f"Search: '{query}' | location: {location} | portals: {target}")

    results = await asyncio.gather(
        *[_scrape_portal_async(p, query, location, results_per_site) for p in target],
        return_exceptions=True,
    )

    all_jobs: list[JobResult] = []
    seen_ids: set[str] = set()

    for portal, result in zip(target, results):
        if isinstance(result, Exception):
            logger.warning(f"[{portal}] gather exception: {result}")
            continue
        for job in (result or []):
            if job.job_id not in seen_ids:
                seen_ids.add(job.job_id)
                all_jobs.append(job)

    per_portal = {}
    for job in all_jobs:
        per_portal[job.platform] = per_portal.get(job.platform, 0) + 1
    logger.info(f"Results per portal: {per_portal} | Total: {len(all_jobs)}")

    # Fire-and-forget embedding — doesn't block search response
    asyncio.create_task(_embed_all(all_jobs))

    return all_jobs


async def _embed_all(jobs: list[JobResult]) -> None:
    """Embed all scraped jobs in the background — runs after search response is sent."""
    ok = 0
    for job in jobs:
        try:
            success = await store_job_embedding(job.model_dump())
            if success:
                ok += 1
        except Exception:
            pass
    logger.info(f"[embedding] Embedded {ok}/{len(jobs)} jobs in background")
