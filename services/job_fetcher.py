import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import pandas as pd
from jobspy import scrape_jobs
from schemas.job import JobResult

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=8)

ALL_PLATFORMS = ["linkedin", "indeed", "glassdoor", "naukri", "ziprecruiter", "google"]

# Per-portal timeout — generous to handle slow portals
PORTAL_TIMEOUT = {
    "linkedin":     40,
    "indeed":       45,
    "glassdoor":    45,
    "naukri":       50,
    "ziprecruiter": 40,
    "google":       40,
}

# Per-portal scrape kwargs — each portal has different requirements
PORTAL_KWARGS = {
    "linkedin": {
        "linkedin_fetch_description": False,
    },
    "indeed": {
        "country_indeed": "India",
    },
    "glassdoor": {},
    "naukri": {},
    "ziprecruiter": {},
    "google": {},
}

# Location overrides per portal
# ZipRecruiter is US-focused; use "Remote" to still get relevant results
PORTAL_LOCATION = {
    "ziprecruiter": "Remote",
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


def _scrape_one_portal(portal: str, query: str, location: str, results: int) -> list[JobResult]:
    """Scrape a single portal in a thread. Never raises — always returns a list."""
    portal_location = PORTAL_LOCATION.get(portal, location)
    extra_kwargs    = PORTAL_KWARGS.get(portal, {})

    try:
        df = scrape_jobs(
            site_name      = [portal],
            search_term    = query,
            location       = portal_location,
            results_wanted = results,
            hours_old      = 168,      # jobs posted in the last 7 days
            **extra_kwargs,
        )

        if df is None or df.empty:
            logger.info(f"[{portal}] 0 results (empty response)")
            return []

        jobs = _normalize_dataframe(df)
        logger.info(f"[{portal}] ✓ {len(jobs)} jobs fetched")
        return jobs

    except Exception as e:
        logger.warning(f"[{portal}] ✗ failed: {type(e).__name__}: {e}")
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

    # Log per-portal summary
    per_portal = {}
    for job in all_jobs:
        per_portal[job.platform] = per_portal.get(job.platform, 0) + 1
    logger.info(f"Results per portal: {per_portal} | Total: {len(all_jobs)}")

    return all_jobs
