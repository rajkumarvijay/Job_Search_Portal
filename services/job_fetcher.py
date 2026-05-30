import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import pandas as pd
from jobspy import scrape_jobs
from schemas.job import JobResult

logger = logging.getLogger(__name__)

# One thread per portal so they run truly in parallel
executor = ThreadPoolExecutor(max_workers=8)

ALL_PLATFORMS = ["linkedin", "indeed", "glassdoor", "naukri", "ziprecruiter", "google"]

# Per-portal timeout in seconds — some portals are slower than others
PORTAL_TIMEOUT = {
    "linkedin":     20,
    "indeed":       25,
    "glassdoor":    25,
    "naukri":       30,
    "ziprecruiter": 20,
    "google":       20,
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
            url   = str(row.get("job_url", "") or "")
            title = str(row.get("title",   "") or "").strip() or "Untitled"
            company = str(row.get("company", "") or "").strip() or "Unknown"
            platform = str(row.get("site",  "") or "").strip()

            min_sal  = _normalize_salary(row.get("min_amount"))
            max_sal  = _normalize_salary(row.get("max_amount"))
            currency = str(row.get("currency", "INR") or "INR")
            interval = str(row.get("interval", "") or "")

            # Convert USD → INR LPA
            if currency.upper() in ("USD", "$") and min_sal:
                min_sal = round(min_sal * 83 / 100000, 2)
                max_sal = round(max_sal * 83 / 100000, 2) if max_sal else None
                currency = "INR"
                interval = "yearly"

            desc = str(row.get("description", "") or "")
            if len(desc) > 2000:
                desc = desc[:2000] + "..."

            date_val = row.get("date_posted")
            date_str = str(date_val) if date_val and str(date_val) != "NaT" else None

            jobs.append(JobResult(
                job_id=_generate_job_id(url, title, company),
                title=title,
                company=company,
                location=str(row.get("location", "") or "India").strip(),
                min_salary=min_sal,
                max_salary=max_sal,
                salary_currency=currency,
                salary_interval=interval or None,
                job_url=url or None,
                platform=platform,
                description=desc or None,
                date_posted=date_str,
                job_type=str(row.get("job_type", "") or "").strip() or None,
                is_remote=bool(row.get("is_remote", False)),
            ))
        except Exception as e:
            logger.warning(f"Skipping malformed row: {e}")
    return jobs


def _scrape_one_portal(portal: str, query: str, location: str, results: int) -> list[JobResult]:
    """Scrape a single portal — runs in a thread."""
    try:
        kwargs = dict(
            site_name=[portal],
            search_term=query,
            location=location,
            results_wanted=results,
            linkedin_fetch_description=False,
        )
        # Indeed needs country specified explicitly
        if portal == "indeed":
            kwargs["country_indeed"] = "India"

        df = scrape_jobs(**kwargs)

        if df is None or df.empty:
            logger.info(f"[{portal}] No results returned")
            return []

        jobs = _normalize_dataframe(df)
        logger.info(f"[{portal}] {len(jobs)} jobs fetched")
        return jobs

    except Exception as e:
        logger.warning(f"[{portal}] Scrape failed: {e}")
        return []


async def _scrape_portal_async(
    portal: str, query: str, location: str, results: int
) -> list[JobResult]:
    """Run a single portal scrape in executor with its own timeout."""
    loop = asyncio.get_event_loop()
    timeout = PORTAL_TIMEOUT.get(portal, 25)
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                _scrape_one_portal,
                portal, query, location, results,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[{portal}] Timed out after {timeout}s")
        return []
    except Exception as e:
        logger.warning(f"[{portal}] Async error: {e}")
        return []


async def fetch_jobs(
    query: str,
    location: str = "India",
    platforms: Optional[list[str]] = None,
    results_per_site: int = 10,
) -> list[JobResult]:
    """
    Scrape each portal individually in parallel.
    One portal failing/timing out never blocks the others.
    """
    target = [p for p in (platforms or ALL_PLATFORMS) if p in ALL_PLATFORMS] or ALL_PLATFORMS

    logger.info(f"Searching '{query}' in {location} across: {target}")

    # Launch all portals concurrently
    tasks = [
        _scrape_portal_async(portal, query, location, results_per_site)
        for portal in target
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_jobs: list[JobResult] = []
    seen_ids: set[str] = set()

    for portal, result in zip(target, results):
        if isinstance(result, Exception):
            logger.warning(f"[{portal}] gather exception: {result}")
            continue
        for job in result:
            if job.job_id not in seen_ids:
                seen_ids.add(job.job_id)
                all_jobs.append(job)

    logger.info(f"Total jobs fetched: {len(all_jobs)} across {len(target)} portals")
    return all_jobs
