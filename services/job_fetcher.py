import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import pandas as pd
from jobspy import scrape_jobs
from schemas.job import JobResult

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)

PLATFORM_MAP = {
    "linkedin": "linkedin",
    "indeed": "indeed",
    "glassdoor": "glassdoor",
    "naukri": "naukri",
    "ziprecruiter": "ziprecruiter",
    "google": "google",
}

ALL_PLATFORMS = list(PLATFORM_MAP.values())


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


def _normalize_dataframe(df: pd.DataFrame, platform_label: str = "") -> list[JobResult]:
    jobs = []
    for _, row in df.iterrows():
        try:
            url = str(row.get("job_url", "")) or ""
            title = str(row.get("title", "")).strip() or "Untitled"
            company = str(row.get("company", "")).strip() or "Unknown"
            platform = str(row.get("site", platform_label or "")).strip()

            job_id = _generate_job_id(url, title, company)

            min_sal = _normalize_salary(row.get("min_amount"))
            max_sal = _normalize_salary(row.get("max_amount"))
            currency = str(row.get("currency", "INR") or "INR")
            interval = str(row.get("interval", "") or "")

            # Convert yearly USD to INR LPA estimate if needed
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
                job_id=job_id,
                title=title,
                company=company,
                location=str(row.get("location", "") or "India").strip(),
                min_salary=min_sal,
                max_salary=max_sal,
                salary_currency=currency,
                salary_interval=interval if interval else None,
                job_url=url if url else None,
                platform=platform,
                description=desc if desc else None,
                date_posted=date_str,
                job_type=str(row.get("job_type", "") or "").strip() or None,
                is_remote=bool(row.get("is_remote", False)),
            ))
        except Exception as e:
            logger.warning(f"Skipping malformed job row: {e}")
    return jobs


async def fetch_jobs(
    query: str,
    location: str = "India",
    platforms: Optional[list[str]] = None,
    results_per_site: int = 10,
) -> list[JobResult]:
    if not platforms:
        platforms = ALL_PLATFORMS

    valid_platforms = [p for p in platforms if p in ALL_PLATFORMS]
    if not valid_platforms:
        valid_platforms = ALL_PLATFORMS

    def _scrape():
        try:
            df = scrape_jobs(
                site_name=valid_platforms,
                search_term=query,
                location=location,
                results_wanted=results_per_site,
                country_indeed="India",
                linkedin_fetch_description=False,
            )
            return df
        except Exception as e:
            logger.error(f"jobspy scrape error: {e}")
            return pd.DataFrame()

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(executor, _scrape)

    if df.empty:
        return []

    return _normalize_dataframe(df)
