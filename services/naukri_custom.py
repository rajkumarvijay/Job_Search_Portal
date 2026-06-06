"""
Custom Naukri fetcher using their v2 public API.
The jobspy v3 endpoint requires recaptcha; v2 works without it.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Optional

import requests

from schemas.job import JobResult

logger = logging.getLogger(__name__)

NAUKRI_V2_URL = "https://www.naukri.com/jobapi/v2/search"

HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "appid": "109",
    "systemid": "Naukri",
    "referer": "https://www.naukri.com/",
}


def _job_id(job_id_raw: str, title: str, company: str) -> str:
    raw = f"naukri-{job_id_raw}-{title}-{company}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _parse_salary(val: Optional[str]) -> Optional[float]:
    if not val or str(val).strip() in ("", "0", "None"):
        return None
    try:
        # Strip currency symbols and parse
        cleaned = re.sub(r"[^\d.]", "", str(val))
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def _parse_location(job: dict) -> str:
    city_field = job.get("cityfield") or ""
    # cityfield contains a long messy string, extract first location
    if city_field:
        # Try to get the first meaningful city from the field
        parts = [p.strip() for p in re.split(r"\s{2,}|,|;", city_field) if p.strip()]
        for part in parts:
            # Skip generic terms
            if part.lower() not in ("", "metropolitan cities", "top", "anywhere in india",
                                    "popular locations", "south preferred jobseeker",
                                    "southindia", "north"):
                return part
    return job.get("location") or "India"


def _parse_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(str(date_str).split(".")[0], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(date_str)[:10] if date_str else None


def _parse_listing(job: dict) -> Optional[JobResult]:
    try:
        title = job.get("post") or job.get("title") or "Untitled"
        company = job.get("companyName") or job.get("CONTCOM") or "Unknown"
        location = _parse_location(job)
        url = job.get("urlStr") or job.get("jdURL") or None
        if url and not url.startswith("http"):
            url = f"https://www.naukri.com{url}"

        job_id_raw = job.get("jobId") or job.get("REFNO") or ""
        description = job.get("jobDesc") or job.get("tupleDesc") or ""
        if len(description) > 2000:
            description = description[:2000] + "..."

        # Salary
        min_sal_raw = job.get("minSal") or job.get("SALARY")
        max_sal_raw = job.get("maxSal")
        min_sal = _parse_salary(str(min_sal_raw)) if min_sal_raw else None
        max_sal = _parse_salary(str(max_sal_raw)) if max_sal_raw else None

        # Convert to LPA if > 100000 (raw values are annual in INR)
        if min_sal and min_sal > 100000:
            min_sal = round(min_sal / 100000, 2)
        if max_sal and max_sal > 100000:
            max_sal = round(max_sal / 100000, 2)

        date_str = _parse_date(job.get("addDate"))
        job_type = job.get("employmentType") or None
        is_remote = "remote" in (job.get("jobDesc") or "").lower() or \
                    "remote" in location.lower()

        return JobResult(
            job_id=_job_id(str(job_id_raw), title, company),
            title=title,
            company=company,
            location=location,
            min_salary=min_sal,
            max_salary=max_sal,
            salary_currency="INR",
            salary_interval="yearly" if (min_sal and min_sal < 1000) else None,
            job_url=url,
            platform="naukri",
            description=description or None,
            date_posted=date_str,
            job_type=job_type,
            is_remote=is_remote,
        )
    except Exception as e:
        logger.warning(f"[naukri_custom] parse error: {e}")
        return None


def fetch_naukri(query: str, location: str = "India", results: int = 10) -> list[JobResult]:
    session = requests.Session()
    session.headers.update(HEADERS)

    all_jobs: list[JobResult] = []
    seen_ids: set[str] = set()
    page = 1
    jobs_per_page = 20

    while len(all_jobs) < results:
        params = {
            "noOfResults": min(jobs_per_page, results - len(all_jobs) + 5),
            "keyword": query,
            "location": location if location.lower() != "india" else "",
            "pageNo": page,
        }
        # Remove empty location param
        params = {k: v for k, v in params.items() if v != ""}

        try:
            resp = session.get(NAUKRI_V2_URL, params=params, timeout=20)
            if resp.status_code not in (200, 206):
                logger.warning(f"[naukri_custom] HTTP {resp.status_code}")
                break

            data = resp.json()
            jobs = data.get("list") or data.get("jobDetails") or []

            if not jobs:
                break

            for job in jobs:
                parsed = _parse_listing(job)
                if parsed and parsed.job_id not in seen_ids:
                    seen_ids.add(parsed.job_id)
                    all_jobs.append(parsed)
                    if len(all_jobs) >= results:
                        break

            total_pages = data.get("totalpages") or 1
            if page >= total_pages or len(all_jobs) >= results:
                break
            page += 1

        except Exception as e:
            logger.warning(f"[naukri_custom] error on page {page}: {e}")
            break

    logger.info(f"[naukri_custom] ✓ {len(all_jobs)} jobs for '{query}' in '{location}'")
    return all_jobs
