"""
Custom Glassdoor fetcher using GraphQL API directly.
Bypasses the blocked location-lookup endpoint by using hardcoded location IDs.
"""
from __future__ import annotations

import hashlib
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

from jobspy.glassdoor.constant import query_template, fallback_token
from schemas.job import JobResult

logger = logging.getLogger(__name__)

GLASSDOOR_BASE = "https://www.glassdoor.com"
GRAPHQL_URL = f"{GLASSDOOR_BASE}/graph"

# Hardcoded location IDs (from Glassdoor URL patterns: IN<id>)
LOCATION_MAP: dict[str, tuple[int, str]] = {
    "india":        (115,   "COUNTRY"),
    "remote":       (11047, "STATE"),
    "bangalore":    (3,     "CITY"),
    "bengaluru":    (3,     "CITY"),
    "mumbai":       (6,     "CITY"),
    "delhi":        (10,    "CITY"),
    "hyderabad":    (4,     "CITY"),
    "chennai":      (5,     "CITY"),
    "pune":         (8,     "CITY"),
    "kolkata":      (9,     "CITY"),
    "ahmedabad":    (12,    "CITY"),
    "noida":        (13,    "CITY"),
    "gurgaon":      (14,    "CITY"),
}

HEADERS = {
    "authority": "www.glassdoor.com",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "apollographql-client-name": "job-search-next",
    "apollographql-client-version": "4.65.5",
    "content-type": "application/json",
    "origin": GLASSDOOR_BASE,
    "referer": f"{GLASSDOOR_BASE}/",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}


def _job_id(listing_id: str, title: str, company: str) -> str:
    raw = f"glassdoor-{listing_id}-{title}-{company}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _parse_date(age_in_days: Optional[int]) -> Optional[str]:
    if age_in_days is None:
        return None
    dt = datetime.utcnow() - timedelta(days=age_in_days)
    return dt.strftime("%Y-%m-%d")


def _parse_listing(listing: dict) -> Optional[JobResult]:
    try:
        jv = listing.get("jobview", {})
        header = jv.get("header", {})
        job = jv.get("job", {})
        overview = jv.get("overview", {})

        title = header.get("jobTitleText") or job.get("jobTitleText") or "Untitled"
        company = (
            (header.get("employer") or {}).get("name")
            or header.get("employerNameFromSearch")
            or overview.get("shortName")
            or "Unknown"
        )
        location = header.get("locationName") or "India"
        job_link = header.get("jobLink") or ""
        url = (GLASSDOOR_BASE + job_link) if job_link else None
        listing_id = str(job.get("listingId") or "")
        description = job.get("description") or ""
        age_in_days = header.get("ageInDays")
        currency = header.get("payCurrency") or "INR"
        pay_period = header.get("payPeriod") or ""

        pay = header.get("payPeriodAdjustedPay") or header.get("salarySource") or {}
        min_sal = max_sal = None
        if isinstance(pay, dict):
            p50 = pay.get("p50") or pay.get("salaryEstimate", {})
            p10 = pay.get("p10")
            p90 = pay.get("p90")
            if isinstance(p50, (int, float)):
                min_sal = p10 if p10 else p50
                max_sal = p90 if p90 else p50
            elif isinstance(p50, dict):
                amt = p50.get("monetaryValue", {}).get("amount")
                if amt:
                    min_sal = max_sal = float(amt)

        if description and len(description) > 2000:
            description = description[:2000] + "..."

        return JobResult(
            job_id=_job_id(listing_id, title, company),
            title=title,
            company=company,
            location=location,
            min_salary=min_sal,
            max_salary=max_sal,
            salary_currency=currency,
            salary_interval=pay_period.lower() if pay_period else None,
            job_url=url,
            platform="glassdoor",
            description=description or None,
            date_posted=_parse_date(age_in_days),
            job_type=None,
            is_remote="remote" in location.lower(),
        )
    except Exception as e:
        logger.warning(f"[glassdoor_custom] parse error: {e}")
        return None


def fetch_glassdoor(query: str, location: str = "India", results: int = 10) -> list[JobResult]:
    location_key = location.lower().strip()
    loc_id, loc_type = LOCATION_MAP.get(location_key, (115, "COUNTRY"))

    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers["gd-csrf-token"] = fallback_token

    all_jobs: list[JobResult] = []
    seen_ids: set[str] = set()
    page = 1
    cursor = None
    jobs_per_page = 30

    while len(all_jobs) < results:
        payload = {
            "operationName": "JobSearchResultsQuery",
            "query": query_template,
            "variables": {
                "excludeJobListingIds": [],
                "keyword": query,
                "locationId": loc_id,
                "locationType": loc_type,
                "numJobsToShow": min(jobs_per_page, results - len(all_jobs) + 5),
                "pageCursor": cursor,
                "pageNumber": page,
                "filterParams": [],
                "originalPageUrl": f"{GLASSDOOR_BASE}/Job/",
                "seoFriendlyUrlInput": f"{query.lower().replace(' ', '-')}-jobs",
                "parameterUrlInput": "KO0,20",
            },
        }
        try:
            resp = session.post(GRAPHQL_URL, json=payload, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"[glassdoor_custom] HTTP {resp.status_code}")
                break

            data = resp.json()
            job_listings_data = (data.get("data") or {}).get("jobListings") or {}
            listings = job_listings_data.get("jobListings") or []

            if not listings:
                break

            for listing in listings:
                job = _parse_listing(listing)
                if job and job.job_id not in seen_ids:
                    seen_ids.add(job.job_id)
                    all_jobs.append(job)
                    if len(all_jobs) >= results:
                        break

            # Get next page cursor
            pagination = job_listings_data.get("paginationCursors") or []
            next_cursors = [c for c in pagination if isinstance(c, dict) and c.get("cursor") and c.get("pageNumber") == page + 1]
            if next_cursors:
                cursor = next_cursors[0]["cursor"]
                page += 1
            else:
                break

        except Exception as e:
            logger.warning(f"[glassdoor_custom] error on page {page}: {e}")
            break

    logger.info(f"[glassdoor_custom] ✓ {len(all_jobs)} jobs for '{query}' in '{location}'")
    return all_jobs
