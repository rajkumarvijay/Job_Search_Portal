"""
Fallback job fetchers for portals that are blocked (ZipRecruiter, Google Jobs).
Uses free public job APIs that require no authentication.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import requests

from schemas.job import JobResult

logger = logging.getLogger(__name__)

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_URL = "https://arbeitnow.com/api/job-board-api"


def _job_id(source: str, uid: str, title: str, company: str) -> str:
    raw = f"{source}-{uid}-{title}-{company}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _truncate(text: str, limit: int = 2000) -> str:
    return (text[:limit] + "...") if len(text) > limit else text


def fetch_remotive(query: str, results: int = 10, platform_label: str = "google") -> list[JobResult]:
    """Fetch remote jobs from Remotive API."""
    all_jobs: list[JobResult] = []
    seen_ids: set[str] = set()

    try:
        resp = requests.get(
            REMOTIVE_URL,
            params={"search": query, "limit": results * 2},
            timeout=20,
        )
        if resp.status_code != 200:
            logger.warning(f"[remotive] HTTP {resp.status_code}")
            return []

        data = resp.json()
        jobs = data.get("jobs") or []

        for job in jobs:
            try:
                title = job.get("title") or "Untitled"
                company = job.get("company_name") or "Unknown"
                location = job.get("candidate_required_location") or "Remote"
                url = job.get("url") or None
                description = _truncate(job.get("description") or "")
                date_str = str(job.get("publication_date") or "")[:10] or None
                job_type = job.get("job_type") or None
                uid = str(job.get("id") or "")

                jid = _job_id(platform_label, uid, title, company)
                if jid in seen_ids:
                    continue
                seen_ids.add(jid)

                all_jobs.append(JobResult(
                    job_id=jid,
                    title=title,
                    company=company,
                    location=location,
                    min_salary=None,
                    max_salary=None,
                    salary_currency="USD",
                    salary_interval=None,
                    job_url=url,
                    platform=platform_label,
                    description=description or None,
                    date_posted=date_str,
                    job_type=job_type,
                    is_remote=True,
                ))
                if len(all_jobs) >= results:
                    break
            except Exception as e:
                logger.warning(f"[remotive] parse error: {e}")

    except Exception as e:
        logger.warning(f"[remotive] fetch error: {e}")

    logger.info(f"[remotive→{platform_label}] ✓ {len(all_jobs)} jobs for '{query}'")
    return all_jobs


def fetch_arbeitnow(query: str, results: int = 10, platform_label: str = "ziprecruiter") -> list[JobResult]:
    """Fetch jobs from Arbeitnow API (EU/remote focused)."""
    all_jobs: list[JobResult] = []
    seen_ids: set[str] = set()

    try:
        resp = requests.get(
            ARBEITNOW_URL,
            params={"search": query},
            timeout=20,
        )
        if resp.status_code != 200:
            logger.warning(f"[arbeitnow] HTTP {resp.status_code}")
            return []

        data = resp.json()
        jobs = data.get("data") or []

        for job in jobs:
            try:
                title = job.get("title") or "Untitled"
                company = job.get("company_name") or "Unknown"
                location = job.get("location") or "Remote"
                url = job.get("url") or None
                description = _truncate(job.get("description") or "")
                created_at = job.get("created_at") or ""
                date_str = str(created_at)[:10] if created_at else None
                job_types = job.get("job_types") or []
                job_type = job_types[0] if job_types else None
                is_remote = bool(job.get("remote"))
                slug = job.get("slug") or ""

                jid = _job_id(platform_label, slug, title, company)
                if jid in seen_ids:
                    continue
                seen_ids.add(jid)

                all_jobs.append(JobResult(
                    job_id=jid,
                    title=title,
                    company=company,
                    location=location,
                    min_salary=None,
                    max_salary=None,
                    salary_currency="USD",
                    salary_interval=None,
                    job_url=url,
                    platform=platform_label,
                    description=description or None,
                    date_posted=date_str,
                    job_type=job_type,
                    is_remote=is_remote,
                ))
                if len(all_jobs) >= results:
                    break
            except Exception as e:
                logger.warning(f"[arbeitnow] parse error: {e}")

    except Exception as e:
        logger.warning(f"[arbeitnow] fetch error: {e}")

    logger.info(f"[arbeitnow→{platform_label}] ✓ {len(all_jobs)} jobs for '{query}'")
    return all_jobs
