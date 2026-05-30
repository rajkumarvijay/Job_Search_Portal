"""
Gemini AI service — powers two features:
  1. AI job search (world-wide, all portals) via Gemini 1.5 Flash
  2. Resume ATS analysis with score + improvement suggestions
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)

# Model candidates — tried in order on FIRST actual content request
_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-pro",
]

_genai   = None   # google.generativeai module
_model_name: str = ""   # name of the working model, empty = not discovered yet


def _configure():
    """Configure genai once with the API key. Never probes models."""
    global _genai
    if _genai is not None:
        return _genai
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Go to Railway → backend service → Variables → add GEMINI_API_KEY."
        )
    genai.configure(api_key=api_key)
    _genai = genai
    return genai


def _call_with_fallback(prompt: str) -> str:
    """
    Try each model candidate until one returns a response.
    Caches the working model name so subsequent calls skip the loop.
    """
    global _model_name
    genai = _configure()

    candidates = [_model_name] + [m for m in _MODELS if m != _model_name] if _model_name else _MODELS

    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            resp  = model.generate_content(prompt)
            _model_name = name      # remember for next call
            logger.info(f"[Gemini] used model: {name}")
            return resp.text
        except Exception as e:
            logger.warning(f"[Gemini] model '{name}' failed: {type(e).__name__}: {e}")
            if "API_KEY" in str(e) or "api key" in str(e).lower():
                raise ValueError(f"Invalid GEMINI_API_KEY: {e}") from e

    raise RuntimeError(
        "All Gemini model candidates failed. "
        "Check your GEMINI_API_KEY and that it has access to Gemini models."
    )


def _extract_json(text: str) -> str:
    """Strip markdown code fences and return raw JSON."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. AI Job Search
# ─────────────────────────────────────────────────────────────────────────────
_JOB_SEARCH_PROMPT = """
You are an AI job search engine with knowledge of global job markets.
Find {count} real, currently active job listings for "{query}" in {location}.
Draw from LinkedIn, Naukri, Indeed, Glassdoor, Internshala, Wellfound, Dice, Monster, and other portals.
Return ONLY a valid JSON array — no markdown, no explanation, just JSON.

Each item must have EXACTLY these fields:
{{
  "title": "exact job title",
  "company": "company name",
  "location": "city, country",
  "job_url": "direct application URL (real URL if known, otherwise https://www.linkedin.com/jobs/search/?keywords={query_encoded})",
  "description": "2-3 sentence job description covering key responsibilities and requirements",
  "date_posted": "YYYY-MM-DD (estimate based on typical posting cycles)",
  "min_salary": null or numeric (annual, in local currency),
  "max_salary": null or numeric,
  "salary_currency": "INR" or "USD" or "EUR" etc,
  "job_type": "Full-time" or "Part-time" or "Contract" or "Remote" or "Internship",
  "is_remote": true or false,
  "platform": "linkedin" or "naukri" or "indeed" or "glassdoor" or "internshala" or "wellfound" or "dice" or "google"
}}

Include a diverse mix of companies (MNCs, startups, Indian companies if location is India).
Return exactly {count} jobs. No duplicates.
""".strip()


def _run_job_search(query: str, location: str, count: int) -> list[dict]:
    prompt = _JOB_SEARCH_PROMPT.format(
        query=query,
        query_encoded=query.replace(" ", "+"),
        location=location,
        count=count,
    )
    try:
        raw_text = _call_with_fallback(prompt)
        raw  = _extract_json(raw_text)
        jobs = json.loads(raw)
        if not isinstance(jobs, list):
            jobs = []
        logger.info(f"[Gemini] {len(jobs)} jobs returned for '{query}' in {location}")
        return jobs
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[Gemini] JSON parse error: {e}")
        return []
    except Exception as e:
        logger.warning(f"[Gemini] Job search error: {e}")
        return []


async def search_jobs_ai(
    query: str,
    location: str = "worldwide",
    results_wanted: int = 20,
) -> list[dict]:
    """Run Gemini job search in thread pool, return normalised job dicts."""
    loop = asyncio.get_event_loop()
    try:
        jobs = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_job_search, query, location, results_wanted),
            timeout=45,
        )
    except asyncio.TimeoutError:
        logger.warning("[Gemini] Job search timed out after 45s")
        return []

    # Normalise + assign job_id
    out = []
    for j in jobs:
        if not j.get("title") or not j.get("company"):
            continue
        raw_id = f"{j.get('job_url','')}{j.get('title','')}{j.get('company','')}".lower()
        j["job_id"] = hashlib.md5(raw_id.encode()).hexdigest()[:16]
        j.setdefault("salary_currency", "INR")
        j.setdefault("is_remote", False)
        j.setdefault("platform", "ai")
        out.append(j)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Resume ATS Analysis
# ─────────────────────────────────────────────────────────────────────────────
_RESUME_PROMPT = """
You are an expert ATS (Applicant Tracking System) analyst and career coach.
Analyse the following resume and return ONLY a valid JSON object — no markdown, no extra text.

Target role (if provided): "{target_role}"

Resume text:
\"\"\"
{resume_text}
\"\"\"

Return this exact JSON structure:
{{
  "ats_score": <integer 0-100>,
  "grade": "A" | "B" | "C" | "D" | "F",
  "summary": "2-3 sentence overall assessment",
  "strengths": ["strength1", "strength2", "strength3"],
  "improvements": [
    {{
      "category": "Keywords",
      "issue": "what is missing or wrong",
      "fix": "exactly what to add or change",
      "impact": "High" | "Medium" | "Low"
    }}
  ],
  "missing_keywords": ["keyword1", "keyword2", "keyword3"],
  "recommended_keywords": ["add these keywords to boost ATS score"],
  "format_issues": ["formatting problem 1", "formatting problem 2"],
  "quick_wins": ["do this immediately to improve score"]
}}

Score rubric:
- 90-100: ATS-optimised, ready to apply
- 70-89:  Good, minor tweaks needed
- 50-69:  Average, several improvements needed
- 30-49:  Needs significant work
- 0-29:   Major overhaul required

Be specific, actionable, and honest.
""".strip()


def _run_resume_analysis(resume_text: str, target_role: str) -> dict:
    truncated = resume_text[:8000]
    prompt = _RESUME_PROMPT.format(resume_text=truncated, target_role=target_role or "General")
    try:
        raw_text = _call_with_fallback(prompt)
        raw    = _extract_json(raw_text)
        result = json.loads(raw)
        logger.info(f"[Gemini] Resume analysed — ATS score: {result.get('ats_score')}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[Gemini] Resume JSON parse error: {e}")
        return {"error": "Could not parse Gemini response. Try again.", "ats_score": 0}
    except ValueError as e:
        # API key error — bubble up cleanly
        raise
    except Exception as e:
        logger.warning(f"[Gemini] Resume analysis error: {e}")
        return {"error": str(e), "ats_score": 0}


async def analyze_resume(resume_text: str, target_role: str = "") -> dict:
    """Analyse resume text with Gemini, return ATS report."""
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_resume_analysis, resume_text, target_role),
            timeout=60,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("[Gemini] Resume analysis timed out")
        return {"error": "Analysis timed out. Please try again.", "ats_score": 0}
