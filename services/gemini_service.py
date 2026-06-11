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
You are a senior ATS analyst and tech career coach with 15+ years of recruitment experience at top Indian and global tech companies.
Deeply analyse the resume below and return ONLY a valid JSON object — no markdown, no explanation, no prose outside the JSON.

Target role (if provided): "{target_role}"

Resume text:
\"\"\"
{resume_text}
\"\"\"

Return EXACTLY this JSON structure (every field is required — use empty arrays/null where data is absent):
{{
  "ats_score": <integer 0-100>,
  "grade": "A+" | "A" | "B" | "C" | "D" | "F",
  "experience_level": "Fresher" | "Junior" | "Mid-Level" | "Senior" | "Lead/Principal",
  "years_experience": <number or null>,
  "summary": "3-4 sentence honest overall assessment: key strengths, experience level, ATS readiness, and single biggest gap",

  "section_scores": {{
    "contact_info":           <integer 0-10>,
    "professional_summary":   <integer 0-15>,
    "work_experience":        <integer 0-30>,
    "skills":                 <integer 0-20>,
    "education":              <integer 0-15>,
    "keywords_and_ats":       <integer 0-10>
  }},

  "extracted_skills": {{
    "technical":  ["all technical/hard skills found — languages, frameworks, tools, platforms, databases, cloud"],
    "soft":       ["soft skills explicitly mentioned — leadership, communication, teamwork, etc."],
    "tools":      ["specific tools, software, IDEs, CI/CD, monitoring — e.g. Git, Docker, Jenkins, Grafana"],
    "certifications": ["any certifications or credentials mentioned"]
  }},

  "top_skills": ["top 10 most relevant skills for the target role, ranked by relevance"],

  "experience_breakdown": [
    {{
      "company":       "company name",
      "title":         "job title",
      "duration":      "e.g. Jan 2022 – Mar 2024 (2 yrs 2 mo)",
      "is_quantified": true | false,
      "key_achievements": ["up to 3 notable achievements or responsibilities extracted from resume"],
      "impact_score":  <integer 1-10 — how well impact/results are shown>
    }}
  ],

  "projects": [
    {{
      "name":         "project name",
      "tech_stack":   ["technologies used"],
      "description":  "1-2 sentence summary of what was built and its impact",
      "has_metrics":  true | false,
      "github_mentioned": true | false
    }}
  ],

  "education": [
    {{
      "degree":      "e.g. B.Tech Computer Science",
      "institution": "college/university name",
      "year":        "graduation year or expected year",
      "gpa_cgpa":    "GPA/CGPA if mentioned, else null"
    }}
  ],

  "recommended_roles": ["4-6 specific job titles this person is best suited for, most relevant first"],

  "strengths": [
    {{
      "title":       "strength name",
      "explanation": "specific evidence from the resume supporting this strength"
    }}
  ],

  "improvements": [
    {{
      "category": "Keywords" | "Formatting" | "Work Experience" | "Skills Section" | "Summary" | "Quantification" | "Education" | "ATS Compatibility" | "Projects",
      "issue":    "specific problem found with direct reference to resume content",
      "fix":      "exact actionable fix — include a rewritten example sentence where helpful",
      "impact":   "High" | "Medium" | "Low"
    }}
  ],

  "missing_keywords": ["important ATS keywords absent from resume that are critical for the target role"],
  "recommended_keywords": ["add these exact words/phrases to significantly boost ATS pass rate — include both acronyms and full forms"],

  "keyword_density": {{
    "present":  ["keywords already in resume that are ATS-valuable"],
    "overused": ["words used too frequently that dilute impact"]
  }},

  "format_issues": ["specific formatting problems that hurt ATS parsing — e.g. tables, headers, graphics, fonts"],

  "quick_wins": [
    {{
      "action":         "specific change to make",
      "time_required":  "e.g. 5 minutes",
      "score_impact":   "+3-5 points"
    }}
  ],

  "indian_job_market_tips": ["2-3 tips specific to the Indian job market — FAANG India, product startups, service companies, Naukri/LinkedIn optimisation"]
}}

Scoring rubric:
- 90-100 (A+/A): ATS-optimised, strong keywords, quantified achievements, clean format, ready to apply
- 75-89  (B):    Good resume, minor keyword or formatting gaps
- 55-74  (C):    Average, several improvements needed to pass ATS filters
- 35-54  (D):    Weak ATS compatibility, needs significant rework
- 0-34   (F):    Major overhaul required — likely filtered out before human review

Be brutally honest, specific, and actionable. Every improvement must reference actual content from the resume.
Tailor all feedback to the target role when provided. If no target role, infer from the resume.
""".strip()


def _run_resume_analysis(resume_text: str, target_role: str) -> dict:
    truncated = resume_text[:8000]
    prompt = _RESUME_PROMPT.format(resume_text=truncated, target_role=target_role or "General")
    try:
        raw_text = _call_with_fallback(prompt)
        raw    = _extract_json(raw_text)
        result = json.loads(raw)

        # Back-fill all fields so callers never get KeyError
        result.setdefault("section_scores", {})
        result.setdefault("top_skills", [])
        result.setdefault("recommended_roles", [])
        result.setdefault("experience_level", "Unknown")
        result.setdefault("years_experience", None)
        result.setdefault("strengths", [])
        result.setdefault("improvements", [])
        result.setdefault("missing_keywords", [])
        result.setdefault("recommended_keywords", [])
        result.setdefault("format_issues", [])
        result.setdefault("quick_wins", [])
        # New enriched fields
        result.setdefault("extracted_skills", {"technical": [], "soft": [], "tools": [], "certifications": []})
        result.setdefault("experience_breakdown", [])
        result.setdefault("projects", [])
        result.setdefault("education", [])
        result.setdefault("keyword_density", {"present": [], "overused": []})
        result.setdefault("indian_job_market_tips", [])

        logger.info(f"[Gemini] Resume analysed — ATS score: {result.get('ats_score')}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[Gemini] Resume JSON parse error: {e}")
        return {"error": "Could not parse Gemini response. Try again.", "ats_score": 0}
    except ValueError as e:
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Resume → Job Recommendations
# ─────────────────────────────────────────────────────────────────────────────
_RECO_PROMPT = """
You are a job matching engine. Based on the extracted resume data, find {count} highly relevant
real job opportunities and return ONLY a valid JSON array — no markdown, no explanation.

Candidate profile:
- Top skills: {skills}
- Recommended roles: {roles}
- Experience level: {level}
- Target role: {target_role}

Each job object must have EXACTLY these fields:
{{
  "title": "specific job title",
  "company": "real company name",
  "location": "city, country (India-focused unless skills suggest otherwise)",
  "job_url": "https://www.linkedin.com/jobs/search/?keywords={query_encoded}",
  "description": "2 sentence description of what makes this a strong match for this candidate",
  "min_salary": null or number (annual INR in lakhs, e.g. 12 means ₹12 LPA),
  "max_salary": null or number,
  "salary_currency": "INR",
  "job_type": "Full-time",
  "is_remote": true or false,
  "platform": "linkedin" or "naukri" or "indeed",
  "match_score": <integer 70-99 — how well this job matches the candidate's profile>,
  "match_reason": "1 sentence explaining why this is a great match"
}}

Return exactly {count} jobs. Prioritise relevance and match quality.
""".strip()


def _run_job_recommendations(
    skills: list[str],
    roles: list[str],
    level: str,
    target_role: str,
    count: int,
) -> list[dict]:
    if not skills and not roles:
        return []
    query_encoded = (roles[0] if roles else "software engineer").replace(" ", "+")
    prompt = _RECO_PROMPT.format(
        skills=", ".join(skills[:8]),
        roles=", ".join(roles[:4]),
        level=level,
        target_role=target_role or roles[0] if roles else "Software Engineer",
        count=count,
        query_encoded=query_encoded,
    )
    try:
        raw_text = _call_with_fallback(prompt)
        raw  = _extract_json(raw_text)
        jobs = json.loads(raw)
        if not isinstance(jobs, list):
            return []
        # Assign unique job_ids
        for j in jobs:
            raw_id = f"{j.get('job_url','')}{j.get('title','')}{j.get('company','')}".lower()
            j["job_id"] = hashlib.md5(raw_id.encode()).hexdigest()[:16]
            j.setdefault("salary_currency", "INR")
            j.setdefault("is_remote", False)
            j.setdefault("platform", "ai")
            j.setdefault("match_score", 80)
            j.setdefault("match_reason", "Matches your skills and experience level")
        logger.info(f"[Gemini] {len(jobs)} job recommendations generated")
        return jobs
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[Gemini] Job reco JSON parse error: {e}")
        return []
    except Exception as e:
        logger.warning(f"[Gemini] Job reco error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. AI Cover Letter Generator
# ─────────────────────────────────────────────────────────────────────────────
_COVER_LETTER_PROMPT = """
You are an expert career coach and professional writer who has helped thousands of candidates land jobs at top Indian and global companies.

Generate a compelling, personalised professional cover letter based on the resume and job description below.

Candidate details:
- Name: {candidate_name}
- Target Company: {company_name}
- Target Role: {job_title}

Resume text:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Tone preference: {tone}

Instructions:
1. Write a 3-4 paragraph cover letter (300-400 words)
2. Opening paragraph: Hook with genuine enthusiasm for this specific role/company + your strongest relevant achievement
3. Middle paragraphs (1-2): Match 3-4 specific skills/experiences from the resume to explicit requirements in the JD. Use numbers/metrics where resume provides them.
4. Closing paragraph: Clear call to action, express eagerness for interview
5. Tailor language to match the JD's tone and keywords — use the same terminology the JD uses
6. Do NOT use clichés like "I am writing to apply", "I believe I am a perfect fit", "Please find attached"
7. Sound human and confident, not robotic or over-formal
8. If company name is "Unknown Company", write a generic but highly professional letter

Return ONLY a valid JSON object — no markdown, no prose outside the JSON:
{{
  "cover_letter": "the full cover letter text with proper paragraph breaks using \\n\\n",
  "subject_line": "suggested email subject line for this application",
  "key_matches": ["3-5 specific skills/experiences from resume that match this JD"],
  "tone_used": "{tone}",
  "word_count": <integer>,
  "tips": ["2-3 personalisation tips to make this letter even stronger before sending"]
}}
""".strip()


def _run_cover_letter(
    resume_text: str,
    job_description: str,
    candidate_name: str,
    company_name: str,
    job_title: str,
    tone: str,
) -> dict:
    prompt = _COVER_LETTER_PROMPT.format(
        resume_text=resume_text[:6000],
        job_description=job_description[:3000],
        candidate_name=candidate_name or "the candidate",
        company_name=company_name or "Unknown Company",
        job_title=job_title or "the role",
        tone=tone,
    )
    try:
        raw_text = _call_with_fallback(prompt)
        raw = _extract_json(raw_text)
        result = json.loads(raw)
        result.setdefault("cover_letter", "")
        result.setdefault("subject_line", f"Application for {job_title} — {candidate_name}")
        result.setdefault("key_matches", [])
        result.setdefault("tone_used", tone)
        result.setdefault("word_count", len(result.get("cover_letter", "").split()))
        result.setdefault("tips", [])
        logger.info(f"[Gemini] Cover letter generated — {result.get('word_count')} words")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[Gemini] Cover letter JSON parse error: {e}")
        return {"error": "Could not parse Gemini response. Try again.", "cover_letter": ""}
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"[Gemini] Cover letter error: {e}")
        return {"error": str(e), "cover_letter": ""}


async def generate_cover_letter(
    resume_text: str,
    job_description: str,
    candidate_name: str = "",
    company_name: str = "",
    job_title: str = "",
    tone: str = "Professional",
) -> dict:
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _run_cover_letter,
                resume_text, job_description,
                candidate_name, company_name, job_title, tone,
            ),
            timeout=60,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("[Gemini] Cover letter generation timed out")
        return {"error": "Generation timed out. Please try again.", "cover_letter": ""}


async def get_resume_job_recommendations(
    skills: list[str],
    roles: list[str],
    level: str,
    target_role: str = "",
    count: int = 6,
) -> list[dict]:
    """Return AI-curated job recommendations for a candidate profile."""
    loop = asyncio.get_event_loop()
    try:
        jobs = await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _run_job_recommendations,
                skills, roles, level, target_role, count,
            ),
            timeout=45,
        )
        return jobs
    except asyncio.TimeoutError:
        logger.warning("[Gemini] Job recommendations timed out")
        return []
