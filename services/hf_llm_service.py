"""
HuggingFace LLM service — replaces Gemini for all AI resume features.

Model:  mistralai/Mistral-7B-Instruct-v0.3  (free on HF Inference API)
Auth:   HUGGINGFACE_API_TOKEN env var

Implements the same public API as gemini_service so routers need no changes:
  analyze_resume(resume_text, target_role)
  match_resume_to_job(resume_text, job_description)
  generate_cover_letter(resume_text, job_description, ...)
  get_resume_job_recommendations(skills, roles, level, ...)
  search_jobs_ai(query, location, results_wanted)
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

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# ── Client setup ─────────────────────────────────────────────────────────────

def _get_client():
    from huggingface_hub import InferenceClient
    token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    if not token:
        raise ValueError(
            "HUGGINGFACE_API_TOKEN is not set. "
            "Add it in Railway → backend service → Variables."
        )
    return InferenceClient(model=HF_MODEL, token=token)


def _call_hf(prompt: str, max_tokens: int = 4096) -> str:
    """Call Mistral-7B via HF Inference API, return raw text."""
    client = _get_client()
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return response.choices[0].message.content


def _extract_json_str(text: str) -> str:
    """Strip markdown fences, return raw JSON string."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find outermost { } or [ ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end   = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
    return text.strip()


# ── 1. Resume ATS Analysis ────────────────────────────────────────────────────

_RESUME_PROMPT = """<s>[INST]
You are a senior ATS analyst and tech career coach with 15+ years of recruitment experience.
Analyse the resume below and return ONLY a valid JSON object — no markdown, no explanation.

Target role: "{target_role}"

Resume:
{resume_text}

Return EXACTLY this JSON (all fields required, use empty arrays/null where absent):
{{
  "ats_score": <integer 0-100>,
  "grade": "A+" or "A" or "B" or "C" or "D" or "F",
  "experience_level": "Fresher" or "Junior" or "Mid-Level" or "Senior" or "Lead/Principal",
  "years_experience": <number or null>,
  "summary": "3-4 sentence honest assessment of strengths, ATS readiness, and biggest gap",
  "section_scores": {{
    "contact_info": <0-10>,
    "professional_summary": <0-15>,
    "work_experience": <0-30>,
    "skills": <0-20>,
    "education": <0-15>,
    "keywords_and_ats": <0-10>
  }},
  "extracted_skills": {{
    "technical": ["languages, frameworks, databases"],
    "tools": ["Git, Docker, Jenkins, etc."],
    "soft": ["Leadership, Communication, etc."],
    "certifications": ["AWS, GCP, PMP, etc."]
  }},
  "top_skills": ["top 10 skills ranked by relevance to target role"],
  "experience_breakdown": [
    {{"company": "...", "title": "...", "duration": "...", "is_quantified": true, "key_achievements": ["..."], "impact_score": <1-10>}}
  ],
  "projects": [
    {{"name": "...", "tech_stack": ["..."], "description": "...", "has_metrics": true, "github_mentioned": false}}
  ],
  "education": [
    {{"degree": "...", "institution": "...", "year": "...", "gpa_cgpa": "..."}}
  ],
  "recommended_roles": ["4-6 specific job titles this person is best suited for"],
  "strengths": [
    {{"title": "...", "explanation": "specific evidence from resume"}}
  ],
  "improvements": [
    {{"category": "Keywords", "issue": "...", "fix": "...", "impact": "High"}}
  ],
  "missing_keywords": ["important ATS keywords absent from resume"],
  "recommended_keywords": ["add these to boost ATS pass rate"],
  "keyword_density": {{"present": ["..."], "overused": ["..."]}},
  "format_issues": ["specific ATS formatting problems"],
  "quick_wins": [
    {{"action": "...", "time_required": "5 minutes", "score_impact": "+3 points"}}
  ],
  "indian_job_market_tips": ["2-3 tips specific to Indian job market"]
}}
[/INST]"""

_DEFAULTS = {
    "section_scores":     {},
    "top_skills":         [],
    "recommended_roles":  [],
    "experience_level":   "Unknown",
    "years_experience":   None,
    "strengths":          [],
    "improvements":       [],
    "missing_keywords":   [],
    "recommended_keywords": [],
    "format_issues":      [],
    "quick_wins":         [],
    "extracted_skills":   {"technical": [], "soft": [], "tools": [], "certifications": []},
    "experience_breakdown": [],
    "projects":           [],
    "education":          [],
    "keyword_density":    {"present": [], "overused": []},
    "indian_job_market_tips": [],
}


def _run_resume_analysis(resume_text: str, target_role: str) -> dict:
    prompt = _RESUME_PROMPT.format(
        resume_text=resume_text[:7000],
        target_role=target_role or "General",
    )
    try:
        raw_text = _call_hf(prompt)
        result   = json.loads(_extract_json_str(raw_text))
        for k, v in _DEFAULTS.items():
            result.setdefault(k, v)
        logger.info(f"[HF] Resume analysed — ATS score: {result.get('ats_score')}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[HF] Resume JSON parse error: {e}")
        return {"error": "Could not parse model response. Try again.", "ats_score": 0}
    except Exception as e:
        logger.warning(f"[HF] Resume analysis error: {e}")
        return {"error": str(e), "ats_score": 0}


async def analyze_resume(resume_text: str, target_role: str = "") -> dict:
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_resume_analysis, resume_text, target_role),
            timeout=90,
        )
    except asyncio.TimeoutError:
        logger.warning("[HF] Resume analysis timed out")
        return {"error": "Analysis timed out. Please try again.", "ats_score": 0}


# ── 2. Job Match Score ────────────────────────────────────────────────────────

_JOB_MATCH_PROMPT = """<s>[INST]
You are an expert technical recruiter. Score how well this resume matches the job description.
Return ONLY a valid JSON object — no markdown, no explanation.

Resume:
{resume_text}

Job Description:
{job_description}

Return EXACTLY this JSON:
{{
  "match_score": <integer 0-100>,
  "match_grade": "Excellent" or "Strong" or "Good" or "Fair" or "Weak",
  "match_summary": "3-4 sentence honest assessment",
  "matched_skills": [
    {{"skill": "...", "found_in_resume": "...", "jd_requirement": "...", "proficiency": "Expert"}}
  ],
  "missing_skills": [
    {{"skill": "...", "importance": "Must-have", "jd_context": "...", "gap_size": "Large"}}
  ],
  "strengths": [
    {{"title": "...", "detail": "...", "impact": "High"}}
  ],
  "learning_recommendations": [
    {{"skill": "...", "reason": "...", "resource": "...", "timeframe": "2-4 weeks", "priority": "High"}}
  ],
  "experience_match": {{
    "required_years": <number or null>,
    "candidate_years": <number or null>,
    "verdict": "Matches"
  }},
  "role_fit_tags": ["Strong Backend", "Needs Cloud Exp"],
  "quick_actions": ["top 3 things to improve match right now"]
}}
[/INST]"""


def _run_job_match(resume_text: str, job_description: str) -> dict:
    prompt = _JOB_MATCH_PROMPT.format(
        resume_text=resume_text[:5000],
        job_description=job_description[:2500],
    )
    try:
        raw_text = _call_hf(prompt)
        result   = json.loads(_extract_json_str(raw_text))
        result.setdefault("match_score", 0)
        result.setdefault("match_grade", "Fair")
        result.setdefault("match_summary", "")
        result.setdefault("matched_skills", [])
        result.setdefault("missing_skills", [])
        result.setdefault("strengths", [])
        result.setdefault("learning_recommendations", [])
        result.setdefault("experience_match", {})
        result.setdefault("role_fit_tags", [])
        result.setdefault("quick_actions", [])
        logger.info(f"[HF] Job match score: {result.get('match_score')}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[HF] Job match JSON parse error: {e}")
        return {"error": "Could not parse model response. Try again.", "match_score": 0}
    except Exception as e:
        logger.warning(f"[HF] Job match error: {e}")
        return {"error": str(e), "match_score": 0}


async def match_resume_to_job(resume_text: str, job_description: str) -> dict:
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_job_match, resume_text, job_description),
            timeout=90,
        )
    except asyncio.TimeoutError:
        logger.warning("[HF] Job match timed out")
        return {"error": "Analysis timed out. Please try again.", "match_score": 0}


# ── 3. Cover Letter Generator ─────────────────────────────────────────────────

_COVER_LETTER_PROMPT = """<s>[INST]
You are an expert career coach. Generate a compelling professional cover letter.
Return ONLY a valid JSON object — no markdown, no explanation.

Candidate: {candidate_name}
Company: {company_name}
Role: {job_title}
Tone: {tone}

Resume:
{resume_text}

Job Description:
{job_description}

Return EXACTLY this JSON:
{{
  "cover_letter": "full cover letter with paragraph breaks using \\n\\n",
  "subject_line": "suggested email subject line",
  "key_matches": ["3-5 skills from resume that match this JD"],
  "tone_used": "{tone}",
  "word_count": <integer>,
  "tips": ["2-3 personalisation tips before sending"]
}}
[/INST]"""


def _run_cover_letter(
    resume_text: str, job_description: str,
    candidate_name: str, company_name: str, job_title: str, tone: str,
) -> dict:
    prompt = _COVER_LETTER_PROMPT.format(
        resume_text=resume_text[:5000],
        job_description=job_description[:2500],
        candidate_name=candidate_name or "the candidate",
        company_name=company_name or "the company",
        job_title=job_title or "the role",
        tone=tone,
    )
    try:
        raw_text = _call_hf(prompt, max_tokens=2048)
        result   = json.loads(_extract_json_str(raw_text))
        result.setdefault("cover_letter", "")
        result.setdefault("subject_line", f"Application for {job_title}")
        result.setdefault("key_matches", [])
        result.setdefault("tone_used", tone)
        result.setdefault("word_count", len(result.get("cover_letter", "").split()))
        result.setdefault("tips", [])
        logger.info(f"[HF] Cover letter generated — {result.get('word_count')} words")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[HF] Cover letter JSON parse error: {e}")
        return {"error": "Could not parse model response. Try again.", "cover_letter": ""}
    except Exception as e:
        logger.warning(f"[HF] Cover letter error: {e}")
        return {"error": str(e), "cover_letter": ""}


async def generate_cover_letter(
    resume_text: str, job_description: str,
    candidate_name: str = "", company_name: str = "",
    job_title: str = "", tone: str = "Professional",
) -> dict:
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _run_cover_letter,
                resume_text, job_description,
                candidate_name, company_name, job_title, tone,
            ),
            timeout=90,
        )
    except asyncio.TimeoutError:
        logger.warning("[HF] Cover letter timed out")
        return {"error": "Generation timed out. Please try again.", "cover_letter": ""}


# ── 4. Job Recommendations ────────────────────────────────────────────────────

_RECO_PROMPT = """<s>[INST]
You are a job matching engine. Based on this candidate profile, generate {count} relevant job opportunities.
Return ONLY a valid JSON array — no markdown, no explanation.

Skills: {skills}
Roles: {roles}
Level: {level}
Target: {target_role}

Each job must have EXACTLY these fields:
{{
  "title": "specific job title",
  "company": "real company name",
  "location": "city, India",
  "job_url": "https://www.linkedin.com/jobs/search/?keywords={query_encoded}",
  "description": "2 sentence match explanation",
  "min_salary": <number in LPA or null>,
  "max_salary": <number in LPA or null>,
  "salary_currency": "INR",
  "job_type": "Full-time",
  "is_remote": false,
  "platform": "linkedin",
  "match_score": <integer 70-99>,
  "match_reason": "1 sentence why this matches the candidate"
}}

Return exactly {count} jobs.
[/INST]"""


def _run_job_recommendations(
    skills: list, roles: list, level: str, target_role: str, count: int,
) -> list:
    if not skills and not roles:
        return []
    query_encoded = (roles[0] if roles else "software engineer").replace(" ", "+")
    prompt = _RECO_PROMPT.format(
        skills=", ".join(skills[:8]),
        roles=", ".join(roles[:4]),
        level=level,
        target_role=target_role or (roles[0] if roles else "Software Engineer"),
        count=count,
        query_encoded=query_encoded,
    )
    try:
        raw_text = _call_hf(prompt)
        jobs     = json.loads(_extract_json_str(raw_text))
        if not isinstance(jobs, list):
            return []
        for j in jobs:
            raw_id = f"{j.get('job_url','')}{j.get('title','')}{j.get('company','')}".lower()
            j["job_id"] = hashlib.md5(raw_id.encode()).hexdigest()[:16]
            j.setdefault("salary_currency", "INR")
            j.setdefault("is_remote", False)
            j.setdefault("platform", "ai")
            j.setdefault("match_score", 80)
            j.setdefault("match_reason", "Matches your skills and experience")
        logger.info(f"[HF] {len(jobs)} job recommendations generated")
        return jobs
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[HF] Job reco JSON parse error: {e}")
        return []
    except Exception as e:
        logger.warning(f"[HF] Job reco error: {e}")
        return []


async def get_resume_job_recommendations(
    skills: list, roles: list, level: str,
    target_role: str = "", count: int = 6,
) -> list:
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _run_job_recommendations,
                skills, roles, level, target_role, count,
            ),
            timeout=60,
        )
    except asyncio.TimeoutError:
        logger.warning("[HF] Job recommendations timed out")
        return []


# ── 5. AI Job Search ─────────────────────────────────────────────────────────

_JOB_SEARCH_PROMPT = """<s>[INST]
You are an AI job search engine. Find {count} real active job listings for "{query}" in {location}.
Return ONLY a valid JSON array — no markdown, no explanation.

Each job must have EXACTLY these fields:
{{
  "title": "exact job title",
  "company": "company name",
  "location": "city, country",
  "job_url": "https://www.linkedin.com/jobs/search/?keywords={query_encoded}",
  "description": "2-3 sentence job description",
  "date_posted": "YYYY-MM-DD",
  "min_salary": null or number,
  "max_salary": null or number,
  "salary_currency": "INR",
  "job_type": "Full-time",
  "is_remote": false,
  "platform": "linkedin"
}}

Return exactly {count} jobs. No duplicates.
[/INST]"""


def _run_job_search(query: str, location: str, count: int) -> list:
    prompt = _JOB_SEARCH_PROMPT.format(
        query=query,
        query_encoded=query.replace(" ", "+"),
        location=location,
        count=count,
    )
    try:
        raw_text = _call_hf(prompt)
        jobs     = json.loads(_extract_json_str(raw_text))
        if not isinstance(jobs, list):
            return []
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
        logger.info(f"[HF] {len(out)} AI jobs returned for '{query}'")
        return out
    except Exception as e:
        logger.warning(f"[HF] Job search error: {e}")
        return []


async def search_jobs_ai(
    query: str, location: str = "worldwide", results_wanted: int = 20,
) -> list:
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_job_search, query, location, results_wanted),
            timeout=45,
        )
    except asyncio.TimeoutError:
        logger.warning("[HF] Job search timed out")
        return []
