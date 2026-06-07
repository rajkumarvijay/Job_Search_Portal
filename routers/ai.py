"""
AI-powered routes:
  GET  /api/v1/ai/jobs/search        — world-wide job search via Gemini
  POST /api/v1/ai/resume             — resume ATS analysis via Gemini
  POST /api/v1/ai/resume/full        — ATS analysis + AI job recommendations
"""

import asyncio
import io
import logging
from fastapi import APIRouter, Query, UploadFile, File, Form, HTTPException
from services.gemini_service import (
    search_jobs_ai,
    analyze_resume,
    get_resume_job_recommendations,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])


# ── AI job search ──────────────────────────────────────────────────────────────
@router.get("/jobs/search")
async def ai_job_search(
    q:              str = Query(..., min_length=1),
    location:       str = Query("worldwide"),
    results_wanted: int = Query(20, ge=5, le=40),
):
    try:
        jobs = await search_jobs_ai(q, location, results_wanted)
        return {"query": q, "location": location, "total": len(jobs), "source": "gemini-ai", "jobs": jobs}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"AI job search error: {e}")
        raise HTTPException(status_code=500, detail="AI search failed")


# ── Resume ATS analysis (score only) ──────────────────────────────────────────
@router.post("/resume")
async def analyse_resume(
    file:        UploadFile = File(...),
    target_role: str        = Form(default=""),
):
    """ATS analysis only — fast path used by the search-bar button."""
    resume_text = await _read_resume(file)
    try:
        result = await analyze_resume(resume_text, target_role)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Resume analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {e}")


# ── Resume full analysis + job recommendations ─────────────────────────────────
@router.post("/resume/full")
async def analyse_resume_full(
    file:        UploadFile = File(...),
    target_role: str        = Form(default=""),
    job_count:   int        = Form(default=6),
):
    """
    Full analysis: ATS report + AI-curated job recommendations in parallel.
    Used by the dedicated /resume page.
    """
    resume_text = await _read_resume(file)
    try:
        # Step 1 — run ATS analysis first (jobs need skills/roles from it)
        ats_result = await analyze_resume(resume_text, target_role)

        if ats_result.get("error"):
            return {**ats_result, "recommended_jobs": []}

        # Step 2 — fetch job recommendations in parallel with returning ATS data
        skills = ats_result.get("top_skills", [])
        roles  = ats_result.get("recommended_roles", [])
        level  = ats_result.get("experience_level", "Mid-Level")

        jobs = await get_resume_job_recommendations(
            skills=skills,
            roles=roles,
            level=level,
            target_role=target_role,
            count=max(3, min(job_count, 9)),
        )

        return {**ats_result, "recommended_jobs": jobs}

    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Full resume analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────
async def _read_resume(file: UploadFile) -> str:
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }
    ct   = file.content_type or ""
    name = (file.filename or "").lower()

    if ct not in allowed_types and not any(name.endswith(e) for e in (".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, or TXT files are accepted")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB)")

    text = _extract_text(content, name, ct)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the file. Make sure the file is not scanned/image-only.")
    return text


def _extract_text(content: bytes, filename: str, content_type: str) -> str:
    if filename.endswith(".pdf") or "pdf" in content_type:
        return _extract_pdf(content)
    if filename.endswith(".docx") or "wordprocessingml" in content_type:
        return _extract_docx(content)
    return content.decode("utf-8", errors="ignore")


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ""


def _extract_docx(content: bytes) -> str:
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
        texts = [
            node.text
            for node in tree.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
            )
            if node.text
        ]
        return " ".join(texts)
    except Exception as e:
        logger.warning(f"DOCX extraction failed: {e}")
        return ""
