"""
AI-powered routes:
  GET  /api/v1/ai/jobs/search  — world-wide job search via Gemini
  POST /api/v1/ai/resume       — resume ATS analysis via Gemini
"""

import io
import logging
from fastapi import APIRouter, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from services.gemini_service import search_jobs_ai, analyze_resume

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])


# ── AI job search ─────────────────────────────────────────────────────────────
@router.get("/jobs/search")
async def ai_job_search(
    q:              str = Query(..., min_length=1),
    location:       str = Query("worldwide"),
    results_wanted: int = Query(20, ge=5, le=40),
):
    try:
        jobs = await search_jobs_ai(q, location, results_wanted)
        return {
            "query":    q,
            "location": location,
            "total":    len(jobs),
            "source":   "gemini-ai",
            "jobs":     jobs,
        }
    except ValueError as e:
        # GEMINI_API_KEY not set
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"AI job search error: {e}")
        raise HTTPException(status_code=500, detail="AI search failed")


# ── Resume ATS analysis ───────────────────────────────────────────────────────
@router.post("/resume")
async def analyse_resume(
    file:        UploadFile = File(...),
    target_role: str        = Form(default=""),
):
    # Validate file type
    allowed = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
               "text/plain"}
    ct = file.content_type or ""
    name = (file.filename or "").lower()

    if ct not in allowed and not any(name.endswith(ext) for ext in (".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, or TXT files are accepted")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:  # 5 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 5 MB)")

    resume_text = _extract_text(content, name, ct)
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the file")

    try:
        result = await analyze_resume(resume_text, target_role)
        return result
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Resume analysis error: {e}")
        raise HTTPException(status_code=500, detail="Resume analysis failed")


# ── Text extraction helpers ───────────────────────────────────────────────────
def _extract_text(content: bytes, filename: str, content_type: str) -> str:
    if filename.endswith(".pdf") or "pdf" in content_type:
        return _extract_pdf(content)
    if filename.endswith(".docx") or "wordprocessingml" in content_type:
        return _extract_docx(content)
    # Plain text fallback
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
        import zipfile, xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = [node.text for node in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if node.text]
        return " ".join(texts)
    except Exception as e:
        logger.warning(f"DOCX extraction failed: {e}")
        return ""
