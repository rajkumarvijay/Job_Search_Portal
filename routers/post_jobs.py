import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from db.database import get_db
from db.models import PostedJob
from schemas.job import PostJobRequest, PostedJobOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/post-jobs", tags=["post-jobs"])


# ── POST /api/v1/post-jobs/ ───────────────────────────────────────────────────
@router.post("/", status_code=201)
async def create_posted_job(
    payload: PostJobRequest,
    db: AsyncSession = Depends(get_db),
):
    """Anyone can post a job. No auth required — abuse protection via email field."""
    job_id = f"portal_{uuid.uuid4().hex[:12]}"
    job = PostedJob(
        job_id          = job_id,
        title           = payload.title.strip(),
        company         = payload.company.strip(),
        location        = payload.location.strip(),
        job_type        = payload.job_type,
        work_mode       = payload.work_mode,
        experience      = payload.experience,
        min_salary      = payload.min_salary,
        max_salary      = payload.max_salary,
        salary_currency = payload.salary_currency,
        description     = payload.description.strip(),
        skills          = payload.skills,
        contact_email   = payload.contact_email.strip().lower(),
        apply_url       = payload.apply_url or None,
        company_url     = payload.company_url or None,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    logger.info(f"New job posted: {job.title} @ {job.company} (id={job_id})")
    return {"success": True, "job_id": job_id, "message": "Job posted successfully!"}


# ── GET /api/v1/post-jobs/ ────────────────────────────────────────────────────
@router.get("/", response_model=list[PostedJobOut])
async def list_posted_jobs(
    q:    str = Query("", description="Filter by keyword"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List all active posted jobs, optionally filtered by keyword."""
    stmt = select(PostedJob).where(PostedJob.is_active == True)
    if q.strip():
        kw = f"%{q.strip()}%"
        stmt = stmt.where(
            or_(
                PostedJob.title.ilike(kw),
                PostedJob.company.ilike(kw),
                PostedJob.description.ilike(kw),
                PostedJob.skills.ilike(kw),
                PostedJob.location.ilike(kw),
            )
        )
    stmt = stmt.order_by(PostedJob.posted_at.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


# ── GET /api/v1/post-jobs/{job_id} ───────────────────────────────────────────
@router.get("/{job_id}", response_model=PostedJobOut)
async def get_posted_job(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PostedJob).where(PostedJob.job_id == job_id, PostedJob.is_active == True)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
