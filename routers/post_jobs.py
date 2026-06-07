import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from db.database import get_db
from db.models import PostedJob
from schemas.job import PostJobRequest, PostedJobOut, EditJobRequest, DeleteJobRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/post-jobs", tags=["post-jobs"])


# ── helpers ───────────────────────────────────────────────────────────────────
async def _get_active(job_id: str, db: AsyncSession) -> PostedJob:
    result = await db.execute(
        select(PostedJob).where(PostedJob.job_id == job_id, PostedJob.is_active == True)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _verify_owner(job: PostedJob, email: str):
    if job.contact_email.lower() != email.strip().lower():
        raise HTTPException(
            status_code=403,
            detail="Email does not match the one used when posting this job."
        )


# ── POST /api/v1/post-jobs/ ───────────────────────────────────────────────────
@router.post("/", status_code=201)
async def create_posted_job(
    payload: PostJobRequest,
    db: AsyncSession = Depends(get_db),
):
    """Anyone can post a job. No auth required — ownership tied to contact email."""
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
    logger.info(f"Job posted: {job.title} @ {job.company} ({job_id})")
    return {"success": True, "job_id": job_id, "message": "Job posted successfully!"}


# ── GET /api/v1/post-jobs/ ────────────────────────────────────────────────────
@router.get("/", response_model=list[PostedJobOut])
async def list_posted_jobs(
    q:     str = Query("", description="Keyword filter"),
    email: str = Query("", description="Filter by poster email"),
    skip:  int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List active posted jobs — optionally filter by keyword or poster email."""
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

    if email.strip():
        stmt = stmt.where(PostedJob.contact_email == email.strip().lower())

    stmt = stmt.order_by(PostedJob.posted_at.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


# ── GET /api/v1/post-jobs/{job_id} ───────────────────────────────────────────
@router.get("/{job_id}", response_model=PostedJobOut)
async def get_posted_job(job_id: str, db: AsyncSession = Depends(get_db)):
    return await _get_active(job_id, db)


# ── PUT /api/v1/post-jobs/{job_id} ───────────────────────────────────────────
@router.put("/{job_id}", response_model=PostedJobOut)
async def edit_posted_job(
    job_id: str,
    payload: EditJobRequest,
    db: AsyncSession = Depends(get_db),
):
    """Edit a posted job. Caller must supply the original contact email."""
    job = await _get_active(job_id, db)
    _verify_owner(job, payload.owner_email)

    # Apply only the fields that were sent
    if payload.title        is not None: job.title           = payload.title.strip()
    if payload.company      is not None: job.company         = payload.company.strip()
    if payload.location     is not None: job.location        = payload.location.strip()
    if payload.job_type     is not None: job.job_type        = payload.job_type
    if payload.work_mode    is not None: job.work_mode       = payload.work_mode
    if payload.experience   is not None: job.experience      = payload.experience
    if payload.min_salary   is not None: job.min_salary      = payload.min_salary
    if payload.max_salary   is not None: job.max_salary      = payload.max_salary
    if payload.salary_currency is not None: job.salary_currency = payload.salary_currency
    if payload.description  is not None: job.description     = payload.description.strip()
    if payload.skills       is not None: job.skills          = payload.skills
    if payload.contact_email is not None: job.contact_email  = payload.contact_email.strip().lower()
    if payload.apply_url    is not None: job.apply_url       = payload.apply_url or None
    if payload.company_url  is not None: job.company_url     = payload.company_url or None

    await db.commit()
    await db.refresh(job)
    logger.info(f"Job updated: {job_id}")
    return job


# ── DELETE /api/v1/post-jobs/{job_id} ────────────────────────────────────────
@router.delete("/{job_id}")
async def delete_posted_job(
    job_id: str,
    payload: DeleteJobRequest,
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete a posted job. Caller must supply the original contact email."""
    job = await _get_active(job_id, db)
    _verify_owner(job, payload.contact_email)

    job.is_active = False
    await db.commit()
    logger.info(f"Job deleted (soft): {job_id}")
    return {"success": True, "message": "Job removed successfully."}
