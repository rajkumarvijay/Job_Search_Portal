from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Optional
from db.database import get_db
from db.models import SavedJob
from schemas.trending import SavedJobCreate, SavedJobRead

router = APIRouter(prefix="/saved", tags=["saved"])

DEFAULT_SESSION = "anonymous"


def _get_session(x_session_id: Optional[str] = Header(None)) -> str:
    return x_session_id or DEFAULT_SESSION


@router.get("", response_model=list[SavedJobRead])
async def get_saved(
    session_id: str = Depends(_get_session),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SavedJob)
        .where(SavedJob.session_id == session_id)
        .order_by(SavedJob.saved_at.desc())
    )
    rows = result.scalars().all()
    return [
        SavedJobRead(
            **{c.name: getattr(r, c.name) for c in SavedJob.__table__.columns if c.name not in ("id", "session_id", "saved_at")},
            id=r.id,
            session_id=r.session_id,
            saved_at=r.saved_at.isoformat() if r.saved_at else "",
        )
        for r in rows
    ]


@router.post("", response_model=SavedJobRead, status_code=201)
async def save_job(
    body: SavedJobCreate,
    session_id: str = Depends(_get_session),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(
        select(SavedJob).where(
            SavedJob.session_id == session_id,
            SavedJob.job_id == body.job_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Job already saved")

    job = SavedJob(
        session_id=session_id,
        job_id=body.job_id,
        title=body.title,
        company=body.company,
        location=body.location,
        min_salary=body.min_salary,
        max_salary=body.max_salary,
        salary_currency=body.salary_currency,
        job_url=body.job_url,
        platform=body.platform,
        description=body.description,
        date_posted=body.date_posted,
        saved_at=datetime.utcnow(),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return SavedJobRead(
        **{k: v for k, v in body.model_dump().items()},
        id=job.id,
        session_id=session_id,
        saved_at=job.saved_at.isoformat(),
    )


@router.delete("/{job_id}", status_code=204)
async def unsave_job(
    job_id: str,
    session_id: str = Depends(_get_session),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SavedJob).where(
            SavedJob.session_id == session_id,
            SavedJob.job_id == job_id,
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Saved job not found")
    await db.delete(job)
    await db.commit()
