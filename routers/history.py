from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import Optional
from db.database import get_db
from db.models import SearchHistory
from schemas.history import SearchHistoryCreate, SearchHistoryRead

router = APIRouter(prefix="/history", tags=["history"])

DEFAULT_SESSION = "anonymous"


def _get_session(x_session_id: Optional[str] = Header(None)) -> str:
    return x_session_id or DEFAULT_SESSION


@router.get("", response_model=list[SearchHistoryRead])
async def get_history(
    session_id: str = Depends(_get_session),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SearchHistory)
        .where(SearchHistory.session_id == session_id)
        .order_by(SearchHistory.searched_at.desc())
        .limit(20)
    )
    return result.scalars().all()


@router.post("", response_model=SearchHistoryRead, status_code=201)
async def save_history(
    body: SearchHistoryCreate,
    session_id: str = Depends(_get_session),
    db: AsyncSession = Depends(get_db),
):
    entry = SearchHistory(
        session_id=session_id,
        query=body.query,
        location=body.location,
        platforms=body.platforms,
        result_count=body.result_count,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


@router.delete("/{entry_id}", status_code=204)
async def delete_history_entry(
    entry_id: int,
    session_id: str = Depends(_get_session),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SearchHistory).where(
            SearchHistory.id == entry_id,
            SearchHistory.session_id == session_id,
        )
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    await db.delete(entry)
    await db.commit()


@router.delete("", status_code=204)
async def clear_history(
    session_id: str = Depends(_get_session),
    db: AsyncSession = Depends(get_db),
):
    await db.execute(
        delete(SearchHistory).where(SearchHistory.session_id == session_id)
    )
    await db.commit()
