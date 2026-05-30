from datetime import datetime
from sqlalchemy import Integer, String, Float, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base


class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    location: Mapped[str] = mapped_column(String(256), default="India")
    platforms: Mapped[str] = mapped_column(String(256), default="all")
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    searched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SavedJob(Base):
    __tablename__ = "saved_jobs"
    __table_args__ = (UniqueConstraint("session_id", "job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512))
    company: Mapped[str] = mapped_column(String(256))
    location: Mapped[str] = mapped_column(String(256), nullable=True)
    min_salary: Mapped[float] = mapped_column(Float, nullable=True)
    max_salary: Mapped[float] = mapped_column(Float, nullable=True)
    salary_currency: Mapped[str] = mapped_column(String(8), default="INR")
    job_url: Mapped[str] = mapped_column(Text, nullable=True)
    platform: Mapped[str] = mapped_column(String(64), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    date_posted: Mapped[str] = mapped_column(String(32), nullable=True)
    saved_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrendingCache(Base):
    __tablename__ = "trending_cache"

    cache_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
