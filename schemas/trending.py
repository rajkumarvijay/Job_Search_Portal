from pydantic import BaseModel
from typing import Optional


class TrendingRole(BaseModel):
    role: str
    count: int
    top_skill: str
    avg_salary_lpa: Optional[str] = None
    icon: str = "briefcase"


class SalaryBand(BaseModel):
    role: str
    fresher: str
    mid: str
    senior: str
    currency: str = "INR LPA"


class TrendingKeyword(BaseModel):
    keyword: str
    count: int
    trend: str = "up"


class StatsResponse(BaseModel):
    total_active_jobs: int
    top_salary_lpa: str
    platform_count: int
    cities_covered: int


class SavedJobCreate(BaseModel):
    job_id: str
    title: str
    company: str
    location: Optional[str] = None
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    salary_currency: str = "INR"
    job_url: Optional[str] = None
    platform: Optional[str] = None
    description: Optional[str] = None
    date_posted: Optional[str] = None


class SavedJobRead(SavedJobCreate):
    id: int
    session_id: str
    saved_at: str

    model_config = {"from_attributes": True}
