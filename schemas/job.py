from pydantic import BaseModel
from typing import Optional


class JobResult(BaseModel):
    job_id: str
    title: str
    company: str
    location: Optional[str] = None
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    salary_currency: str = "INR"
    salary_interval: Optional[str] = None
    job_url: Optional[str] = None
    platform: str
    description: Optional[str] = None
    date_posted: Optional[str] = None
    job_type: Optional[str] = None
    is_remote: Optional[bool] = None
    company_logo: Optional[str] = None
    company_url: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    location: str
    total: int
    page: int
    per_page: int
    jobs: list[JobResult]
    platforms_searched: list[str]
    cached: bool = False
