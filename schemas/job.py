from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


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


class PostJobRequest(BaseModel):
    title:           str
    company:         str
    location:        str
    job_type:        Optional[str] = None
    work_mode:       Optional[str] = None
    experience:      Optional[str] = None
    min_salary:      Optional[float] = None
    max_salary:      Optional[float] = None
    salary_currency: str = "INR"
    description:     str
    skills:          Optional[str] = None
    contact_email:   str
    apply_url:       Optional[str] = None
    company_url:     Optional[str] = None


class PostedJobOut(BaseModel):
    job_id:          str
    title:           str
    company:         str
    location:        str
    job_type:        Optional[str] = None
    work_mode:       Optional[str] = None
    experience:      Optional[str] = None
    min_salary:      Optional[float] = None
    max_salary:      Optional[float] = None
    salary_currency: str = "INR"
    description:     str
    skills:          Optional[str] = None
    contact_email:   str
    apply_url:       Optional[str] = None
    company_url:     Optional[str] = None
    is_active:       bool = True
    posted_at:       datetime

    class Config:
        from_attributes = True


class EditJobRequest(BaseModel):
    """All fields are optional except owner_email (used for verification)."""
    owner_email:     str                   # must match stored contact_email
    title:           Optional[str] = None
    company:         Optional[str] = None
    location:        Optional[str] = None
    job_type:        Optional[str] = None
    work_mode:       Optional[str] = None
    experience:      Optional[str] = None
    min_salary:      Optional[float] = None
    max_salary:      Optional[float] = None
    salary_currency: Optional[str] = None
    description:     Optional[str] = None
    skills:          Optional[str] = None
    contact_email:   Optional[str] = None   # new email if the poster wants to change it
    apply_url:       Optional[str] = None
    company_url:     Optional[str] = None


class DeleteJobRequest(BaseModel):
    contact_email: str   # must match stored contact_email


class SearchResponse(BaseModel):
    query: str
    location: str
    total: int
    page: int
    per_page: int
    jobs: list[JobResult]
    platforms_searched: list[str]
    cached: bool = False
