from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SearchHistoryCreate(BaseModel):
    query: str
    location: str = "India"
    platforms: str = "all"
    result_count: int = 0


class SearchHistoryRead(BaseModel):
    id: int
    session_id: str
    query: str
    location: str
    platforms: str
    result_count: int
    searched_at: datetime

    model_config = {"from_attributes": True}
