from fastapi import APIRouter
from services.trending_service import (
    get_trending_roles,
    get_salary_bands,
    get_trending_keywords,
    get_stats,
)
from schemas.trending import TrendingRole, SalaryBand, TrendingKeyword, StatsResponse

router = APIRouter(prefix="/trending", tags=["trending"])


@router.get("/roles", response_model=list[TrendingRole])
async def trending_roles():
    return await get_trending_roles()


@router.get("/salary-bands", response_model=list[SalaryBand])
async def salary_bands():
    return await get_salary_bands()


@router.get("/keywords", response_model=list[TrendingKeyword])
async def trending_keywords():
    return await get_trending_keywords()


@router.get("/stats", response_model=StatsResponse)
async def portal_stats():
    return await get_stats()
