import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.init_db import init_db
from routers import jobs, trending, history, saved
from scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    start_scheduler()
    logger.info("Job Search Portal API ready")
    yield
    stop_scheduler()


app = FastAPI(
    title="Job Search Portal API",
    description="Aggregate jobs from LinkedIn, Indeed, Glassdoor, Naukri, ZipRecruiter, Google Jobs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/api/v1")
app.include_router(trending.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(saved.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Job Search Portal API"}
