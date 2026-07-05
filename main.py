import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.init_db import init_db
from routers import jobs, trending, history, saved, ai, payments, post_jobs, auth
from scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx request logs — they include full URLs with API keys as query params
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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

# ── CORS ──────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS in Railway backend Variables:
#   ALLOWED_ORIGINS=https://yourapp.vercel.app,https://jobquest.in
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

app.include_router(jobs.router,     prefix="/api/v1")
app.include_router(trending.router, prefix="/api/v1")
app.include_router(history.router,  prefix="/api/v1")
app.include_router(saved.router,    prefix="/api/v1")
app.include_router(ai.router,       prefix="/api/v1")
app.include_router(payments.router,   prefix="/api/v1")
app.include_router(post_jobs.router,  prefix="/api/v1")
app.include_router(auth.router,       prefix="/api/v1")


# ── Root & health endpoints ───────────────────────────────────────────────────
# GET / — Railway load balancer and uptime monitors hit this.
# Must return 200 or Railway marks the service as unhealthy.
@app.get("/")
async def root():
    return {
        "service": "Job Search Portal API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Job Search Portal API"}

