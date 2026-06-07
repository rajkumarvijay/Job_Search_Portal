import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from db.init_db import init_db
from routers import jobs, trending, history, saved, ai, payments, post_jobs
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

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# ── OPTIONS preflight fallback ────────────────────────────────────────────────
# Only matches paths that start with /api/ so the root GET / is not shadowed.
@app.options("/api/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        },
    )
