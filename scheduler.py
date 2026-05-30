import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None


def start_scheduler():
    global _scheduler
    from services.trending_service import refresh_trending_cache

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        refresh_trending_cache,
        trigger=IntervalTrigger(hours=6),
        id="refresh_trending",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("APScheduler started — trending cache refreshes every 6 hours")


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped")
