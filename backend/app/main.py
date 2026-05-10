import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

sys.path.insert(0, '/app')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def _run_pipeline():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '/app/entrypoint.py',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd='/app',
    )
    async for line in proc.stdout:
        logger.info(line.decode().rstrip())
    await proc.wait()
    if proc.returncode != 0:
        logger.error("Pipeline exited with code %s", proc.returncode)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler(timezone="Asia/Taipei")
    scheduler.add_job(_run_pipeline, CronTrigger(hour=12, minute=0))
    scheduler.start()
    logger.info("Scheduler started — pipeline runs daily at 12:00 Asia/Taipei")
    yield
    scheduler.shutdown()


app = FastAPI(title="ATS Backend API", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ATS backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pipeline/run")
async def trigger_pipeline():
    """手動觸發 pipeline，不等待執行完成。"""
    asyncio.create_task(_run_pipeline())
    return {"status": "pipeline triggered"}
