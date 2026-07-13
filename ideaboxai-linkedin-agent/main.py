import argparse
import logging

from fastapi import FastAPI

from config import settings
from src import models  # noqa: F401  (imported for its side effect: creates DB tables)

# Setup logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IdeaBoxAI Engage",
    version="1.0.0",
    description="Autonomous LinkedIn engagement agent",
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.on_event("startup")
async def startup():
    logger.info("IdeaBoxAI Engage starting up...")
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"Debug mode: {settings.debug}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("IdeaBoxAI Engage shutting down...")


def _parse_args():
    parser = argparse.ArgumentParser(description="IdeaBoxAI Engage")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.version:
        print("IdeaBoxAI Engage 1.0.0")
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=settings.port)
