from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import app as api_app
import threading
import time
import subprocess
import sys
import logging

app = FastAPI(title="News Recommender Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.mount(path="/", app=api_app)


def _start_ingest_scheduler(interval_minutes: int = 30):
    def _loop():
        while True:
            try:
                logging.info("[scheduler] Running ingestion...")
                subprocess.run([sys.executable, "-m", "src.ingest_rss"], check=False)
                logging.info("[scheduler] Ingestion finished")
            except Exception as e:
                logging.exception(f"[scheduler] Ingestion error: {e}")
            time.sleep(max(60, interval_minutes * 60))

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


@app.on_event("startup")
async def _on_startup():
    # Disable background ingestion for testing (uncomment to enable)
    # _start_ingest_scheduler(interval_minutes=30)
    pass
