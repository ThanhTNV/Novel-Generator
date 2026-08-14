"""Entry point for the Novel Generator server."""

import os

import uvicorn

from backend.config import settings

if __name__ == "__main__":
    # Auto-reload is opt-in (RELOAD=true). It runs the app in a child process,
    # and on Windows that child regularly outlives Ctrl+C / a killed parent —
    # leaving an orphan still bound to the port, so the next start silently
    # serves stale config (an old API key, an old model) from the zombie.
    reload = os.getenv("RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run(
        "backend.server:app",
        host=settings.host,
        port=settings.port,
        reload=reload,
    )
