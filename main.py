"""Entry point for the Novel Generator server."""

import uvicorn
from backend.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
