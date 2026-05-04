from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.database import Database
from src.web.dependencies import set_database
from src.web.routes import dashboard, htmx, jobs, pages, submit, videos


def create_app(
    database_url: str = "sqlite+aiosqlite:///data/app.db",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Raises if the database_url is blank (configuration error — never silently
    falls back to a default that might mask misconfiguration).
    """
    if not database_url.strip():
        raise ValueError("database_url must not be empty")

    db = Database(database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_database(db)
        await db.create_tables()
        yield
        await db.close()

    app = FastAPI(
        title="WhatsApp Transcriber",
        version="0.1.0",
        description="Video transcription and summarization pipeline",
        lifespan=lifespan,
    )

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # REST API
    app.include_router(dashboard.router)
    app.include_router(videos.router)
    app.include_router(jobs.router)
    app.include_router(submit.router)
    # HTML pages + HTMX partials
    app.include_router(pages.router)
    app.include_router(htmx.router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
