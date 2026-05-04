from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.database import Database
from src.web.dependencies import set_database, set_orchestrator
from src.web.routes import dashboard, htmx, jobs, pages, submit, videos
from src.web.routes import inbox


def create_app(
    database_url: str = "sqlite+aiosqlite:///data/app.db",
    cors_origins: list[str] | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Pass ``config_path`` to enable the pipeline orchestrator (required for
    the inbox Process button). Without it, processing endpoints return 503.

    Raises if the database_url is blank — never silently falls back.
    """
    if not database_url.strip():
        raise ValueError("database_url must not be empty")

    db = Database(database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_database(db)
        await db.create_tables()

        if config_path is not None:
            _setup_orchestrator(db, config_path)

        yield

        set_orchestrator(None)
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
    app.include_router(inbox.router)
    # HTML pages + HTMX partials
    app.include_router(pages.router)
    app.include_router(htmx.router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def _setup_orchestrator(db: Database, config_path: Path) -> None:
    """Build the pipeline orchestrator from config and register it globally.

    Raises ConfigurationError (from config loading) or any import/init error —
    never silently swallows failures.
    """
    from src.core.config import load_config_with_env_overrides
    from src.downloaders.ytdlp_downloader import YtDlpDownloader
    from src.pipeline.deduplication import DeduplicationService
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.summarizers.litellm_summarizer import LiteLLMSummarizer
    from src.transcribers.faster_whisper import FasterWhisperTranscriber

    cfg = load_config_with_env_overrides(config_path)

    downloader = YtDlpDownloader()
    transcriber = FasterWhisperTranscriber(
        model_size=cfg.transcriber.params.model_size,
        device=cfg.transcriber.params.device,
        compute_type=cfg.transcriber.params.compute_type,
    )
    summarizer = LiteLLMSummarizer(
        model=cfg.summarizer.params.model,
        api_base=cfg.summarizer.params.api_base,
        temperature=cfg.summarizer.params.temperature,
        prompt_path=Path(cfg.summarizer.prompt_path),
    )
    dedup = DeduplicationService(db) if cfg.deduplication.enabled else None
    orchestrator = PipelineOrchestrator(
        db=db,
        downloader=downloader,
        transcriber=transcriber,
        summarizer=summarizer,
        audio_cache_dir=Path(cfg.storage.audio_cache_dir),
        dedup_service=dedup,
    )
    set_orchestrator(orchestrator)
