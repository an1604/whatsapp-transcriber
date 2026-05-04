from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import PipelineNotConfiguredError

if TYPE_CHECKING:
    from src.pipeline.orchestrator import PipelineOrchestrator

# Module-level singletons set at startup by create_app()
_db: Database | None = None
_orchestrator: "PipelineOrchestrator | None" = None


def set_database(db: Database) -> None:
    global _db
    _db = db


def get_database() -> Database:
    if _db is None:
        raise RuntimeError(
            "Database not initialized. Call set_database() before handling requests."
        )
    return _db


def set_orchestrator(orchestrator: "PipelineOrchestrator | None") -> None:
    global _orchestrator
    _orchestrator = orchestrator


def get_orchestrator() -> "PipelineOrchestrator":
    if _orchestrator is None:
        raise PipelineNotConfiguredError(
            "Pipeline orchestrator is not configured. "
            "Start the app with config_path set to enable video processing."
        )
    return _orchestrator


async def get_session() -> AsyncIterator[AsyncSession]:
    db = get_database()
    async with db._session_factory() as session:
        yield session
