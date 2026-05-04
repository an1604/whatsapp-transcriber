from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database

# Module-level singletons set at startup by create_app()
_db: Database | None = None


def set_database(db: Database) -> None:
    global _db
    _db = db


def get_database() -> Database:
    if _db is None:
        raise RuntimeError(
            "Database not initialized. Call set_database() before handling requests."
        )
    return _db


async def get_session() -> AsyncIterator[AsyncSession]:
    db = get_database()
    async with db._session_factory() as session:
        yield session
