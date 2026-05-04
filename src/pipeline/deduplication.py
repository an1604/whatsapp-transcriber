from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import DuplicateVideoError
from src.core.models import Platform, VideoCreate, compute_url_hash
from src.scrapers.url_extractor import ExtractedURL

logger = logging.getLogger(__name__)


class DeduplicationService:
    """Checks whether a video URL has already been processed.

    Uses the canonical URL hash as the deduplication key. Two different URL
    forms for the same video (e.g. short link vs. full URL) will have the
    same canonical form and hash after canonicalization, and will dedup correctly.

    Raises DuplicateVideoError (propagated from the database layer) if the
    video is already known — callers must handle this explicitly.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def is_duplicate(self, session: AsyncSession, url_hash: str) -> bool:
        """Return True if a video with this canonical URL hash already exists."""
        try:
            await self._db.get_video_by_hash(session, url_hash)
            return True
        except Exception:
            return False

    async def register_video(
        self,
        session: AsyncSession,
        extracted: ExtractedURL,
        source_group: str | None = None,
        source_message_id: str | None = None,
    ):
        """Register a new video in the database.

        Returns the newly created VideoORM.
        Raises DuplicateVideoError if already registered (no silent ignore).
        """
        data = VideoCreate(
            canonical_url=extracted.canonical_url,
            url_hash=extracted.url_hash,
            original_url=extracted.raw_url,
            platform=extracted.platform,
            source_group=source_group,
            source_message_id=source_message_id,
        )
        logger.info(
            "Registering new video: platform=%s canonical=%s",
            extracted.platform.value,
            extracted.canonical_url,
        )
        return await self._db.create_video(session, data)

    async def get_or_register_video(
        self,
        session: AsyncSession,
        extracted: ExtractedURL,
        source_group: str | None = None,
        source_message_id: str | None = None,
    ):
        """Return existing video ORM if already known, otherwise register it.

        This is the primary entry point for the pipeline — it handles the
        common case where a video is shared multiple times in a community.
        Returns (video_orm, is_new_registration).

        Raises if registration fails for any reason other than duplicate.
        """
        try:
            existing = await self._db.get_video_by_hash(session, extracted.url_hash)
            logger.info(
                "Duplicate video detected: id=%d hash=%s",
                existing.id,
                extracted.url_hash,
            )
            return existing, False
        except Exception:
            pass  # not found — register it

        video = await self.register_video(
            session, extracted, source_group=source_group,
            source_message_id=source_message_id
        )
        return video, True
