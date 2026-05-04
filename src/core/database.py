from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, delete as sql_delete, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.core.exceptions import (
    DuplicateVideoError,
    InvalidStatusTransitionError,
    JobNotFoundError,
    VideoNotFoundError,
)
from src.core.models import (
    JobCreate,
    JobStatus,
    JobType,
    Platform,
    VideoCreate,
    VideoStatus,
    VideoUpdate,
    VALID_STATUS_TRANSITIONS,
)


def _utcnow() -> datetime:
    """Return the current UTC time as a naive datetime (for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class VideoORM(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    original_url: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detected_language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=VideoStatus.DISCOVERED.value, index=True
    )
    audio_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audio_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_group: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_message_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    jobs: Mapped[list[JobORM]] = relationship(
        "JobORM", back_populates="video", cascade="all, delete-orphan"
    )


class JobORM(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=JobStatus.PENDING.value
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    video: Mapped[VideoORM] = relationship("VideoORM", back_populates="jobs")


class Database:
    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        await self._engine.dispose()

    # ------------------------------------------------------------------
    # Video operations
    # ------------------------------------------------------------------

    async def create_video(self, session: AsyncSession, data: VideoCreate) -> VideoORM:
        video = VideoORM(
            canonical_url=data.canonical_url,
            url_hash=data.url_hash,
            original_url=data.original_url,
            platform=data.platform.value,
            source_group=data.source_group,
            source_message_id=data.source_message_id,
            status=VideoStatus.DISCOVERED.value,
            audio_deleted=False,
        )
        session.add(video)
        try:
            await session.flush()
        except IntegrityError as exc:
            await session.rollback()
            raise DuplicateVideoError(
                f"A video with url_hash={data.url_hash!r} already exists"
            ) from exc
        await session.commit()
        await session.refresh(video)
        return video

    async def get_video_by_id(self, session: AsyncSession, video_id: int) -> VideoORM:
        result = await session.get(VideoORM, video_id)
        if result is None:
            raise VideoNotFoundError(f"Video with id={video_id} not found")
        return result

    async def get_video_by_hash(self, session: AsyncSession, url_hash: str) -> VideoORM:
        stmt = select(VideoORM).where(VideoORM.url_hash == url_hash)
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise VideoNotFoundError(f"Video with url_hash={url_hash!r} not found")
        return row

    async def update_video_status(
        self,
        session: AsyncSession,
        video_id: int,
        new_status: VideoStatus,
    ) -> VideoORM:
        video = await self.get_video_by_id(session, video_id)
        current = VideoStatus(video.status)
        allowed = VALID_STATUS_TRANSITIONS[current]
        if new_status not in allowed:
            raise InvalidStatusTransitionError(
                f"Cannot transition video id={video_id} from {current.value!r} "
                f"to {new_status.value!r}. "
                f"Allowed next states: {sorted(s.value for s in allowed) or '(none — terminal state)'}"
            )
        video.status = new_status.value
        video.updated_at = _utcnow()
        if new_status == VideoStatus.COMPLETE:
            video.completed_at = _utcnow()
        await session.commit()
        await session.refresh(video)
        return video

    async def update_video(
        self,
        session: AsyncSession,
        video_id: int,
        data: VideoUpdate,
    ) -> VideoORM:
        video = await self.get_video_by_id(session, video_id)
        for field, value in data.model_dump(exclude_none=True).items():
            setattr(video, field, value)
        video.updated_at = _utcnow()
        await session.commit()
        await session.refresh(video)
        return video

    async def mark_audio_deleted(self, session: AsyncSession, video_id: int) -> VideoORM:
        video = await self.get_video_by_id(session, video_id)
        video.audio_deleted = True
        video.audio_path = None
        video.updated_at = _utcnow()
        await session.commit()
        await session.refresh(video)
        return video

    async def list_videos(
        self,
        session: AsyncSession,
        platform: Optional[Platform] = None,
        status: Optional[VideoStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VideoORM]:
        stmt = select(VideoORM)
        if platform is not None:
            stmt = stmt.where(VideoORM.platform == platform.value)
        if status is not None:
            stmt = stmt.where(VideoORM.status == status.value)
        stmt = stmt.order_by(VideoORM.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def search_videos(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 25,
        offset: int = 0,
    ) -> list[VideoORM]:
        """Full-text LIKE search across canonical_url, transcript, and summary."""
        if not query or not query.strip():
            raise ValueError("search query must not be empty")
        if limit < 1 or limit > 200:
            raise ValueError(f"limit must be between 1 and 200, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        pattern = f"%{query.strip()}%"
        stmt = (
            select(VideoORM)
            .where(
                or_(
                    VideoORM.canonical_url.like(pattern),
                    VideoORM.transcript.like(pattern),
                    VideoORM.summary.like(pattern),
                )
            )
            .order_by(VideoORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def bulk_mark_audio_deleted(
        self,
        session: AsyncSession,
        video_ids: list[int],
    ) -> int:
        """Mark audio as deleted for multiple videos at once.

        Returns the number of rows updated. Raises ValueError for empty list.
        Silently skips IDs that don't exist.
        """
        if not video_ids:
            raise ValueError("video_ids must not be empty")
        stmt = (
            update(VideoORM)
            .where(VideoORM.id.in_(video_ids))
            .values(audio_deleted=True, audio_path=None, updated_at=_utcnow())
            .execution_options(synchronize_session="fetch")
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount

    async def delete_video(self, session: AsyncSession, video_id: int) -> None:
        """Hard-delete a video and its jobs.

        Raises VideoNotFoundError if the video does not exist.
        Jobs are deleted explicitly first to avoid relying on SQLite FK cascade.
        """
        await self.get_video_by_id(session, video_id)  # raises if missing
        await session.execute(sql_delete(JobORM).where(JobORM.video_id == video_id))
        await session.execute(sql_delete(VideoORM).where(VideoORM.id == video_id))
        await session.commit()

    async def bulk_delete_videos(
        self,
        session: AsyncSession,
        video_ids: list[int],
    ) -> int:
        """Hard-delete multiple videos and their jobs.

        Returns count of deleted video rows. Raises ValueError for empty list.
        Silently skips IDs that do not exist.
        """
        if not video_ids:
            raise ValueError("video_ids must not be empty")
        await session.execute(sql_delete(JobORM).where(JobORM.video_id.in_(video_ids)))
        result = await session.execute(
            sql_delete(VideoORM).where(VideoORM.id.in_(video_ids))
        )
        await session.commit()
        return result.rowcount

    async def list_inbox_videos(
        self,
        session: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> list[VideoORM]:
        """Return videos in 'discovered' status, newest first (the inbox)."""
        if limit < 1 or limit > 500:
            raise ValueError(f"limit must be between 1 and 500, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        stmt = (
            select(VideoORM)
            .where(VideoORM.status == VideoStatus.DISCOVERED.value)
            .order_by(VideoORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def count_inbox_videos(self, session: AsyncSession) -> int:
        """Count videos currently in 'discovered' state."""
        from sqlalchemy import func
        stmt = (
            select(func.count())
            .select_from(VideoORM)
            .where(VideoORM.status == VideoStatus.DISCOVERED.value)
        )
        return (await session.execute(stmt)).scalar_one()

    async def get_disk_stats(self, session: AsyncSession) -> dict[str, int]:
        """Return aggregate counts for disk/audio state.

        Returns a dict with:
        - videos_with_audio: videos that have an audio file on disk
        - videos_audio_deleted: videos whose audio was deleted
        - videos_without_audio: videos that never had audio (no audio_path)
        """
        from sqlalchemy import func, case

        stmt = select(
            func.count().label("total"),
            func.sum(
                case((VideoORM.audio_path.is_not(None) & ~VideoORM.audio_deleted, 1), else_=0)
            ).label("with_audio"),
            func.sum(
                case((VideoORM.audio_deleted == True, 1), else_=0)  # noqa: E712
            ).label("audio_deleted"),
        )
        row = (await session.execute(stmt)).one()
        total = row.total or 0
        with_audio = int(row.with_audio or 0)
        audio_deleted = int(row.audio_deleted or 0)
        return {
            "videos_with_audio": with_audio,
            "videos_audio_deleted": audio_deleted,
            "videos_without_audio": total - with_audio - audio_deleted,
        }

    # ------------------------------------------------------------------
    # Job operations
    # ------------------------------------------------------------------

    async def create_job(self, session: AsyncSession, data: JobCreate) -> JobORM:
        await self.get_video_by_id(session, data.video_id)  # raises if missing
        job = JobORM(
            video_id=data.video_id,
            job_type=data.job_type.value,
            status=JobStatus.PENDING.value,
        )
        session.add(job)
        await session.flush()
        await session.commit()
        await session.refresh(job)
        return job

    async def get_job_by_id(self, session: AsyncSession, job_id: int) -> JobORM:
        result = await session.get(JobORM, job_id)
        if result is None:
            raise JobNotFoundError(f"Job with id={job_id} not found")
        return result

    async def list_jobs_for_video(
        self, session: AsyncSession, video_id: int
    ) -> list[JobORM]:
        await self.get_video_by_id(session, video_id)  # raises if missing
        stmt = (
            select(JobORM)
            .where(JobORM.video_id == video_id)
            .order_by(JobORM.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def update_job_status(
        self,
        session: AsyncSession,
        job_id: int,
        new_status: JobStatus,
        error: Optional[str] = None,
    ) -> JobORM:
        job = await self.get_job_by_id(session, job_id)
        job.status = new_status.value
        if new_status == JobStatus.RUNNING:
            job.started_at = _utcnow()
        elif new_status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = _utcnow()
        if error is not None:
            job.error = error
        await session.commit()
        await session.refresh(job)
        return job
