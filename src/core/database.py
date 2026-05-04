from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, select
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
