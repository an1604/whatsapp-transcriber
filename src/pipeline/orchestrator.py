from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import (
    DownloadError,
    InvalidStatusTransitionError,
    SummarizationError,
    TranscriptionError,
)
from src.core.models import JobCreate, JobStatus, JobType, VideoStatus, VideoUpdate
from src.downloaders.base import AbstractDownloader
from src.pipeline.deduplication import DeduplicationService
from src.pipeline.job_queue import JobQueue, PipelineJob
from src.scrapers.url_extractor import ExtractedURL
from src.summarizers.base import AbstractSummarizer
from src.transcribers.base import AbstractTranscriber

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates the full download → transcribe → summarize pipeline.

    Each stage is tracked in the database with explicit status transitions.
    Any stage failure sets the video to FAILED with an error message —
    the pipeline never silently ignores errors.

    The orchestrator raises PipelineError (a subclass of the stage-specific
    exceptions) only if an unexpected error occurs outside a known stage.
    Stage-specific errors are recorded in the DB and do NOT propagate — this
    allows the caller to process the next video after a failure.
    """

    def __init__(
        self,
        db: Database,
        downloader: AbstractDownloader,
        transcriber: AbstractTranscriber,
        summarizer: AbstractSummarizer,
        audio_cache_dir: Path,
        dedup_service: DeduplicationService | None = None,
    ) -> None:
        self._db = db
        self._downloader = downloader
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._audio_cache_dir = Path(audio_cache_dir)
        self._dedup = dedup_service

    async def process_video_by_id(self, session: AsyncSession, video_id: int):
        """Process an already-registered video that is in 'discovered' state.

        Raises VideoNotFoundError if the video does not exist.
        Raises InvalidStatusTransitionError if the video is not in 'discovered' status.
        """
        video = await self._db.get_video_by_id(session, video_id)
        if video.status != VideoStatus.DISCOVERED.value:
            raise InvalidStatusTransitionError(
                f"Video id={video_id} has status={video.status!r}; "
                f"only 'discovered' videos can be triggered for processing"
            )
        video = await self._run_download(session, video_id, video.canonical_url)
        if video.status == VideoStatus.FAILED.value:
            return video
        video = await self._run_transcription(session, video_id)
        if video.status == VideoStatus.FAILED.value:
            return video
        return await self._run_summarization(session, video_id)

    async def process_url(
        self,
        session: AsyncSession,
        extracted: ExtractedURL,
        source_group: str | None = None,
        source_message_id: str | None = None,
    ):
        """Full pipeline for a single URL: register → download → transcribe → summarize.

        Returns the final VideoORM in COMPLETE or FAILED state.
        """
        if self._dedup:
            video, is_new = await self._dedup.get_or_register_video(
                session, extracted,
                source_group=source_group,
                source_message_id=source_message_id,
            )
            if not is_new:
                logger.info("Skipping duplicate video id=%d", video.id)
                return video
        else:
            video = await self._db.create_video(
                session,
                __import__("src.core.models", fromlist=["VideoCreate"]).VideoCreate(
                    canonical_url=extracted.canonical_url,
                    url_hash=extracted.url_hash,
                    original_url=extracted.raw_url,
                    platform=extracted.platform,
                    source_group=source_group,
                    source_message_id=source_message_id,
                ),
            )

        video = await self._run_download(session, video.id, extracted.canonical_url)
        if video.status == VideoStatus.FAILED.value:
            return video

        video = await self._run_transcription(session, video.id)
        if video.status == VideoStatus.FAILED.value:
            return video

        video = await self._run_summarization(session, video.id)
        return video

    async def _run_download(self, session: AsyncSession, video_id: int, url: str):
        video = await self._db.update_video_status(
            session, video_id, VideoStatus.DOWNLOADING
        )
        job = await self._db.create_job(
            session, JobCreate(video_id=video_id, job_type=JobType.DOWNLOAD)
        )
        await self._db.update_job_status(session, job.id, JobStatus.RUNNING)

        try:
            audio_path = await self._downloader.download_audio(url, self._audio_cache_dir)
        except DownloadError as exc:
            await self._fail_video(session, video_id, job.id, str(exc))
            return await self._db.get_video_by_id(session, video_id)

        await self._db.update_video(
            session, video_id, VideoUpdate(audio_path=str(audio_path))
        )
        await self._db.update_job_status(session, job.id, JobStatus.COMPLETED)
        return await self._db.update_video_status(
            session, video_id, VideoStatus.DOWNLOADED
        )

    async def _run_transcription(self, session: AsyncSession, video_id: int):
        video = await self._db.get_video_by_id(session, video_id)
        if not video.audio_path:
            await self._fail_video(
                session, video_id, None,
                "audio_path is missing — cannot transcribe"
            )
            return await self._db.get_video_by_id(session, video_id)

        await self._db.update_video_status(session, video_id, VideoStatus.TRANSCRIBING)
        job = await self._db.create_job(
            session, JobCreate(video_id=video_id, job_type=JobType.TRANSCRIBE)
        )
        await self._db.update_job_status(session, job.id, JobStatus.RUNNING)

        try:
            result = await self._transcriber.transcribe(Path(video.audio_path))
        except TranscriptionError as exc:
            await self._fail_video(session, video_id, job.id, str(exc))
            return await self._db.get_video_by_id(session, video_id)

        await self._db.update_video(
            session,
            video_id,
            VideoUpdate(
                transcript=result.text,
                detected_language=result.detected_language,
            ),
        )
        await self._db.update_job_status(session, job.id, JobStatus.COMPLETED)
        return await self._db.update_video_status(
            session, video_id, VideoStatus.TRANSCRIBED
        )

    async def _run_summarization(self, session: AsyncSession, video_id: int):
        video = await self._db.get_video_by_id(session, video_id)
        if not video.transcript:
            await self._fail_video(
                session, video_id, None,
                "transcript is missing — cannot summarize"
            )
            return await self._db.get_video_by_id(session, video_id)

        await self._db.update_video_status(session, video_id, VideoStatus.SUMMARIZING)
        job = await self._db.create_job(
            session, JobCreate(video_id=video_id, job_type=JobType.SUMMARIZE)
        )
        await self._db.update_job_status(session, job.id, JobStatus.RUNNING)

        language = video.detected_language or "en"
        try:
            result = await self._summarizer.summarize(video.transcript, language)
        except SummarizationError as exc:
            await self._fail_video(session, video_id, job.id, str(exc))
            return await self._db.get_video_by_id(session, video_id)

        await self._db.update_video(
            session, video_id, VideoUpdate(summary=result.summary)
        )
        await self._db.update_job_status(session, job.id, JobStatus.COMPLETED)
        return await self._db.update_video_status(
            session, video_id, VideoStatus.COMPLETE
        )

    async def _fail_video(
        self,
        session: AsyncSession,
        video_id: int,
        job_id: int | None,
        error: str,
    ) -> None:
        logger.error("Pipeline failure for video id=%d: %s", video_id, error)
        if job_id is not None:
            try:
                await self._db.update_job_status(
                    session, job_id, JobStatus.FAILED, error=error
                )
            except Exception as inner:
                logger.error("Failed to update job status: %s", inner)
        try:
            await self._db.update_video(
                session, video_id, VideoUpdate(error_message=error)
            )
            await self._db.update_video_status(
                session, video_id, VideoStatus.FAILED
            )
        except Exception as inner:
            logger.error("Failed to update video status to FAILED: %s", inner)

    def make_job_processor(self, session_factory):
        """Return a coroutine suitable for JobQueue.set_processor().

        The processor opens a new session per job to avoid session sharing issues.
        """
        async def processor(job: PipelineJob) -> None:
            from src.scrapers.url_extractor import extract_single_url
            extracted = extract_single_url(job.url)
            async with session_factory() as session:
                await self.process_url(session, extracted)

        return processor
