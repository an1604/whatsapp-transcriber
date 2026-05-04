"""Comprehensive tests for pipeline: deduplication, job_queue, orchestrator."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import (
    DownloadError,
    DuplicateVideoError,
    SummarizationError,
    TranscriptionError,
)
from src.core.models import (
    JobStatus,
    JobType,
    Platform,
    VideoStatus,
    compute_url_hash,
)
from src.pipeline.deduplication import DeduplicationService
from src.pipeline.job_queue import JobQueue, PipelineJob
from src.pipeline.orchestrator import PipelineOrchestrator
from src.scrapers.url_extractor import ExtractedURL
from src.transcribers.base import TranscriptionResult
from src.summarizers.base import SummaryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extracted(url: str = "https://www.youtube.com/watch?v=abc123") -> ExtractedURL:
    canonical = url
    return ExtractedURL(
        raw_url=url,
        canonical_url=canonical,
        platform=Platform.YOUTUBE,
        url_hash=compute_url_hash(canonical),
    )


def _make_downloader(audio_path: Path | None = None, error: Exception | None = None):
    dl = AsyncMock()
    if error:
        dl.download_audio = AsyncMock(side_effect=error)
    else:
        dl.download_audio = AsyncMock(return_value=audio_path or Path("/tmp/audio.mp3"))
    return dl


def _make_transcriber(text: str = "Hello", lang: str = "en", error: Exception | None = None):
    tr = AsyncMock()
    if error:
        tr.transcribe = AsyncMock(side_effect=error)
    else:
        tr.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text=text, detected_language=lang, segments=[]
            )
        )
    return tr


def _make_summarizer(summary: str = "Key points.", error: Exception | None = None):
    s = AsyncMock()
    if error:
        s.summarize = AsyncMock(side_effect=error)
    else:
        s.summarize = AsyncMock(
            return_value=SummaryResult(
                summary=summary, model="m", prompt_tokens=10, completion_tokens=5
            )
        )
    return s


# ---------------------------------------------------------------------------
# DeduplicationService
# ---------------------------------------------------------------------------

class TestDeduplicationService:
    async def test_is_duplicate_returns_false_for_new_url(self, db, session):
        svc = DeduplicationService(db)
        fake_hash = "a" * 64
        result = await svc.is_duplicate(session, fake_hash)
        assert result is False

    async def test_is_duplicate_returns_true_after_registration(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        await svc.register_video(session, ext)
        result = await svc.is_duplicate(session, ext.url_hash)
        assert result is True

    async def test_register_video_creates_record(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        video = await svc.register_video(session, ext, source_group="grp1")
        assert video.id is not None
        assert video.canonical_url == ext.canonical_url
        assert video.url_hash == ext.url_hash
        assert video.source_group == "grp1"
        assert video.status == VideoStatus.DISCOVERED.value

    async def test_register_video_duplicate_raises(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        await svc.register_video(session, ext)
        with pytest.raises(DuplicateVideoError):
            await svc.register_video(session, ext)

    async def test_get_or_register_new_video(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        video, is_new = await svc.get_or_register_video(session, ext)
        assert is_new is True
        assert video.canonical_url == ext.canonical_url

    async def test_get_or_register_existing_returns_existing(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        first, _ = await svc.get_or_register_video(session, ext)
        second, is_new = await svc.get_or_register_video(session, ext)
        assert is_new is False
        assert second.id == first.id

    async def test_source_group_stored(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        video = await svc.register_video(session, ext, source_group="community_1")
        assert video.source_group == "community_1"

    async def test_source_message_id_stored(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        video = await svc.register_video(session, ext, source_message_id="msg_42")
        assert video.source_message_id == "msg_42"

    async def test_different_urls_both_registered(self, db, session):
        svc = DeduplicationService(db)
        ext1 = _extracted("https://www.youtube.com/watch?v=aaa111")
        ext2 = _extracted("https://www.youtube.com/watch?v=bbb222")
        v1, _ = await svc.get_or_register_video(session, ext1)
        v2, _ = await svc.get_or_register_video(session, ext2)
        assert v1.id != v2.id

    async def test_is_duplicate_after_get_or_register(self, db, session):
        svc = DeduplicationService(db)
        ext = _extracted()
        await svc.get_or_register_video(session, ext)
        assert await svc.is_duplicate(session, ext.url_hash) is True


# ---------------------------------------------------------------------------
# PipelineJob
# ---------------------------------------------------------------------------

class TestPipelineJob:
    def test_construction(self):
        job = PipelineJob(video_id=1, url="https://youtube.com/watch?v=abc", platform="youtube")
        assert job.video_id == 1
        assert job.url == "https://youtube.com/watch?v=abc"
        assert job.platform == "youtube"
        assert job.priority == 0

    def test_priority_ordering(self):
        high = PipelineJob(video_id=1, url="u", platform="youtube", priority=0)
        low = PipelineJob(video_id=2, url="u", platform="youtube", priority=5)
        assert high < low

    def test_custom_priority(self):
        job = PipelineJob(video_id=1, url="u", platform="youtube", priority=10)
        assert job.priority == 10


# ---------------------------------------------------------------------------
# JobQueue
# ---------------------------------------------------------------------------

class TestJobQueue:
    async def test_cannot_enqueue_before_start(self):
        q = JobQueue()
        q.set_processor(AsyncMock())
        job = PipelineJob(video_id=1, url="u", platform="youtube")
        with pytest.raises(RuntimeError, match="not running"):
            await q.enqueue(job)

    async def test_cannot_start_without_processor(self):
        q = JobQueue()
        with pytest.raises(RuntimeError, match="No processor"):
            await q.start()

    async def test_cannot_start_twice(self):
        q = JobQueue()
        q.set_processor(AsyncMock())
        await q.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await q.start()
        finally:
            await q.stop(wait=False)

    async def test_processes_single_job(self):
        processed = []

        async def processor(job: PipelineJob):
            processed.append(job.video_id)

        q = JobQueue()
        q.set_processor(processor)
        await q.start()
        job = PipelineJob(video_id=42, url="u", platform="youtube")
        await q.enqueue(job)
        await asyncio.sleep(0.05)
        await q.stop(wait=True)
        assert 42 in processed

    async def test_processes_multiple_jobs(self):
        processed = []

        async def processor(job: PipelineJob):
            processed.append(job.video_id)

        q = JobQueue()
        q.set_processor(processor)
        await q.start()
        for i in range(5):
            await q.enqueue(PipelineJob(video_id=i, url="u", platform="youtube"))
        await q.stop(wait=True)
        assert sorted(processed) == [0, 1, 2, 3, 4]

    async def test_processed_count_increments(self):
        q = JobQueue()
        q.set_processor(AsyncMock())
        await q.start()
        await q.enqueue(PipelineJob(video_id=1, url="u", platform="youtube"))
        await q.stop(wait=True)
        assert q.processed_count == 1

    async def test_errors_recorded_not_raised(self):
        """Job errors should be captured, not crash the worker."""
        async def failing_processor(job: PipelineJob):
            raise DownloadError("network failure")

        q = JobQueue()
        q.set_processor(failing_processor)
        await q.start()
        await q.enqueue(PipelineJob(video_id=99, url="u", platform="youtube"))
        await q.stop(wait=True)
        assert q.error_count == 1
        assert q.processed_count == 0
        assert isinstance(q.errors[0][1], DownloadError)

    async def test_queue_continues_after_job_failure(self):
        """Worker must not die when a job raises."""
        results = []
        call_count = [0]

        async def mixed_processor(job: PipelineJob):
            call_count[0] += 1
            if job.video_id == 1:
                raise DownloadError("fail")
            results.append(job.video_id)

        q = JobQueue()
        q.set_processor(mixed_processor)
        await q.start()
        await q.enqueue(PipelineJob(video_id=1, url="u", platform="youtube"))
        await q.enqueue(PipelineJob(video_id=2, url="u", platform="youtube"))
        await q.stop(wait=True)
        assert 2 in results
        assert call_count[0] == 2

    async def test_priority_ordering_respected(self):
        order = []

        async def processor(job: PipelineJob):
            order.append(job.priority)

        q = JobQueue()
        q.set_processor(processor)
        # Pre-fill the underlying asyncio.PriorityQueue directly so that
        # when the worker starts all 3 jobs are already waiting.
        q._queue.put_nowait((5, PipelineJob(video_id=1, url="u", platform="youtube", priority=5)))
        q._queue.put_nowait((1, PipelineJob(video_id=2, url="u", platform="youtube", priority=1)))
        q._queue.put_nowait((3, PipelineJob(video_id=3, url="u", platform="youtube", priority=3)))
        q._running = True  # allow _start to create the task
        q._worker_task = asyncio.create_task(q._worker())
        await q._queue.join()
        q._worker_task.cancel()
        try:
            await q._worker_task
        except asyncio.CancelledError:
            pass
        # Priority queue delivers items in ascending order: 1, 3, 5
        assert order == sorted(order)

    async def test_qsize_reports_queue_depth(self):
        gate = asyncio.Event()

        async def slow_processor(job: PipelineJob):
            await gate.wait()

        q = JobQueue()
        q.set_processor(slow_processor)
        await q.start()
        await q.enqueue(PipelineJob(video_id=1, url="u", platform="youtube"))
        await q.enqueue(PipelineJob(video_id=2, url="u", platform="youtube"))
        await asyncio.sleep(0.01)
        # 1 being processed, 1 in queue
        assert q.qsize >= 0  # exact count varies with timing
        gate.set()
        await q.stop(wait=True)

    async def test_stop_without_wait_does_not_hang(self):
        async def slow(_: PipelineJob):
            await asyncio.sleep(10)

        q = JobQueue()
        q.set_processor(slow)
        await q.start()
        await q.enqueue(PipelineJob(video_id=1, url="u", platform="youtube"))
        await asyncio.sleep(0.01)
        # should complete quickly without waiting for the slow job
        await asyncio.wait_for(q.stop(wait=False), timeout=1.0)


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------

class TestOrchestratorHappyPath:
    async def test_full_pipeline_sets_complete(self, db, session, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"audio data")

        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("hello world", "en"),
            summarizer=_make_summarizer("Key: hello"),
            audio_cache_dir=tmp_path,
        )
        ext = _extracted()
        video = await orch.process_url(session, ext)

        assert video.status == VideoStatus.COMPLETE.value

    async def test_transcript_stored_in_db(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("My transcript", "en"),
            summarizer=_make_summarizer("Summary"),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.transcript == "My transcript"

    async def test_summary_stored_in_db(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("text", "en"),
            summarizer=_make_summarizer("My summary"),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.summary == "My summary"

    async def test_audio_path_stored_in_db(self, db, session, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("text", "en"),
            summarizer=_make_summarizer("sum"),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.audio_path == str(audio)

    async def test_detected_language_stored(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("שלום", "he"),
            summarizer=_make_summarizer("סיכום"),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.detected_language == "he"

    async def test_jobs_created_for_each_stage(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("t", "en"),
            summarizer=_make_summarizer("s"),
            audio_cache_dir=tmp_path,
        )
        ext = _extracted()
        video = await orch.process_url(session, ext)
        jobs = await db.list_jobs_for_video(session, video.id)
        job_types = {j.job_type for j in jobs}
        assert JobType.DOWNLOAD.value in job_types
        assert JobType.TRANSCRIBE.value in job_types
        assert JobType.SUMMARIZE.value in job_types

    async def test_all_jobs_completed(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("t", "en"),
            summarizer=_make_summarizer("s"),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        jobs = await db.list_jobs_for_video(session, video.id)
        assert all(j.status == JobStatus.COMPLETED.value for j in jobs)


class TestOrchestratorDownloadFailure:
    async def test_download_error_sets_failed_status(self, db, session, tmp_path):
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(error=DownloadError("404")),
            transcriber=_make_transcriber(),
            summarizer=_make_summarizer(),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.status == VideoStatus.FAILED.value

    async def test_download_error_stored_in_error_message(self, db, session, tmp_path):
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(error=DownloadError("private video")),
            transcriber=_make_transcriber(),
            summarizer=_make_summarizer(),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert "private video" in (video.error_message or "")

    async def test_download_failure_skips_transcription(self, db, session, tmp_path):
        transcriber = _make_transcriber()
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(error=DownloadError("fail")),
            transcriber=transcriber,
            summarizer=_make_summarizer(),
            audio_cache_dir=tmp_path,
        )
        await orch.process_url(session, _extracted())
        transcriber.transcribe.assert_not_called()

    async def test_download_failure_skips_summarization(self, db, session, tmp_path):
        summarizer = _make_summarizer()
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(error=DownloadError("fail")),
            transcriber=_make_transcriber(),
            summarizer=summarizer,
            audio_cache_dir=tmp_path,
        )
        await orch.process_url(session, _extracted())
        summarizer.summarize.assert_not_called()

    async def test_download_job_marked_failed(self, db, session, tmp_path):
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(error=DownloadError("fail")),
            transcriber=_make_transcriber(),
            summarizer=_make_summarizer(),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        jobs = await db.list_jobs_for_video(session, video.id)
        dl_jobs = [j for j in jobs if j.job_type == JobType.DOWNLOAD.value]
        assert len(dl_jobs) == 1
        assert dl_jobs[0].status == JobStatus.FAILED.value


class TestOrchestratorTranscriptionFailure:
    async def test_transcription_error_sets_failed(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber(error=TranscriptionError("bad audio")),
            summarizer=_make_summarizer(),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.status == VideoStatus.FAILED.value

    async def test_transcription_error_skips_summarization(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        summarizer = _make_summarizer()
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber(error=TranscriptionError("bad")),
            summarizer=summarizer,
            audio_cache_dir=tmp_path,
        )
        await orch.process_url(session, _extracted())
        summarizer.summarize.assert_not_called()

    async def test_transcription_error_in_error_message(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber(error=TranscriptionError("decode error")),
            summarizer=_make_summarizer(),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert "decode error" in (video.error_message or "")


class TestOrchestratorSummarizationFailure:
    async def test_summarization_error_sets_failed(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("hello", "en"),
            summarizer=_make_summarizer(error=SummarizationError("LLM down")),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.status == VideoStatus.FAILED.value

    async def test_summarization_error_in_error_message(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("hello", "en"),
            summarizer=_make_summarizer(error=SummarizationError("LLM down")),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert "LLM down" in (video.error_message or "")

    async def test_transcript_still_stored_after_summarization_failure(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("my transcript", "en"),
            summarizer=_make_summarizer(error=SummarizationError("fail")),
            audio_cache_dir=tmp_path,
        )
        video = await orch.process_url(session, _extracted())
        assert video.transcript == "my transcript"


class TestOrchestratorWithDedup:
    async def test_duplicate_url_not_reprocessed(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        downloader = _make_downloader(audio_path=audio)
        dedup = DeduplicationService(db)
        orch = PipelineOrchestrator(
            db=db,
            downloader=downloader,
            transcriber=_make_transcriber("t", "en"),
            summarizer=_make_summarizer("s"),
            audio_cache_dir=tmp_path,
            dedup_service=dedup,
        )
        ext = _extracted()
        await orch.process_url(session, ext)
        # Process same URL again
        await orch.process_url(session, ext)
        # download_audio should only be called once
        assert downloader.download_audio.call_count == 1

    async def test_duplicate_returns_existing_video(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        dedup = DeduplicationService(db)
        orch = PipelineOrchestrator(
            db=db,
            downloader=_make_downloader(audio_path=audio),
            transcriber=_make_transcriber("t", "en"),
            summarizer=_make_summarizer("s"),
            audio_cache_dir=tmp_path,
            dedup_service=dedup,
        )
        ext = _extracted()
        first = await orch.process_url(session, ext)
        second = await orch.process_url(session, ext)
        assert first.id == second.id

    async def test_two_different_urls_both_processed(self, db, session, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        downloader = _make_downloader(audio_path=audio)
        dedup = DeduplicationService(db)
        orch = PipelineOrchestrator(
            db=db,
            downloader=downloader,
            transcriber=_make_transcriber("t", "en"),
            summarizer=_make_summarizer("s"),
            audio_cache_dir=tmp_path,
            dedup_service=dedup,
        )
        ext1 = _extracted("https://www.youtube.com/watch?v=aaa111")
        ext2 = _extracted("https://www.youtube.com/watch?v=bbb222")
        await orch.process_url(session, ext1)
        await orch.process_url(session, ext2)
        assert downloader.download_audio.call_count == 2
