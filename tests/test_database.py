from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.database import Database
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
    compute_url_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video(url: str, platform: Platform = Platform.YOUTUBE) -> VideoCreate:
    return VideoCreate(
        canonical_url=url,
        url_hash=compute_url_hash(url),
        original_url=url,
        platform=platform,
        source_group="test_group",
        source_message_id="msg_001",
    )


def _yt(suffix: str) -> VideoCreate:
    return _video(f"https://youtube.com/watch?v={suffix}")


# ---------------------------------------------------------------------------
# create_video
# ---------------------------------------------------------------------------

class TestCreateVideo:
    async def test_success_returns_orm_with_id(self, db, session):
        v = await db.create_video(session, _yt("abc"))
        assert v.id is not None
        assert v.id > 0

    async def test_fields_set_correctly(self, db, session):
        data = _yt("fields")
        v = await db.create_video(session, data)
        assert v.canonical_url == data.canonical_url
        assert v.url_hash == data.url_hash
        assert v.original_url == data.original_url
        assert v.platform == Platform.YOUTUBE.value
        assert v.source_group == "test_group"
        assert v.source_message_id == "msg_001"

    async def test_initial_status_is_discovered(self, db, session):
        v = await db.create_video(session, _yt("status"))
        assert v.status == VideoStatus.DISCOVERED.value

    async def test_audio_deleted_defaults_false(self, db, session):
        v = await db.create_video(session, _yt("audio_del"))
        assert v.audio_deleted is False

    async def test_nullable_fields_are_none(self, db, session):
        v = await db.create_video(session, _yt("nulls"))
        assert v.title is None
        assert v.transcript is None
        assert v.summary is None
        assert v.detected_language is None
        assert v.audio_path is None
        assert v.error_message is None
        assert v.completed_at is None

    async def test_created_at_is_set(self, db, session):
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        v = await db.create_video(session, _yt("ts"))
        after = datetime.now(timezone.utc).replace(tzinfo=None)
        assert isinstance(v.created_at, datetime)
        assert before <= v.created_at <= after

    async def test_duplicate_hash_raises_duplicate_video_error(self, db, session):
        data = _yt("dup")
        await db.create_video(session, data)
        with pytest.raises(DuplicateVideoError, match=data.url_hash):
            await db.create_video(session, data)

    async def test_same_url_different_objects_still_raises(self, db, session):
        url = "https://youtube.com/watch?v=same"
        await db.create_video(session, _video(url))
        with pytest.raises(DuplicateVideoError):
            await db.create_video(session, _video(url))

    async def test_different_urls_both_succeed(self, db, session):
        v1 = await db.create_video(session, _yt("uniq1"))
        v2 = await db.create_video(session, _yt("uniq2"))
        assert v1.id != v2.id

    async def test_facebook_platform_stored(self, db, session):
        v = await db.create_video(
            session, _video("https://facebook.com/reel/12345", Platform.FACEBOOK)
        )
        assert v.platform == Platform.FACEBOOK.value

    async def test_no_source_group_stored_as_none(self, db, session):
        url = "https://youtube.com/watch?v=nosrc"
        data = VideoCreate(
            canonical_url=url,
            url_hash=compute_url_hash(url),
            original_url=url,
            platform=Platform.YOUTUBE,
        )
        v = await db.create_video(session, data)
        assert v.source_group is None
        assert v.source_message_id is None


# ---------------------------------------------------------------------------
# get_video_by_id
# ---------------------------------------------------------------------------

class TestGetVideoById:
    async def test_returns_correct_video(self, db, session):
        created = await db.create_video(session, _yt("get_id"))
        fetched = await db.get_video_by_id(session, created.id)
        assert fetched.id == created.id
        assert fetched.canonical_url == created.canonical_url

    async def test_not_found_raises_video_not_found_error(self, db, session):
        with pytest.raises(VideoNotFoundError, match="id=99999"):
            await db.get_video_by_id(session, 99999)

    async def test_error_message_contains_id(self, db, session):
        try:
            await db.get_video_by_id(session, 42)
        except VideoNotFoundError as exc:
            assert "42" in str(exc)

    async def test_id_zero_raises(self, db, session):
        with pytest.raises(VideoNotFoundError):
            await db.get_video_by_id(session, 0)


# ---------------------------------------------------------------------------
# get_video_by_hash
# ---------------------------------------------------------------------------

class TestGetVideoByHash:
    async def test_returns_correct_video(self, db, session):
        data = _yt("by_hash")
        created = await db.create_video(session, data)
        fetched = await db.get_video_by_hash(session, data.url_hash)
        assert fetched.id == created.id

    async def test_not_found_raises_video_not_found_error(self, db, session):
        fake = "a" * 64
        with pytest.raises(VideoNotFoundError, match=fake):
            await db.get_video_by_hash(session, fake)

    async def test_wrong_hash_raises(self, db, session):
        await db.create_video(session, _yt("hash_miss"))
        with pytest.raises(VideoNotFoundError):
            await db.get_video_by_hash(session, "b" * 64)


# ---------------------------------------------------------------------------
# update_video_status
# ---------------------------------------------------------------------------

class TestUpdateVideoStatus:
    async def test_discovered_to_downloading(self, db, session):
        v = await db.create_video(session, _yt("trans1"))
        updated = await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        assert updated.status == VideoStatus.DOWNLOADING.value

    async def test_full_happy_path_transitions(self, db, session):
        v = await db.create_video(session, _yt("fullpath"))
        pipeline = [
            VideoStatus.DOWNLOADING,
            VideoStatus.DOWNLOADED,
            VideoStatus.TRANSCRIBING,
            VideoStatus.TRANSCRIBED,
            VideoStatus.SUMMARIZING,
            VideoStatus.COMPLETE,
        ]
        for status in pipeline:
            v = await db.update_video_status(session, v.id, status)
        assert v.status == VideoStatus.COMPLETE.value

    async def test_complete_sets_completed_at(self, db, session):
        v = await db.create_video(session, _yt("comp_at"))
        for s in [
            VideoStatus.DOWNLOADING, VideoStatus.DOWNLOADED,
            VideoStatus.TRANSCRIBING, VideoStatus.TRANSCRIBED,
            VideoStatus.SUMMARIZING, VideoStatus.COMPLETE,
        ]:
            v = await db.update_video_status(session, v.id, s)
        assert v.completed_at is not None
        assert isinstance(v.completed_at, datetime)

    async def test_non_complete_transitions_do_not_set_completed_at(self, db, session):
        v = await db.create_video(session, _yt("no_comp_at"))
        v = await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        assert v.completed_at is None

    async def test_invalid_transition_discovered_to_complete_raises(self, db, session):
        v = await db.create_video(session, _yt("inv1"))
        with pytest.raises(InvalidStatusTransitionError, match="discovered"):
            await db.update_video_status(session, v.id, VideoStatus.COMPLETE)

    async def test_invalid_transition_discovered_to_transcribing_raises(self, db, session):
        v = await db.create_video(session, _yt("inv2"))
        with pytest.raises(InvalidStatusTransitionError):
            await db.update_video_status(session, v.id, VideoStatus.TRANSCRIBING)

    async def test_invalid_transition_complete_to_anything_raises(self, db, session):
        v = await db.create_video(session, _yt("terminal"))
        for s in [
            VideoStatus.DOWNLOADING, VideoStatus.DOWNLOADED,
            VideoStatus.TRANSCRIBING, VideoStatus.TRANSCRIBED,
            VideoStatus.SUMMARIZING, VideoStatus.COMPLETE,
        ]:
            v = await db.update_video_status(session, v.id, s)
        with pytest.raises(InvalidStatusTransitionError):
            await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)

    async def test_failed_from_downloading(self, db, session):
        v = await db.create_video(session, _yt("fail_dl"))
        v = await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        v = await db.update_video_status(session, v.id, VideoStatus.FAILED)
        assert v.status == VideoStatus.FAILED.value

    async def test_failed_from_transcribing(self, db, session):
        v = await db.create_video(session, _yt("fail_tr"))
        for s in [VideoStatus.DOWNLOADING, VideoStatus.DOWNLOADED, VideoStatus.TRANSCRIBING]:
            v = await db.update_video_status(session, v.id, s)
        v = await db.update_video_status(session, v.id, VideoStatus.FAILED)
        assert v.status == VideoStatus.FAILED.value

    async def test_retry_from_failed_goes_to_discovered(self, db, session):
        v = await db.create_video(session, _yt("retry"))
        v = await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        v = await db.update_video_status(session, v.id, VideoStatus.FAILED)
        v = await db.update_video_status(session, v.id, VideoStatus.DISCOVERED)
        assert v.status == VideoStatus.DISCOVERED.value

    async def test_status_update_on_nonexistent_video_raises(self, db, session):
        with pytest.raises(VideoNotFoundError):
            await db.update_video_status(session, 99999, VideoStatus.DOWNLOADING)

    async def test_error_message_mentions_current_and_target_status(self, db, session):
        v = await db.create_video(session, _yt("err_msg"))
        try:
            await db.update_video_status(session, v.id, VideoStatus.COMPLETE)
        except InvalidStatusTransitionError as exc:
            assert "discovered" in str(exc).lower()
            assert "complete" in str(exc).lower()


# ---------------------------------------------------------------------------
# update_video
# ---------------------------------------------------------------------------

class TestUpdateVideo:
    async def test_update_transcript(self, db, session):
        v = await db.create_video(session, _yt("upd_tx"))
        v = await db.update_video(session, v.id, VideoUpdate(transcript="Hello world"))
        assert v.transcript == "Hello world"

    async def test_update_summary(self, db, session):
        v = await db.create_video(session, _yt("upd_sum"))
        v = await db.update_video(session, v.id, VideoUpdate(summary="A summary"))
        assert v.summary == "A summary"

    async def test_update_detected_language(self, db, session):
        v = await db.create_video(session, _yt("upd_lang"))
        v = await db.update_video(session, v.id, VideoUpdate(detected_language="he"))
        assert v.detected_language == "he"

    async def test_update_audio_path(self, db, session):
        v = await db.create_video(session, _yt("upd_ap"))
        v = await db.update_video(session, v.id, VideoUpdate(audio_path="/data/audio/x.mp3"))
        assert v.audio_path == "/data/audio/x.mp3"

    async def test_update_error_message(self, db, session):
        v = await db.create_video(session, _yt("upd_err"))
        v = await db.update_video(session, v.id, VideoUpdate(error_message="Network timeout"))
        assert v.error_message == "Network timeout"

    async def test_partial_update_does_not_clear_other_fields(self, db, session):
        v = await db.create_video(session, _yt("upd_partial"))
        await db.update_video(session, v.id, VideoUpdate(transcript="T1"))
        v = await db.update_video(session, v.id, VideoUpdate(summary="S1"))
        assert v.transcript == "T1"
        assert v.summary == "S1"

    async def test_second_update_overwrites_first(self, db, session):
        v = await db.create_video(session, _yt("upd_overwrite"))
        await db.update_video(session, v.id, VideoUpdate(transcript="First"))
        v = await db.update_video(session, v.id, VideoUpdate(transcript="Second"))
        assert v.transcript == "Second"

    async def test_update_nonexistent_video_raises(self, db, session):
        with pytest.raises(VideoNotFoundError):
            await db.update_video(session, 99999, VideoUpdate(transcript="x"))

    async def test_empty_update_is_noop(self, db, session):
        v = await db.create_video(session, _yt("upd_noop"))
        v2 = await db.update_video(session, v.id, VideoUpdate())
        assert v2.transcript is None
        assert v2.summary is None


# ---------------------------------------------------------------------------
# mark_audio_deleted
# ---------------------------------------------------------------------------

class TestMarkAudioDeleted:
    async def test_sets_audio_deleted_true(self, db, session):
        v = await db.create_video(session, _yt("del_aud"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/x.mp3"))
        v = await db.mark_audio_deleted(session, v.id)
        assert v.audio_deleted is True

    async def test_clears_audio_path(self, db, session):
        v = await db.create_video(session, _yt("del_path"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/x.mp3"))
        v = await db.mark_audio_deleted(session, v.id)
        assert v.audio_path is None

    async def test_is_idempotent(self, db, session):
        v = await db.create_video(session, _yt("del_idem"))
        await db.mark_audio_deleted(session, v.id)
        v = await db.mark_audio_deleted(session, v.id)
        assert v.audio_deleted is True

    async def test_preserves_transcript(self, db, session):
        v = await db.create_video(session, _yt("del_preserve"))
        await db.update_video(session, v.id, VideoUpdate(transcript="Keep me"))
        v = await db.mark_audio_deleted(session, v.id)
        assert v.transcript == "Keep me"

    async def test_preserves_summary(self, db, session):
        v = await db.create_video(session, _yt("del_sum"))
        await db.update_video(session, v.id, VideoUpdate(summary="Keep summary"))
        v = await db.mark_audio_deleted(session, v.id)
        assert v.summary == "Keep summary"

    async def test_nonexistent_video_raises(self, db, session):
        with pytest.raises(VideoNotFoundError):
            await db.mark_audio_deleted(session, 99999)


# ---------------------------------------------------------------------------
# list_videos
# ---------------------------------------------------------------------------

class TestListVideos:
    async def test_empty_database_returns_empty_list(self, db, session):
        result = await db.list_videos(session)
        assert result == []

    async def test_returns_all_created_videos(self, db, session):
        for i in range(3):
            await db.create_video(session, _yt(f"list_{i}"))
        result = await db.list_videos(session)
        assert len(result) == 3

    async def test_filter_by_platform_youtube(self, db, session):
        await db.create_video(session, _yt("plat_yt"))
        await db.create_video(
            session, _video("https://facebook.com/reel/999", Platform.FACEBOOK)
        )
        result = await db.list_videos(session, platform=Platform.YOUTUBE)
        assert all(v.platform == Platform.YOUTUBE.value for v in result)

    async def test_filter_by_platform_facebook(self, db, session):
        await db.create_video(session, _yt("plat_yt2"))
        await db.create_video(
            session, _video("https://facebook.com/reel/888", Platform.FACEBOOK)
        )
        result = await db.list_videos(session, platform=Platform.FACEBOOK)
        assert all(v.platform == Platform.FACEBOOK.value for v in result)
        assert len(result) == 1

    async def test_filter_by_status(self, db, session):
        v = await db.create_video(session, _yt("stat_filter"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        result = await db.list_videos(session, status=VideoStatus.DOWNLOADING)
        assert any(r.id == v.id for r in result)
        assert all(r.status == VideoStatus.DOWNLOADING.value for r in result)

    async def test_filter_no_results_returns_empty(self, db, session):
        await db.create_video(session, _yt("no_res"))
        result = await db.list_videos(session, status=VideoStatus.COMPLETE)
        assert result == []

    async def test_limit_restricts_count(self, db, session):
        for i in range(5):
            await db.create_video(session, _yt(f"limit_{i}"))
        result = await db.list_videos(session, limit=3)
        assert len(result) == 3

    async def test_offset_skips_records(self, db, session):
        for i in range(4):
            await db.create_video(session, _yt(f"offset_{i}"))
        page1 = await db.list_videos(session, limit=2, offset=0)
        page2 = await db.list_videos(session, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        ids1 = {v.id for v in page1}
        ids2 = {v.id for v in page2}
        assert ids1.isdisjoint(ids2)

    async def test_ordered_by_created_at_descending(self, db, session):
        for i in range(3):
            await db.create_video(session, _yt(f"order_{i}"))
        result = await db.list_videos(session)
        dates = [v.created_at for v in result]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# Job operations
# ---------------------------------------------------------------------------

class TestCreateJob:
    async def test_create_job_returns_pending(self, db, session):
        v = await db.create_video(session, _yt("job_create"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        assert job.id is not None
        assert job.video_id == v.id
        assert job.job_type == JobType.DOWNLOAD.value
        assert job.status == JobStatus.PENDING.value

    async def test_create_job_for_nonexistent_video_raises(self, db, session):
        with pytest.raises(VideoNotFoundError):
            await db.create_job(session, JobCreate(video_id=99999, job_type=JobType.DOWNLOAD))

    async def test_multiple_job_types_for_same_video(self, db, session):
        v = await db.create_video(session, _yt("multi_job"))
        j1 = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        j2 = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        j3 = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.SUMMARIZE))
        assert j1.id != j2.id != j3.id

    async def test_job_created_at_is_set(self, db, session):
        v = await db.create_video(session, _yt("job_ts"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        assert isinstance(job.created_at, datetime)


class TestGetJobById:
    async def test_returns_correct_job(self, db, session):
        v = await db.create_video(session, _yt("getjob"))
        created = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        fetched = await db.get_job_by_id(session, created.id)
        assert fetched.id == created.id
        assert fetched.job_type == JobType.TRANSCRIBE.value

    async def test_not_found_raises(self, db, session):
        with pytest.raises(JobNotFoundError, match="id=99999"):
            await db.get_job_by_id(session, 99999)


class TestListJobsForVideo:
    async def test_returns_all_jobs_for_video(self, db, session):
        v = await db.create_video(session, _yt("listjobs"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        jobs = await db.list_jobs_for_video(session, v.id)
        assert len(jobs) == 2

    async def test_empty_for_video_with_no_jobs(self, db, session):
        v = await db.create_video(session, _yt("nojobs"))
        jobs = await db.list_jobs_for_video(session, v.id)
        assert jobs == []

    async def test_does_not_return_other_videos_jobs(self, db, session):
        v1 = await db.create_video(session, _yt("v1_jobs"))
        v2 = await db.create_video(session, _yt("v2_jobs"))
        await db.create_job(session, JobCreate(video_id=v1.id, job_type=JobType.DOWNLOAD))
        jobs_v2 = await db.list_jobs_for_video(session, v2.id)
        assert jobs_v2 == []

    async def test_nonexistent_video_raises(self, db, session):
        with pytest.raises(VideoNotFoundError):
            await db.list_jobs_for_video(session, 99999)


class TestUpdateJobStatus:
    async def test_transition_to_running_sets_started_at(self, db, session):
        v = await db.create_video(session, _yt("job_run"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        updated = await db.update_job_status(session, job.id, JobStatus.RUNNING)
        assert updated.status == JobStatus.RUNNING.value
        assert updated.started_at is not None
        assert isinstance(updated.started_at, datetime)

    async def test_transition_to_completed_sets_completed_at(self, db, session):
        v = await db.create_video(session, _yt("job_comp"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        await db.update_job_status(session, job.id, JobStatus.RUNNING)
        completed = await db.update_job_status(session, job.id, JobStatus.COMPLETED)
        assert completed.status == JobStatus.COMPLETED.value
        assert completed.completed_at is not None

    async def test_transition_to_failed_sets_completed_at(self, db, session):
        v = await db.create_video(session, _yt("job_fail"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        failed = await db.update_job_status(session, job.id, JobStatus.FAILED, error="Timeout")
        assert failed.status == JobStatus.FAILED.value
        assert failed.completed_at is not None

    async def test_error_stored_on_failure(self, db, session):
        v = await db.create_video(session, _yt("job_err"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        updated = await db.update_job_status(
            session, job.id, JobStatus.FAILED, error="CUDA out of memory"
        )
        assert updated.error == "CUDA out of memory"

    async def test_pending_to_running_does_not_set_completed_at(self, db, session):
        v = await db.create_video(session, _yt("job_no_comp"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        updated = await db.update_job_status(session, job.id, JobStatus.RUNNING)
        assert updated.completed_at is None

    async def test_nonexistent_job_raises(self, db, session):
        with pytest.raises(JobNotFoundError):
            await db.update_job_status(session, 99999, JobStatus.RUNNING)

    async def test_no_error_leaves_error_field_none(self, db, session):
        v = await db.create_video(session, _yt("job_noerr"))
        job = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        updated = await db.update_job_status(session, job.id, JobStatus.RUNNING)
        assert updated.error is None
