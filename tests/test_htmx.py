"""Tests for HTMX partial routes and full-page HTML routes."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

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
from src.web.app import create_app
from src.web.dependencies import set_database
from src.core.database import Database


# ---------------------------------------------------------------------------
# Fixtures (same pattern as test_web.py)
# ---------------------------------------------------------------------------

@pytest.fixture
async def app_and_db():
    db = Database("sqlite+aiosqlite:///:memory:")
    app = create_app(database_url="sqlite+aiosqlite:///:memory:")
    set_database(db)
    await db.create_tables()
    yield app, db
    await db.drop_tables()
    await db.close()


@pytest.fixture
async def client(app_and_db):
    app, _ = app_and_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
async def db(app_and_db):
    _, db = app_and_db
    return db


@pytest.fixture
async def session(db):
    async with db._session_factory() as s:
        yield s


def _vid(suffix: str, platform: Platform = Platform.YOUTUBE) -> VideoCreate:
    url = f"https://www.youtube.com/watch?v={suffix}"
    return VideoCreate(
        canonical_url=url,
        url_hash=compute_url_hash(url),
        original_url=url,
        platform=platform,
    )


# ---------------------------------------------------------------------------
# Full-page HTML routes — test HTTP status and HTML content
# ---------------------------------------------------------------------------

class TestPageRoutes:
    async def test_dashboard_returns_200_html(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_dashboard_contains_dashboard_text(self, client):
        resp = await client.get("/")
        assert "Dashboard" in resp.text

    async def test_videos_page_returns_200(self, client):
        resp = await client.get("/videos")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_videos_page_contains_submit_link(self, client):
        resp = await client.get("/videos")
        assert "Submit URL" in resp.text or "/submit" in resp.text

    async def test_submit_page_returns_200(self, client):
        resp = await client.get("/submit")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_submit_page_contains_form(self, client):
        resp = await client.get("/submit")
        assert "<form" in resp.text

    async def test_jobs_page_returns_200(self, client):
        resp = await client.get("/jobs")
        assert resp.status_code == 200

    async def test_video_detail_existing_returns_200(self, client, db, session):
        v = await db.create_video(session, _vid("detail_ok"))
        resp = await client.get(f"/videos/{v.id}")
        assert resp.status_code == 200

    async def test_video_detail_contains_platform(self, client, db, session):
        v = await db.create_video(session, _vid("platform_shown"))
        resp = await client.get(f"/videos/{v.id}")
        assert "youtube" in resp.text.lower()

    async def test_video_detail_nonexistent_returns_404(self, client):
        resp = await client.get("/videos/99999")
        assert resp.status_code == 404

    async def test_video_detail_404_shows_friendly_message(self, client):
        resp = await client.get("/videos/99999")
        assert "not found" in resp.text.lower() or "404" in resp.text

    async def test_nav_links_present_in_all_pages(self, client):
        for path in ["/", "/videos", "/submit", "/jobs"]:
            resp = await client.get(path)
            # All pages share base template with nav
            assert "/videos" in resp.text
            assert "/submit" in resp.text

    async def test_htmx_script_included_in_pages(self, client):
        resp = await client.get("/")
        assert "htmx" in resp.text.lower()

    async def test_video_detail_shows_transcript_if_present(self, client, db, session):
        v = await db.create_video(session, _vid("with_tx"))
        await db.update_video(session, v.id, VideoUpdate(transcript="Hello world transcript"))
        resp = await client.get(f"/videos/{v.id}")
        assert "Hello world transcript" in resp.text

    async def test_video_detail_shows_summary_if_present(self, client, db, session):
        v = await db.create_video(session, _vid("with_sum"))
        await db.update_video(session, v.id, VideoUpdate(summary="Key summary points here"))
        resp = await client.get(f"/videos/{v.id}")
        assert "Key summary points here" in resp.text

    async def test_video_detail_shows_error_if_failed(self, client, db, session):
        v = await db.create_video(session, _vid("failed_vid"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        await db.update_video_status(session, v.id, VideoStatus.FAILED)
        await db.update_video(session, v.id, VideoUpdate(error_message="Network timeout"))
        resp = await client.get(f"/videos/{v.id}")
        assert "Network timeout" in resp.text


# ---------------------------------------------------------------------------
# HTMX partial routes
# ---------------------------------------------------------------------------

class TestHtmxDashboardStats:
    async def test_returns_html(self, client):
        resp = await client.get("/htmx/dashboard/stats")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_shows_total_count(self, client, db, session):
        await db.create_video(session, _vid("stat1"))
        await db.create_video(session, _vid("stat2"))
        resp = await client.get("/htmx/dashboard/stats")
        assert "2" in resp.text

    async def test_shows_complete_count(self, client, db, session):
        v = await db.create_video(session, _vid("comp_stat"))
        for s in [VideoStatus.DOWNLOADING, VideoStatus.DOWNLOADED,
                  VideoStatus.TRANSCRIBING, VideoStatus.TRANSCRIBED,
                  VideoStatus.SUMMARIZING, VideoStatus.COMPLETE]:
            v = await db.update_video_status(session, v.id, s)
        resp = await client.get("/htmx/dashboard/stats")
        assert "1" in resp.text

    async def test_shows_failed_count(self, client, db, session):
        v = await db.create_video(session, _vid("fail_stat"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        await db.update_video_status(session, v.id, VideoStatus.FAILED)
        resp = await client.get("/htmx/dashboard/stats")
        assert "1" in resp.text

    async def test_empty_db_shows_zeros(self, client):
        resp = await client.get("/htmx/dashboard/stats")
        assert resp.status_code == 200
        # All counts should be 0 — page should still render
        assert "0" in resp.text


class TestHtmxRecentVideos:
    async def test_returns_html(self, client):
        resp = await client.get("/htmx/videos/recent")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_shows_no_videos_message_when_empty(self, client):
        resp = await client.get("/htmx/videos/recent")
        assert "No videos found" in resp.text

    async def test_shows_created_videos(self, client, db, session):
        await db.create_video(session, _vid("recent1"))
        resp = await client.get("/htmx/videos/recent")
        assert "recent1" in resp.text or "/videos/" in resp.text

    async def test_limits_to_10(self, client, db, session):
        for i in range(15):
            await db.create_video(session, _vid(f"lim{i:03d}"))
        resp = await client.get("/htmx/videos/recent")
        # Table rows — at most 10 videos
        row_count = resp.text.count("<tr>") - 1  # -1 for header row
        assert row_count <= 10


class TestHtmxVideoTable:
    async def test_returns_html(self, client):
        resp = await client.get("/htmx/videos/table")
        assert resp.status_code == 200

    async def test_empty_db_shows_message(self, client):
        resp = await client.get("/htmx/videos/table")
        assert "No videos found" in resp.text

    async def test_shows_videos(self, client, db, session):
        await db.create_video(session, _vid("tbl1"))
        resp = await client.get("/htmx/videos/table")
        assert "/videos/" in resp.text

    async def test_platform_filter(self, client, db, session):
        await db.create_video(session, _vid("yt_filter"))
        url_ig = "https://www.instagram.com/reel/IGTEST123/"
        await db.create_video(
            session,
            VideoCreate(
                canonical_url=url_ig,
                url_hash=compute_url_hash(url_ig),
                original_url=url_ig,
                platform=Platform.INSTAGRAM,
            ),
        )
        resp = await client.get("/htmx/videos/table?platform=instagram")
        assert "instagram" in resp.text
        # YouTube row should not appear
        assert "yt_filter" not in resp.text

    async def test_status_filter(self, client, db, session):
        v = await db.create_video(session, _vid("st_filter"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        await db.create_video(session, _vid("st_filter2"))  # stays discovered
        resp = await client.get("/htmx/videos/table?status=downloading")
        assert "downloading" in resp.text

    async def test_pagination_limit(self, client, db, session):
        for i in range(10):
            await db.create_video(session, _vid(f"page{i:03d}"))
        resp = await client.get("/htmx/videos/table?limit=5")
        row_count = resp.text.count("<tr>") - 1
        assert row_count <= 5

    async def test_pagination_shows_next_button_when_full_page(self, client, db, session):
        for i in range(6):
            await db.create_video(session, _vid(f"nxt{i:03d}"))
        resp = await client.get("/htmx/videos/table?limit=5&offset=0")
        assert "Next" in resp.text

    async def test_pagination_no_next_when_fewer_than_limit(self, client, db, session):
        await db.create_video(session, _vid("single_pg"))
        resp = await client.get("/htmx/videos/table?limit=25&offset=0")
        assert "Next" not in resp.text

    async def test_pagination_prev_button_when_offset_gt_zero(self, client, db, session):
        await db.create_video(session, _vid("prev_pg"))
        resp = await client.get("/htmx/videos/table?limit=25&offset=25")
        assert "Prev" in resp.text


class TestHtmxDeleteAudio:
    async def test_delete_audio_returns_200(self, client, db, session):
        v = await db.create_video(session, _vid("del_aud"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/x.mp3"))
        resp = await client.delete(f"/htmx/videos/{v.id}/audio")
        assert resp.status_code == 200

    async def test_delete_audio_returns_deleted_text(self, client, db, session):
        v = await db.create_video(session, _vid("del_txt"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/x.mp3"))
        resp = await client.delete(f"/htmx/videos/{v.id}/audio")
        assert "Deleted" in resp.text

    async def test_delete_audio_nonexistent_returns_404(self, client):
        resp = await client.delete("/htmx/videos/99999/audio")
        assert resp.status_code == 404

    async def test_delete_audio_actually_marks_in_db(self, client, db, session):
        v = await db.create_video(session, _vid("del_db"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/x.mp3"))
        await client.delete(f"/htmx/videos/{v.id}/audio")
        # Open a fresh session to see the committed change from the route handler
        async with db._session_factory() as fresh:
            refreshed = await db.get_video_by_id(fresh, v.id)
            assert refreshed.audio_deleted is True
            assert refreshed.audio_path is None

    async def test_delete_audio_idempotent(self, client, db, session):
        v = await db.create_video(session, _vid("del_idem"))
        resp1 = await client.delete(f"/htmx/videos/{v.id}/audio")
        resp2 = await client.delete(f"/htmx/videos/{v.id}/audio")
        assert resp1.status_code == 200
        assert resp2.status_code == 200


class TestHtmxJobs:
    async def test_recent_jobs_returns_html(self, client):
        resp = await client.get("/htmx/jobs/recent")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_recent_jobs_empty_shows_message(self, client):
        resp = await client.get("/htmx/jobs/recent")
        assert "No jobs" in resp.text

    async def test_recent_jobs_shows_created_job(self, client, db, session):
        v = await db.create_video(session, _vid("job_recent"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        resp = await client.get("/htmx/jobs/recent")
        assert "download" in resp.text.lower()

    async def test_jobs_for_video_returns_html(self, client, db, session):
        v = await db.create_video(session, _vid("jobs_vid"))
        resp = await client.get(f"/htmx/jobs/video/{v.id}")
        assert resp.status_code == 200

    async def test_jobs_for_video_empty(self, client, db, session):
        v = await db.create_video(session, _vid("nojob"))
        resp = await client.get(f"/htmx/jobs/video/{v.id}")
        assert "No jobs" in resp.text

    async def test_jobs_for_video_nonexistent_returns_404(self, client):
        resp = await client.get("/htmx/jobs/video/99999")
        assert resp.status_code == 404

    async def test_jobs_for_video_shows_all_job_types(self, client, db, session):
        v = await db.create_video(session, _vid("all_types"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.SUMMARIZE))
        resp = await client.get(f"/htmx/jobs/video/{v.id}")
        assert "download" in resp.text.lower()
        assert "transcribe" in resp.text.lower()
        assert "summarize" in resp.text.lower()

    async def test_failed_job_shows_error(self, client, db, session):
        v = await db.create_video(session, _vid("err_job"))
        j = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        await db.update_job_status(session, j.id, JobStatus.FAILED, error="Connection refused")
        resp = await client.get(f"/htmx/jobs/video/{v.id}")
        assert "Connection refused" in resp.text


class TestHtmxSubmit:
    async def test_valid_youtube_url_returns_success(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert resp.status_code == 200
        assert "Queued" in resp.text or "Already" in resp.text

    async def test_valid_url_shows_video_link(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert "/videos/" in resp.text

    async def test_duplicate_url_shows_already_in_queue(self, client):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        await client.post("/htmx/submit", data={"url": url})
        resp = await client.post("/htmx/submit", data={"url": url})
        assert "Already" in resp.text or "duplicate" in resp.text.lower()

    async def test_unsupported_url_shows_error(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://www.example.com/video/123"},
        )
        assert resp.status_code == 200  # HTMX always 200, error in body
        assert "error" in resp.text.lower() or "alert" in resp.text.lower()

    async def test_empty_url_shows_error(self, client):
        resp = await client.post("/htmx/submit", data={"url": ""})
        assert resp.status_code == 200
        assert "empty" in resp.text.lower() or "error" in resp.text.lower()

    async def test_instagram_url_accepted(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://www.instagram.com/reel/CxYZ123/"},
        )
        assert resp.status_code == 200
        assert "Queued" in resp.text or "Already" in resp.text

    async def test_tiktok_url_accepted(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://vm.tiktok.com/ZMxxxxxx/"},
        )
        assert resp.status_code == 200
        assert "Queued" in resp.text or "Already" in resp.text

    async def test_facebook_url_accepted(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://www.facebook.com/reel/1234567890"},
        )
        assert resp.status_code == 200
        assert "Queued" in resp.text or "Already" in resp.text

    async def test_url_stored_in_database(self, client, db, session):
        await client.post(
            "/htmx/submit",
            data={"url": "https://www.youtube.com/watch?v=newvideo123"},
        )
        videos = await db.list_videos(session)
        assert any("newvideo123" in v.canonical_url for v in videos)

    async def test_canonical_url_shown_in_response(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://youtu.be/dQw4w9WgXcQ?si=tracking"},
        )
        # Short link should be canonicalized
        assert "youtube.com" in resp.text

    async def test_source_group_accepted(self, client):
        resp = await client.post(
            "/htmx/submit",
            data={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "source_group": "grp1"},
        )
        assert resp.status_code == 200

    async def test_missing_url_field_shows_error(self, client):
        resp = await client.post("/htmx/submit", data={})
        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "empty" in resp.text.lower()
