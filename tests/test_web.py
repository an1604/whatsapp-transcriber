"""Comprehensive tests for the FastAPI web layer."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.database import Database
from src.core.models import (
    JobCreate,
    JobType,
    Platform,
    VideoCreate,
    VideoStatus,
    VideoUpdate,
    compute_url_hash,
)
from src.web.app import create_app
from src.web.dependencies import set_database


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def app_and_db(tmp_path):
    """Create an in-memory FastAPI app for testing."""
    db_url = "sqlite+aiosqlite:///:memory:"
    db = Database(db_url)
    app = create_app(database_url=db_url)
    # Override the lifespan db with our controlled instance
    set_database(db)
    await db.create_tables()
    yield app, db
    await db.drop_tables()
    await db.close()


@pytest.fixture
async def client(app_and_db):
    """AsyncClient for the test app."""
    app, _ = app_and_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
async def db(app_and_db):
    app, db = app_and_db
    return db


@pytest.fixture
async def session(db):
    async with db._session_factory() as s:
        yield s


def _video_data(url_suffix: str, platform: Platform = Platform.YOUTUBE) -> VideoCreate:
    url = f"https://www.youtube.com/watch?v={url_suffix}"
    return VideoCreate(
        canonical_url=url,
        url_hash=compute_url_hash(url),
        original_url=url,
        platform=platform,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_returns_ok(self, client):
        assert resp.json()["status"] == "ok" if (resp := await client.get("/health")) else True


# ---------------------------------------------------------------------------
# create_app validation
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_blank_database_url_raises(self):
        with pytest.raises(ValueError, match="database_url"):
            create_app(database_url="  ")

    def test_empty_database_url_raises(self):
        with pytest.raises(ValueError, match="database_url"):
            create_app(database_url="")


# ---------------------------------------------------------------------------
# GET /api/videos
# ---------------------------------------------------------------------------

class TestListVideos:
    async def test_empty_database_returns_empty_list(self, client):
        resp = await client.get("/api/videos")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_created_videos(self, client, db, session):
        await db.create_video(session, _video_data("aaa"))
        await db.create_video(session, _video_data("bbb"))
        resp = await client.get("/api/videos")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_limit_param_respected(self, client, db, session):
        for i in range(5):
            await db.create_video(session, _video_data(f"v{i:03d}"))
        resp = await client.get("/api/videos?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    async def test_offset_param_respected(self, client, db, session):
        for i in range(4):
            await db.create_video(session, _video_data(f"off{i:03d}"))
        page1 = await client.get("/api/videos?limit=2&offset=0")
        page2 = await client.get("/api/videos?limit=2&offset=2")
        ids1 = {v["id"] for v in page1.json()}
        ids2 = {v["id"] for v in page2.json()}
        assert ids1.isdisjoint(ids2)

    async def test_filter_by_platform(self, client, db, session):
        await db.create_video(session, _video_data("yt1"))
        url = "https://www.instagram.com/reel/CxYZ123/"
        await db.create_video(
            session,
            VideoCreate(
                canonical_url=url,
                url_hash=compute_url_hash(url),
                original_url=url,
                platform=Platform.INSTAGRAM,
            ),
        )
        resp = await client.get("/api/videos?platform=instagram")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["platform"] == "instagram"

    async def test_filter_by_status(self, client, db, session):
        v = await db.create_video(session, _video_data("st1"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        await db.create_video(session, _video_data("st2"))
        resp = await client.get("/api/videos?video_status=downloading")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "downloading"

    async def test_limit_below_one_returns_422(self, client):
        resp = await client.get("/api/videos?limit=0")
        assert resp.status_code == 422

    async def test_limit_above_500_returns_422(self, client):
        resp = await client.get("/api/videos?limit=501")
        assert resp.status_code == 422

    async def test_negative_offset_returns_422(self, client):
        resp = await client.get("/api/videos?offset=-1")
        assert resp.status_code == 422

    async def test_response_contains_expected_fields(self, client, db, session):
        await db.create_video(session, _video_data("fields"))
        resp = await client.get("/api/videos")
        v = resp.json()[0]
        assert "id" in v
        assert "canonical_url" in v
        assert "status" in v
        assert "platform" in v
        assert "created_at" in v


# ---------------------------------------------------------------------------
# GET /api/videos/{id}
# ---------------------------------------------------------------------------

class TestGetVideo:
    async def test_returns_video_by_id(self, client, db, session):
        created = await db.create_video(session, _video_data("byid"))
        resp = await client.get(f"/api/videos/{created.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == created.id

    async def test_nonexistent_id_returns_404(self, client):
        resp = await client.get("/api/videos/99999")
        assert resp.status_code == 404

    async def test_all_fields_present(self, client, db, session):
        v = await db.create_video(session, _video_data("flds"))
        await db.update_video(session, v.id, VideoUpdate(transcript="hello", summary="sum"))
        resp = await client.get(f"/api/videos/{v.id}")
        data = resp.json()
        assert data["transcript"] == "hello"
        assert data["summary"] == "sum"

    async def test_zero_id_returns_404(self, client):
        resp = await client.get("/api/videos/0")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/videos/{id}/audio
# ---------------------------------------------------------------------------

class TestDeleteAudio:
    async def test_marks_audio_deleted(self, client, db, session):
        v = await db.create_video(session, _video_data("delaud"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/x.mp3"))
        resp = await client.delete(f"/api/videos/{v.id}/audio")
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_deleted"] is True
        assert data["audio_path"] is None

    async def test_nonexistent_video_returns_404(self, client):
        resp = await client.delete("/api/videos/99999/audio")
        assert resp.status_code == 404

    async def test_preserves_transcript_and_summary(self, client, db, session):
        v = await db.create_video(session, _video_data("delpreserve"))
        await db.update_video(session, v.id, VideoUpdate(transcript="T", summary="S"))
        resp = await client.delete(f"/api/videos/{v.id}/audio")
        data = resp.json()
        assert data["transcript"] == "T"
        assert data["summary"] == "S"


# ---------------------------------------------------------------------------
# GET /api/jobs/video/{video_id}
# ---------------------------------------------------------------------------

class TestListJobs:
    async def test_returns_jobs_for_video(self, client, db, session):
        v = await db.create_video(session, _video_data("joblist"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        resp = await client.get(f"/api/jobs/video/{v.id}")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_nonexistent_video_returns_404(self, client):
        resp = await client.get("/api/jobs/video/99999")
        assert resp.status_code == 404

    async def test_empty_for_video_with_no_jobs(self, client, db, session):
        v = await db.create_video(session, _video_data("nojobs"))
        resp = await client.get(f"/api/jobs/video/{v.id}")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_multiple_jobs_returned(self, client, db, session):
        v = await db.create_video(session, _video_data("multijobs"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        resp = await client.get(f"/api/jobs/video/{v.id}")
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}
# ---------------------------------------------------------------------------

class TestGetJob:
    async def test_returns_job_by_id(self, client, db, session):
        v = await db.create_video(session, _video_data("getjob"))
        j = await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        resp = await client.get(f"/api/jobs/{j.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == j.id

    async def test_nonexistent_job_returns_404(self, client):
        resp = await client.get("/api/jobs/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/dashboard
# ---------------------------------------------------------------------------

class TestDashboard:
    async def test_empty_db_all_zeros(self, client):
        resp = await client.get("/api/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_videos"] == 0
        assert data["complete"] == 0
        assert data["failed"] == 0
        assert data["in_progress"] == 0

    async def test_counts_complete_videos(self, client, db, session):
        v = await db.create_video(session, _video_data("comp"))
        for s in [
            VideoStatus.DOWNLOADING, VideoStatus.DOWNLOADED,
            VideoStatus.TRANSCRIBING, VideoStatus.TRANSCRIBED,
            VideoStatus.SUMMARIZING, VideoStatus.COMPLETE,
        ]:
            v = await db.update_video_status(session, v.id, s)
        resp = await client.get("/api/dashboard")
        assert resp.json()["complete"] == 1

    async def test_counts_failed_videos(self, client, db, session):
        v = await db.create_video(session, _video_data("fail"))
        v = await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        await db.update_video_status(session, v.id, VideoStatus.FAILED)
        resp = await client.get("/api/dashboard")
        assert resp.json()["failed"] == 1

    async def test_counts_in_progress_videos(self, client, db, session):
        v = await db.create_video(session, _video_data("inp"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.get("/api/dashboard")
        assert resp.json()["in_progress"] == 1

    async def test_total_videos_count(self, client, db, session):
        for i in range(3):
            await db.create_video(session, _video_data(f"tot{i:03d}"))
        resp = await client.get("/api/dashboard")
        assert resp.json()["total_videos"] == 3

    async def test_response_has_all_fields(self, client):
        resp = await client.get("/api/dashboard")
        data = resp.json()
        for field in ["total_videos", "complete", "failed", "in_progress", "discovered"]:
            assert field in data


# ---------------------------------------------------------------------------
# POST /api/submit
# ---------------------------------------------------------------------------

class TestSubmitURL:
    async def test_valid_youtube_url_accepted(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["platform"] == "youtube"
        assert data["is_duplicate"] is False

    async def test_valid_instagram_url_accepted(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.instagram.com/reel/CxYZ123/"},
        )
        assert resp.status_code == 202
        assert resp.json()["platform"] == "instagram"

    async def test_valid_tiktok_url_accepted(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://vm.tiktok.com/ZMxxxxxx/"},
        )
        assert resp.status_code == 202
        assert resp.json()["platform"] == "tiktok"

    async def test_valid_facebook_url_accepted(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.facebook.com/reel/1234567890"},
        )
        assert resp.status_code == 202
        assert resp.json()["platform"] == "facebook"

    async def test_duplicate_url_returns_202_with_flag(self, client):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        await client.post("/api/submit", json={"url": url})
        resp = await client.post("/api/submit", json={"url": url})
        assert resp.status_code == 202
        assert resp.json()["is_duplicate"] is True

    async def test_duplicate_returns_same_video_id(self, client):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        r1 = await client.post("/api/submit", json={"url": url})
        r2 = await client.post("/api/submit", json={"url": url})
        assert r1.json()["video_id"] == r2.json()["video_id"]

    async def test_unsupported_url_returns_422(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.example.com/video/123"},
        )
        assert resp.status_code == 422

    async def test_empty_url_returns_422(self, client):
        resp = await client.post("/api/submit", json={"url": ""})
        assert resp.status_code == 422

    async def test_whitespace_only_url_returns_422(self, client):
        resp = await client.post("/api/submit", json={"url": "   "})
        assert resp.status_code == 422

    async def test_missing_url_field_returns_422(self, client):
        resp = await client.post("/api/submit", json={})
        assert resp.status_code == 422

    async def test_canonical_url_in_response(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://youtu.be/dQw4w9WgXcQ"},
        )
        data = resp.json()
        assert "youtube.com" in data["canonical_url"]
        assert "dQw4w9WgXcQ" in data["canonical_url"]

    async def test_video_id_in_response(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert isinstance(resp.json()["video_id"], int)
        assert resp.json()["video_id"] > 0

    async def test_video_stored_in_database(self, client, db, session):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.youtube.com/watch?v=abc123xyz"},
        )
        video_id = resp.json()["video_id"]
        video = await db.get_video_by_id(session, video_id)
        assert video is not None
        assert video.status == VideoStatus.DISCOVERED.value

    async def test_source_group_optional(self, client):
        resp = await client.post(
            "/api/submit",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "source_group": "my_group",
            },
        )
        assert resp.status_code == 202

    async def test_youtube_tracking_params_stripped(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&si=abc"},
        )
        assert resp.status_code == 202
        assert "feature" not in resp.json()["canonical_url"]

    async def test_bad_youtube_playlist_only_url_returns_422(self, client):
        resp = await client.post(
            "/api/submit",
            json={"url": "https://www.youtube.com/watch?list=PL123"},
        )
        assert resp.status_code == 422
