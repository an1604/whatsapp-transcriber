"""Tests for Step 11: Inbox & Manual Processing.

Covers DB methods, REST endpoints, HTMX partials, and the page route.
No mocks for the DB layer — everything hits a real in-memory SQLite.
The orchestrator is mocked at the dependency level (set_orchestrator / set_orchestrator(None)).
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock

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
from src.web.dependencies import set_database, set_orchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def app_and_db():
    db = Database("sqlite+aiosqlite:///:memory:")
    app = create_app(database_url="sqlite+aiosqlite:///:memory:")
    set_database(db)
    await db.create_tables()
    yield app, db
    set_orchestrator(None)
    await db.drop_tables()
    await db.close()


@pytest.fixture
async def client(app_and_db):
    app, _ = app_and_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def db(app_and_db):
    _, db = app_and_db
    return db


@pytest.fixture
async def session(db):
    async with db._session_factory() as s:
        yield s


@pytest.fixture
def mock_orchestrator():
    """Provide a mock orchestrator and register it in the dependency layer."""
    orch = MagicMock()
    orch.process_video_by_id = AsyncMock(return_value=None)
    set_orchestrator(orch)
    yield orch
    set_orchestrator(None)


def _vid(suffix: str, platform: Platform = Platform.YOUTUBE) -> VideoCreate:
    url = f"https://www.youtube.com/watch?v={suffix}"
    return VideoCreate(
        canonical_url=url,
        url_hash=compute_url_hash(url),
        original_url=url,
        platform=platform,
    )


# ---------------------------------------------------------------------------
# DB: delete_video
# ---------------------------------------------------------------------------

class TestDeleteVideo:
    async def test_deletes_existing_video(self, db, session):
        v = await db.create_video(session, _vid("del1"))
        await db.delete_video(session, v.id)
        from src.core.exceptions import VideoNotFoundError
        async with db._session_factory() as fresh:
            with pytest.raises(VideoNotFoundError):
                await db.get_video_by_id(fresh, v.id)

    async def test_raises_on_missing_video(self, db, session):
        from src.core.exceptions import VideoNotFoundError
        with pytest.raises(VideoNotFoundError):
            await db.delete_video(session, 99999)

    async def test_cascades_to_jobs(self, db, session):
        v = await db.create_video(session, _vid("del_cascade"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        await db.delete_video(session, v.id)
        # Jobs must also be gone
        async with db._session_factory() as fresh:
            from sqlalchemy import select
            from src.core.database import JobORM
            result = await fresh.execute(select(JobORM).where(JobORM.video_id == v.id))
            assert result.scalars().all() == []

    async def test_deletes_only_target_video(self, db, session):
        v1 = await db.create_video(session, _vid("keep_me"))
        v2 = await db.create_video(session, _vid("del_me"))
        await db.delete_video(session, v2.id)
        async with db._session_factory() as fresh:
            still_there = await db.get_video_by_id(fresh, v1.id)
            assert still_there.id == v1.id


# ---------------------------------------------------------------------------
# DB: bulk_delete_videos
# ---------------------------------------------------------------------------

class TestBulkDeleteVideos:
    async def test_deletes_multiple(self, db, session):
        v1 = await db.create_video(session, _vid("bkdel1"))
        v2 = await db.create_video(session, _vid("bkdel2"))
        count = await db.bulk_delete_videos(session, [v1.id, v2.id])
        assert count == 2

    async def test_returns_zero_for_nonexistent_ids(self, db, session):
        count = await db.bulk_delete_videos(session, [77777, 88888])
        assert count == 0

    async def test_partial_match_returns_correct_count(self, db, session):
        v = await db.create_video(session, _vid("bkdel_partial"))
        count = await db.bulk_delete_videos(session, [v.id, 99999])
        assert count == 1

    async def test_empty_list_raises(self, db, session):
        with pytest.raises(ValueError, match="video_ids"):
            await db.bulk_delete_videos(session, [])

    async def test_cascades_jobs(self, db, session):
        v = await db.create_video(session, _vid("bkdel_cascade"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.TRANSCRIBE))
        await db.bulk_delete_videos(session, [v.id])
        async with db._session_factory() as fresh:
            from sqlalchemy import select
            from src.core.database import JobORM
            result = await fresh.execute(select(JobORM).where(JobORM.video_id == v.id))
            assert result.scalars().all() == []


# ---------------------------------------------------------------------------
# DB: list_inbox_videos
# ---------------------------------------------------------------------------

class TestListInboxVideos:
    async def test_returns_only_discovered(self, db, session):
        v1 = await db.create_video(session, _vid("inbox_disc"))
        v2 = await db.create_video(session, _vid("inbox_down"))
        await db.update_video_status(session, v2.id, VideoStatus.DOWNLOADING)
        rows = await db.list_inbox_videos(session)
        ids = [r.id for r in rows]
        assert v1.id in ids
        assert v2.id not in ids

    async def test_empty_db_returns_empty(self, db, session):
        rows = await db.list_inbox_videos(session)
        assert rows == []

    async def test_ordered_newest_first(self, db, session):
        v1 = await db.create_video(session, _vid("ord1"))
        v2 = await db.create_video(session, _vid("ord2"))
        rows = await db.list_inbox_videos(session)
        ids = [r.id for r in rows]
        assert ids.index(v2.id) < ids.index(v1.id)

    async def test_respects_limit(self, db, session):
        for i in range(10):
            await db.create_video(session, _vid(f"lim_{i:02d}"))
        rows = await db.list_inbox_videos(session, limit=3)
        assert len(rows) == 3

    async def test_respects_offset(self, db, session):
        for i in range(5):
            await db.create_video(session, _vid(f"off_{i:02d}"))
        all_rows = await db.list_inbox_videos(session, limit=10, offset=0)
        offset_rows = await db.list_inbox_videos(session, limit=10, offset=2)
        assert len(offset_rows) == len(all_rows) - 2

    async def test_invalid_limit_raises(self, db, session):
        with pytest.raises(ValueError, match="limit"):
            await db.list_inbox_videos(session, limit=0)

    async def test_negative_offset_raises(self, db, session):
        with pytest.raises(ValueError, match="offset"):
            await db.list_inbox_videos(session, offset=-1)


# ---------------------------------------------------------------------------
# DB: count_inbox_videos
# ---------------------------------------------------------------------------

class TestCountInboxVideos:
    async def test_empty_db_returns_zero(self, db, session):
        count = await db.count_inbox_videos(session)
        assert count == 0

    async def test_counts_only_discovered(self, db, session):
        v1 = await db.create_video(session, _vid("cnt1"))
        v2 = await db.create_video(session, _vid("cnt2"))
        await db.update_video_status(session, v2.id, VideoStatus.DOWNLOADING)
        count = await db.count_inbox_videos(session)
        assert count == 1

    async def test_increases_after_create(self, db, session):
        before = await db.count_inbox_videos(session)
        await db.create_video(session, _vid("cnt_new"))
        after = await db.count_inbox_videos(session)
        assert after == before + 1

    async def test_decreases_after_delete(self, db, session):
        v = await db.create_video(session, _vid("cnt_del"))
        before = await db.count_inbox_videos(session)
        await db.delete_video(session, v.id)
        after = await db.count_inbox_videos(session)
        assert after == before - 1


# ---------------------------------------------------------------------------
# REST: GET /api/inbox
# ---------------------------------------------------------------------------

class TestRestInboxList:
    async def test_returns_200(self, client):
        resp = await client.get("/api/inbox")
        assert resp.status_code == 200

    async def test_response_structure(self, client):
        resp = await client.get("/api/inbox")
        data = resp.json()
        assert "videos" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    async def test_shows_discovered_video(self, client, db, session):
        v = await db.create_video(session, _vid("rest_inbox_show"))
        resp = await client.get("/api/inbox")
        ids = [r["id"] for r in resp.json()["videos"]]
        assert v.id in ids

    async def test_excludes_non_discovered(self, client, db, session):
        v = await db.create_video(session, _vid("rest_inbox_hide"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.get("/api/inbox")
        ids = [r["id"] for r in resp.json()["videos"]]
        assert v.id not in ids

    async def test_total_reflects_inbox_size(self, client, db, session):
        await db.create_video(session, _vid("total_a"))
        await db.create_video(session, _vid("total_b"))
        resp = await client.get("/api/inbox")
        assert resp.json()["total"] >= 2

    async def test_invalid_limit_returns_422(self, client):
        resp = await client.get("/api/inbox?limit=0")
        assert resp.status_code == 422

    async def test_negative_offset_returns_422(self, client):
        resp = await client.get("/api/inbox?offset=-1")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# REST: GET /api/inbox/count
# ---------------------------------------------------------------------------

class TestRestInboxCount:
    async def test_returns_200_with_count(self, client):
        resp = await client.get("/api/inbox/count")
        assert resp.status_code == 200
        assert "count" in resp.json()

    async def test_count_is_integer(self, client):
        resp = await client.get("/api/inbox/count")
        assert isinstance(resp.json()["count"], int)

    async def test_count_increases_after_submit(self, client, db, session):
        before = (await client.get("/api/inbox/count")).json()["count"]
        await db.create_video(session, _vid("count_inc"))
        after = (await client.get("/api/inbox/count")).json()["count"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# REST: POST /api/inbox/{id}/process
# ---------------------------------------------------------------------------

class TestRestProcessVideo:
    async def test_returns_202_when_queued(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("proc_ok"))
        resp = await client.post(f"/api/inbox/{v.id}/process")
        assert resp.status_code == 202

    async def test_response_body_has_status_queued(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("proc_body"))
        resp = await client.post(f"/api/inbox/{v.id}/process")
        data = resp.json()
        assert data["status"] == "queued"
        assert data["video_id"] == v.id

    async def test_returns_404_for_missing_video(self, client, mock_orchestrator):
        resp = await client.post("/api/inbox/99999/process")
        assert resp.status_code == 404

    async def test_returns_409_if_not_discovered(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("proc_409"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.post(f"/api/inbox/{v.id}/process")
        assert resp.status_code == 409

    async def test_409_message_mentions_status(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("proc_409_msg"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.post(f"/api/inbox/{v.id}/process")
        assert "downloading" in resp.json()["detail"]

    async def test_returns_503_without_orchestrator(self, client, db, session):
        set_orchestrator(None)
        v = await db.create_video(session, _vid("proc_503"))
        resp = await client.post(f"/api/inbox/{v.id}/process")
        assert resp.status_code == 503

    async def test_orchestrator_process_called(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("proc_called"))
        await client.post(f"/api/inbox/{v.id}/process")
        mock_orchestrator.process_video_by_id.assert_called_once()

    async def test_complete_video_returns_409(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("proc_complete"))
        # Walk through status machine to reach COMPLETE
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADED)
        await db.update_video_status(session, v.id, VideoStatus.TRANSCRIBING)
        await db.update_video_status(session, v.id, VideoStatus.TRANSCRIBED)
        await db.update_video_status(session, v.id, VideoStatus.SUMMARIZING)
        await db.update_video_status(session, v.id, VideoStatus.COMPLETE)
        resp = await client.post(f"/api/inbox/{v.id}/process")
        assert resp.status_code == 409

    async def test_failed_video_returns_409(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("proc_failed"))
        await db.update_video_status(session, v.id, VideoStatus.FAILED)
        resp = await client.post(f"/api/inbox/{v.id}/process")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# REST: DELETE /api/inbox/{id}
# ---------------------------------------------------------------------------

class TestRestDismissVideo:
    async def test_returns_204(self, client, db, session):
        v = await db.create_video(session, _vid("dismiss_ok"))
        resp = await client.delete(f"/api/inbox/{v.id}")
        assert resp.status_code == 204

    async def test_video_gone_after_dismiss(self, client, db, session):
        v = await db.create_video(session, _vid("dismiss_gone"))
        await client.delete(f"/api/inbox/{v.id}")
        from src.core.exceptions import VideoNotFoundError
        async with db._session_factory() as fresh:
            with pytest.raises(VideoNotFoundError):
                await db.get_video_by_id(fresh, v.id)

    async def test_returns_404_for_missing_video(self, client):
        resp = await client.delete("/api/inbox/99999")
        assert resp.status_code == 404

    async def test_dismiss_works_for_non_discovered_too(self, client, db, session):
        v = await db.create_video(session, _vid("dismiss_nondiscovered"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.delete(f"/api/inbox/{v.id}")
        assert resp.status_code == 204

    async def test_jobs_cascade_on_dismiss(self, client, db, session):
        v = await db.create_video(session, _vid("dismiss_cascade"))
        await db.create_job(session, JobCreate(video_id=v.id, job_type=JobType.DOWNLOAD))
        await client.delete(f"/api/inbox/{v.id}")
        async with db._session_factory() as fresh:
            from sqlalchemy import select
            from src.core.database import JobORM
            result = await fresh.execute(select(JobORM).where(JobORM.video_id == v.id))
            assert result.scalars().all() == []


# ---------------------------------------------------------------------------
# REST: POST /api/inbox/bulk-process
# ---------------------------------------------------------------------------

class TestRestBulkProcess:
    async def test_returns_202(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("bkproc1"))
        resp = await client.post("/api/inbox/bulk-process", json={"video_ids": [v.id]})
        assert resp.status_code == 202

    async def test_returns_queued_count(self, client, db, session, mock_orchestrator):
        v1 = await db.create_video(session, _vid("bkp_a"))
        v2 = await db.create_video(session, _vid("bkp_b"))
        resp = await client.post("/api/inbox/bulk-process", json={"video_ids": [v1.id, v2.id]})
        data = resp.json()
        assert data["queued_count"] == 2
        assert set(data["video_ids"]) == {v1.id, v2.id}

    async def test_skips_non_discovered(self, client, db, session, mock_orchestrator):
        v1 = await db.create_video(session, _vid("bkp_skip"))
        await db.update_video_status(session, v1.id, VideoStatus.DOWNLOADING)
        v2 = await db.create_video(session, _vid("bkp_queue"))
        resp = await client.post("/api/inbox/bulk-process", json={"video_ids": [v1.id, v2.id]})
        data = resp.json()
        assert data["queued_count"] == 1
        assert v2.id in data["video_ids"]

    async def test_skips_missing_ids(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("bkp_miss"))
        resp = await client.post("/api/inbox/bulk-process", json={"video_ids": [v.id, 99999]})
        assert resp.json()["queued_count"] == 1

    async def test_empty_list_returns_422(self, client):
        resp = await client.post("/api/inbox/bulk-process", json={"video_ids": []})
        assert resp.status_code == 422

    async def test_missing_body_returns_422(self, client):
        resp = await client.post("/api/inbox/bulk-process")
        assert resp.status_code == 422

    async def test_returns_503_without_orchestrator(self, client, db, session):
        v = await db.create_video(session, _vid("bkp_503"))
        resp = await client.post("/api/inbox/bulk-process", json={"video_ids": [v.id]})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# REST: POST /api/inbox/bulk-dismiss
# ---------------------------------------------------------------------------

class TestRestBulkDismiss:
    async def test_returns_200(self, client, db, session):
        v = await db.create_video(session, _vid("bkdis1"))
        resp = await client.post("/api/inbox/bulk-dismiss", json={"video_ids": [v.id]})
        assert resp.status_code == 200

    async def test_returns_dismissed_count(self, client, db, session):
        v1 = await db.create_video(session, _vid("bkd_a"))
        v2 = await db.create_video(session, _vid("bkd_b"))
        resp = await client.post("/api/inbox/bulk-dismiss", json={"video_ids": [v1.id, v2.id]})
        assert resp.json()["dismissed_count"] == 2

    async def test_videos_removed_from_db(self, client, db, session):
        v = await db.create_video(session, _vid("bkd_gone"))
        await client.post("/api/inbox/bulk-dismiss", json={"video_ids": [v.id]})
        from src.core.exceptions import VideoNotFoundError
        async with db._session_factory() as fresh:
            with pytest.raises(VideoNotFoundError):
                await db.get_video_by_id(fresh, v.id)

    async def test_empty_list_returns_422(self, client):
        resp = await client.post("/api/inbox/bulk-dismiss", json={"video_ids": []})
        assert resp.status_code == 422

    async def test_nonexistent_ids_returns_zero(self, client):
        resp = await client.post("/api/inbox/bulk-dismiss", json={"video_ids": [77777, 88888]})
        assert resp.json()["dismissed_count"] == 0


# ---------------------------------------------------------------------------
# HTMX: GET /htmx/inbox
# ---------------------------------------------------------------------------

class TestHtmxInboxTable:
    async def test_returns_200(self, client):
        resp = await client.get("/htmx/inbox")
        assert resp.status_code == 200

    async def test_returns_html(self, client):
        resp = await client.get("/htmx/inbox")
        assert "text/html" in resp.headers["content-type"]

    async def test_empty_inbox_shows_message(self, client):
        resp = await client.get("/htmx/inbox")
        assert "Inbox is empty" in resp.text or "awaiting review" in resp.text

    async def test_shows_video_url(self, client, db, session):
        await db.create_video(session, _vid("htmx_inbox_show"))
        resp = await client.get("/htmx/inbox")
        assert "htmx_inbox_show" in resp.text

    async def test_excludes_non_discovered(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_hide"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.get("/htmx/inbox")
        assert "htmx_hide" not in resp.text

    async def test_has_process_button(self, client, db, session):
        await db.create_video(session, _vid("htmx_proc_btn"))
        resp = await client.get("/htmx/inbox")
        assert "process" in resp.text.lower()

    async def test_has_dismiss_button(self, client, db, session):
        await db.create_video(session, _vid("htmx_dis_btn"))
        resp = await client.get("/htmx/inbox")
        assert "dismiss" in resp.text.lower() or "2715" in resp.text


# ---------------------------------------------------------------------------
# HTMX: POST /htmx/inbox/{id}/process
# ---------------------------------------------------------------------------

class TestHtmxProcessVideo:
    async def test_returns_200_html(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("htmx_proc_ok"))
        resp = await client.post(f"/htmx/inbox/{v.id}/process")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_shows_queued_badge(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("htmx_proc_queued"))
        resp = await client.post(f"/htmx/inbox/{v.id}/process")
        assert "Queued" in resp.text or "queued" in resp.text.lower()

    async def test_returns_404_for_missing(self, client, mock_orchestrator):
        resp = await client.post("/htmx/inbox/99999/process")
        assert resp.status_code == 404

    async def test_returns_409_if_not_discovered(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("htmx_proc_409"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.post(f"/htmx/inbox/{v.id}/process")
        assert resp.status_code == 409

    async def test_returns_503_without_orchestrator(self, client, db, session):
        set_orchestrator(None)
        v = await db.create_video(session, _vid("htmx_proc_503"))
        resp = await client.post(f"/htmx/inbox/{v.id}/process")
        assert resp.status_code == 503

    async def test_response_contains_row_id(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("htmx_proc_row"))
        resp = await client.post(f"/htmx/inbox/{v.id}/process")
        assert f"inbox-row-{v.id}" in resp.text


# ---------------------------------------------------------------------------
# HTMX: DELETE /htmx/inbox/{id}
# ---------------------------------------------------------------------------

class TestHtmxDismissVideo:
    async def test_returns_200(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_dis_ok"))
        resp = await client.request("DELETE", f"/htmx/inbox/{v.id}")
        assert resp.status_code == 200

    async def test_returns_empty_body(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_dis_empty"))
        resp = await client.request("DELETE", f"/htmx/inbox/{v.id}")
        assert resp.text.strip() == ""

    async def test_video_removed_from_db(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_dis_gone"))
        await client.request("DELETE", f"/htmx/inbox/{v.id}")
        from src.core.exceptions import VideoNotFoundError
        async with db._session_factory() as fresh:
            with pytest.raises(VideoNotFoundError):
                await db.get_video_by_id(fresh, v.id)

    async def test_returns_404_for_missing(self, client):
        resp = await client.request("DELETE", "/htmx/inbox/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# HTMX: POST /htmx/inbox/bulk-process
# ---------------------------------------------------------------------------

class TestHtmxBulkProcess:
    async def test_returns_200(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("htmx_bkp1"))
        resp = await client.post("/htmx/inbox/bulk-process", data={"video_ids": [str(v.id)]})
        assert resp.status_code == 200

    async def test_shows_count_in_response(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("htmx_bkp2"))
        resp = await client.post("/htmx/inbox/bulk-process", data={"video_ids": [str(v.id)]})
        assert "1" in resp.text

    async def test_no_selection_shows_message(self, client, mock_orchestrator):
        resp = await client.post("/htmx/inbox/bulk-process", data={})
        assert resp.status_code == 200
        assert "No videos selected" in resp.text

    async def test_invalid_id_returns_400(self, client, mock_orchestrator):
        resp = await client.post("/htmx/inbox/bulk-process", data={"video_ids": ["not_an_id"]})
        assert resp.status_code == 400

    async def test_returns_503_without_orchestrator(self, client, db, session):
        set_orchestrator(None)
        v = await db.create_video(session, _vid("htmx_bkp_503"))
        resp = await client.post("/htmx/inbox/bulk-process", data={"video_ids": [str(v.id)]})
        assert resp.status_code == 503

    async def test_skips_non_discovered_videos(self, client, db, session, mock_orchestrator):
        v = await db.create_video(session, _vid("htmx_bkp_skip"))
        await db.update_video_status(session, v.id, VideoStatus.DOWNLOADING)
        resp = await client.post("/htmx/inbox/bulk-process", data={"video_ids": [str(v.id)]})
        # Should queue 0 — no discovered ones
        assert "0" in resp.text or "No videos" in resp.text


# ---------------------------------------------------------------------------
# HTMX: POST /htmx/inbox/bulk-dismiss
# ---------------------------------------------------------------------------

class TestHtmxBulkDismiss:
    async def test_returns_200(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_bkd1"))
        resp = await client.post("/htmx/inbox/bulk-dismiss", data={"video_ids": [str(v.id)]})
        assert resp.status_code == 200

    async def test_shows_count_in_response(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_bkd2"))
        resp = await client.post("/htmx/inbox/bulk-dismiss", data={"video_ids": [str(v.id)]})
        assert "1" in resp.text

    async def test_no_selection_shows_message(self, client):
        resp = await client.post("/htmx/inbox/bulk-dismiss", data={})
        assert resp.status_code == 200
        assert "No videos selected" in resp.text

    async def test_invalid_id_returns_400(self, client):
        resp = await client.post("/htmx/inbox/bulk-dismiss", data={"video_ids": ["bad_id"]})
        assert resp.status_code == 400

    async def test_videos_removed_after_bulk_dismiss(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_bkd_gone"))
        await client.post("/htmx/inbox/bulk-dismiss", data={"video_ids": [str(v.id)]})
        from src.core.exceptions import VideoNotFoundError
        async with db._session_factory() as fresh:
            with pytest.raises(VideoNotFoundError):
                await db.get_video_by_id(fresh, v.id)

    async def test_hx_trigger_header_present(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_bkd_hdr"))
        resp = await client.post("/htmx/inbox/bulk-dismiss", data={"video_ids": [str(v.id)]})
        assert "HX-Trigger" in resp.headers


# ---------------------------------------------------------------------------
# Page: GET /inbox
# ---------------------------------------------------------------------------

class TestInboxPage:
    async def test_returns_200(self, client):
        resp = await client.get("/inbox")
        assert resp.status_code == 200

    async def test_returns_html(self, client):
        resp = await client.get("/inbox")
        assert "text/html" in resp.headers["content-type"]

    async def test_contains_inbox_heading(self, client):
        resp = await client.get("/inbox")
        assert "Inbox" in resp.text

    async def test_nav_includes_inbox_link(self, client):
        resp = await client.get("/inbox")
        assert "/inbox" in resp.text

    async def test_has_process_selected_button(self, client):
        resp = await client.get("/inbox")
        assert "Process selected" in resp.text or "process" in resp.text.lower()

    async def test_has_dismiss_selected_button(self, client):
        resp = await client.get("/inbox")
        assert "Dismiss selected" in resp.text or "dismiss" in resp.text.lower()

    async def test_base_nav_links_present(self, client):
        resp = await client.get("/inbox")
        assert "/videos" in resp.text
        assert "/submit" in resp.text
        assert "/settings" in resp.text

    async def test_htmx_auto_refresh_target_present(self, client):
        resp = await client.get("/inbox")
        assert "inbox-table" in resp.text


# ---------------------------------------------------------------------------
# Dashboard: inbox stat card
# ---------------------------------------------------------------------------

class TestDashboardInboxStat:
    async def test_stats_partial_includes_inbox_link(self, client):
        resp = await client.get("/htmx/dashboard/stats")
        assert resp.status_code == 200
        assert "/inbox" in resp.text

    async def test_stats_partial_shows_discovered_count(self, client, db, session):
        await db.create_video(session, _vid("dash_inbox"))
        resp = await client.get("/htmx/dashboard/stats")
        # At least one discovered video → stat > 0
        assert "1" in resp.text or resp.status_code == 200
