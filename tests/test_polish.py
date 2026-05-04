"""Tests for Step 10: search, bulk delete, disk stats, and settings page."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.database import Database
from src.core.models import (
    Platform,
    VideoCreate,
    VideoStatus,
    VideoUpdate,
    compute_url_hash,
)
from src.web.app import create_app
from src.web.dependencies import set_database


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


def _vid(suffix: str, platform: Platform = Platform.YOUTUBE) -> VideoCreate:
    url = f"https://www.youtube.com/watch?v={suffix}"
    return VideoCreate(
        canonical_url=url,
        url_hash=compute_url_hash(url),
        original_url=url,
        platform=platform,
    )


# ---------------------------------------------------------------------------
# Database: search_videos
# ---------------------------------------------------------------------------

class TestSearchVideos:
    async def test_finds_by_url_substring(self, db, session):
        await db.create_video(session, _vid("findme123"))
        results = await db.search_videos(session, "findme123")
        assert len(results) == 1

    async def test_returns_empty_when_no_match(self, db, session):
        await db.create_video(session, _vid("nomatch"))
        results = await db.search_videos(session, "zzz_not_here")
        assert results == []

    async def test_finds_by_transcript(self, db, session):
        v = await db.create_video(session, _vid("txtsearch"))
        await db.update_video(session, v.id, VideoUpdate(transcript="the quick brown fox"))
        results = await db.search_videos(session, "quick brown")
        assert any(r.id == v.id for r in results)

    async def test_finds_by_summary(self, db, session):
        v = await db.create_video(session, _vid("sumsearch"))
        await db.update_video(session, v.id, VideoUpdate(summary="key insights revealed"))
        results = await db.search_videos(session, "key insights")
        assert any(r.id == v.id for r in results)

    async def test_case_insensitive_like(self, db, session):
        await db.create_video(session, _vid("UPPERCASE_ID"))
        results = await db.search_videos(session, "uppercase_id")
        assert len(results) == 1

    async def test_returns_multiple_matches(self, db, session):
        await db.create_video(session, _vid("shared_tag_1"))
        await db.create_video(session, _vid("shared_tag_2"))
        results = await db.search_videos(session, "shared_tag")
        assert len(results) == 2

    async def test_respects_limit(self, db, session):
        for i in range(10):
            await db.create_video(session, _vid(f"lim_search_{i:02d}"))
        results = await db.search_videos(session, "lim_search", limit=3)
        assert len(results) <= 3

    async def test_respects_offset(self, db, session):
        for i in range(5):
            await db.create_video(session, _vid(f"off_search_{i:02d}"))
        all_results = await db.search_videos(session, "off_search", limit=10, offset=0)
        offset_results = await db.search_videos(session, "off_search", limit=10, offset=3)
        assert len(offset_results) == len(all_results) - 3

    async def test_empty_query_raises(self, db, session):
        with pytest.raises(ValueError, match="query"):
            await db.search_videos(session, "")

    async def test_blank_query_raises(self, db, session):
        with pytest.raises(ValueError, match="query"):
            await db.search_videos(session, "   ")

    async def test_invalid_limit_raises(self, db, session):
        with pytest.raises(ValueError, match="limit"):
            await db.search_videos(session, "x", limit=0)

    async def test_negative_offset_raises(self, db, session):
        with pytest.raises(ValueError, match="offset"):
            await db.search_videos(session, "x", offset=-1)


# ---------------------------------------------------------------------------
# Database: bulk_mark_audio_deleted
# ---------------------------------------------------------------------------

class TestBulkMarkAudioDeleted:
    async def test_deletes_single_video(self, db, session):
        v = await db.create_video(session, _vid("bulk1"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/a.mp3"))
        count = await db.bulk_mark_audio_deleted(session, [v.id])
        assert count == 1

    async def test_deletes_multiple_videos(self, db, session):
        v1 = await db.create_video(session, _vid("bk1"))
        v2 = await db.create_video(session, _vid("bk2"))
        count = await db.bulk_mark_audio_deleted(session, [v1.id, v2.id])
        assert count == 2

    async def test_marks_audio_deleted_in_db(self, db, session):
        v = await db.create_video(session, _vid("bulkcheck"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/b.mp3"))
        await db.bulk_mark_audio_deleted(session, [v.id])
        async with db._session_factory() as fresh:
            refreshed = await db.get_video_by_id(fresh, v.id)
            assert refreshed.audio_deleted is True
            assert refreshed.audio_path is None

    async def test_silently_skips_nonexistent_ids(self, db, session):
        count = await db.bulk_mark_audio_deleted(session, [99999, 88888])
        assert count == 0

    async def test_partial_match(self, db, session):
        v = await db.create_video(session, _vid("partial"))
        count = await db.bulk_mark_audio_deleted(session, [v.id, 99999])
        assert count == 1

    async def test_empty_list_raises(self, db, session):
        with pytest.raises(ValueError, match="video_ids"):
            await db.bulk_mark_audio_deleted(session, [])


# ---------------------------------------------------------------------------
# Database: get_disk_stats
# ---------------------------------------------------------------------------

class TestGetDiskStats:
    async def test_empty_db_returns_zeros(self, db, session):
        stats = await db.get_disk_stats(session)
        assert stats["videos_with_audio"] == 0
        assert stats["videos_audio_deleted"] == 0
        assert stats["videos_without_audio"] == 0

    async def test_counts_video_with_audio(self, db, session):
        v = await db.create_video(session, _vid("stat_audio"))
        await db.update_video(session, v.id, VideoUpdate(audio_path="/data/x.mp3"))
        stats = await db.get_disk_stats(session)
        assert stats["videos_with_audio"] == 1

    async def test_counts_deleted_audio(self, db, session):
        v = await db.create_video(session, _vid("stat_del"))
        await db.mark_audio_deleted(session, v.id)
        stats = await db.get_disk_stats(session)
        assert stats["videos_audio_deleted"] == 1

    async def test_counts_without_audio(self, db, session):
        await db.create_video(session, _vid("stat_no"))
        stats = await db.get_disk_stats(session)
        assert stats["videos_without_audio"] == 1

    async def test_all_stat_keys_present(self, db, session):
        stats = await db.get_disk_stats(session)
        assert "videos_with_audio" in stats
        assert "videos_audio_deleted" in stats
        assert "videos_without_audio" in stats


# ---------------------------------------------------------------------------
# REST API: GET /api/videos/search
# ---------------------------------------------------------------------------

class TestRestSearch:
    async def test_returns_200_with_matches(self, client, db, session):
        await db.create_video(session, _vid("rest_search_hit"))
        resp = await client.get("/api/videos/search?q=rest_search_hit")
        assert resp.status_code == 200

    async def test_returns_search_response_structure(self, client, db, session):
        await db.create_video(session, _vid("struct_check"))
        resp = await client.get("/api/videos/search?q=struct_check")
        data = resp.json()
        assert "results" in data
        assert "query" in data
        assert "limit" in data
        assert "offset" in data

    async def test_query_echoed_in_response(self, client):
        resp = await client.get("/api/videos/search?q=myquery")
        assert resp.json()["query"] == "myquery"

    async def test_empty_q_returns_422(self, client):
        resp = await client.get("/api/videos/search?q=")
        assert resp.status_code == 422

    async def test_missing_q_returns_422(self, client):
        resp = await client.get("/api/videos/search")
        assert resp.status_code == 422

    async def test_invalid_limit_returns_422(self, client):
        resp = await client.get("/api/videos/search?q=x&limit=0")
        assert resp.status_code == 422

    async def test_invalid_offset_returns_422(self, client):
        resp = await client.get("/api/videos/search?q=x&offset=-1")
        assert resp.status_code == 422

    async def test_no_results_returns_empty_list(self, client):
        resp = await client.get("/api/videos/search?q=zzz_no_match_xyzzy")
        assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# REST API: POST /api/videos/bulk-delete-audio
# ---------------------------------------------------------------------------

class TestRestBulkDeleteAudio:
    async def test_returns_200_with_count(self, client, db, session):
        v = await db.create_video(session, _vid("rest_bulk"))
        resp = await client.post(
            "/api/videos/bulk-delete-audio", json={"video_ids": [v.id]}
        )
        assert resp.status_code == 200
        assert resp.json()["deleted_count"] >= 0

    async def test_returns_video_ids_in_response(self, client, db, session):
        v = await db.create_video(session, _vid("rest_bulk2"))
        resp = await client.post(
            "/api/videos/bulk-delete-audio", json={"video_ids": [v.id]}
        )
        assert v.id in resp.json()["video_ids"]

    async def test_empty_video_ids_returns_422(self, client):
        resp = await client.post(
            "/api/videos/bulk-delete-audio", json={"video_ids": []}
        )
        assert resp.status_code == 422

    async def test_missing_body_returns_422(self, client):
        resp = await client.post("/api/videos/bulk-delete-audio")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# REST API: GET /api/dashboard — now includes disk stats
# ---------------------------------------------------------------------------

class TestDashboardDiskStats:
    async def test_dashboard_includes_audio_fields(self, client):
        resp = await client.get("/api/dashboard")
        data = resp.json()
        assert "videos_with_audio" in data
        assert "videos_audio_deleted" in data

    async def test_audio_fields_are_zero_on_empty_db(self, client):
        resp = await client.get("/api/dashboard")
        data = resp.json()
        assert data["videos_with_audio"] == 0
        assert data["videos_audio_deleted"] == 0


# ---------------------------------------------------------------------------
# HTMX: GET /htmx/videos/search
# ---------------------------------------------------------------------------

class TestHtmxSearch:
    async def test_returns_html(self, client):
        resp = await client.get("/htmx/videos/search?q=anything")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_empty_q_returns_empty_table(self, client):
        resp = await client.get("/htmx/videos/search?q=")
        assert resp.status_code == 200
        assert "No videos found" in resp.text or resp.text.strip() != ""

    async def test_missing_q_returns_empty_table(self, client):
        resp = await client.get("/htmx/videos/search")
        assert resp.status_code == 200

    async def test_shows_matching_video(self, client, db, session):
        await db.create_video(session, _vid("htmx_search_hit"))
        resp = await client.get("/htmx/videos/search?q=htmx_search_hit")
        assert "htmx_search_hit" in resp.text or "/videos/" in resp.text

    async def test_shows_no_videos_when_no_match(self, client, db, session):
        await db.create_video(session, _vid("htmx_no_match"))
        resp = await client.get("/htmx/videos/search?q=zzz_never_present_xyz")
        assert "No videos found" in resp.text


# ---------------------------------------------------------------------------
# HTMX: POST /htmx/videos/bulk-delete-audio
# ---------------------------------------------------------------------------

class TestHtmxBulkDeleteAudio:
    async def test_returns_200(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_bulk"))
        resp = await client.post(
            "/htmx/videos/bulk-delete-audio",
            data={"video_ids": [str(v.id)]},
        )
        assert resp.status_code == 200

    async def test_shows_count_in_response(self, client, db, session):
        v = await db.create_video(session, _vid("htmx_bulk2"))
        resp = await client.post(
            "/htmx/videos/bulk-delete-audio",
            data={"video_ids": [str(v.id)]},
        )
        assert "1" in resp.text or "deleted" in resp.text.lower()

    async def test_empty_selection_shows_message(self, client):
        resp = await client.post("/htmx/videos/bulk-delete-audio", data={})
        assert resp.status_code == 200
        assert "No videos selected" in resp.text

    async def test_invalid_id_returns_400(self, client):
        resp = await client.post(
            "/htmx/videos/bulk-delete-audio",
            data={"video_ids": ["not_a_number"]},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Page routes: /settings
# ---------------------------------------------------------------------------

class TestSettingsPage:
    async def test_returns_200(self, client):
        resp = await client.get("/settings")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_contains_settings_text(self, client):
        resp = await client.get("/settings")
        assert "Settings" in resp.text

    async def test_nav_includes_settings_link(self, client):
        resp = await client.get("/settings")
        assert "/settings" in resp.text

    async def test_base_nav_present(self, client):
        resp = await client.get("/settings")
        assert "/videos" in resp.text
        assert "/submit" in resp.text
