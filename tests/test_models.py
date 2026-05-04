from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from src.core.models import (
    VALID_STATUS_TRANSITIONS,
    JobCreate,
    JobRead,
    JobStatus,
    JobType,
    Platform,
    VideoCreate,
    VideoRead,
    VideoStatus,
    VideoUpdate,
    compute_url_hash,
)


# ---------------------------------------------------------------------------
# compute_url_hash
# ---------------------------------------------------------------------------

class TestComputeUrlHash:
    def test_returns_64_char_string(self):
        h = compute_url_hash("https://youtube.com/watch?v=test")
        assert len(h) == 64

    def test_output_is_lowercase_hex(self):
        h = compute_url_hash("https://youtube.com/watch?v=test")
        assert all(c in "0123456789abcdef" for c in h)

    def test_is_deterministic(self):
        url = "https://youtube.com/watch?v=abc"
        assert compute_url_hash(url) == compute_url_hash(url)

    def test_matches_stdlib_sha256(self):
        url = "https://youtube.com/watch?v=check"
        assert compute_url_hash(url) == hashlib.sha256(url.encode()).hexdigest()

    def test_different_urls_produce_different_hashes(self):
        assert compute_url_hash("https://a.com") != compute_url_hash("https://b.com")

    def test_empty_string_produces_valid_hash(self):
        h = compute_url_hash("")
        assert len(h) == 64

    def test_url_with_tracking_params_is_unique(self):
        base = "https://youtube.com/watch?v=abc"
        with_param = base + "&utm_source=whatsapp"
        assert compute_url_hash(base) != compute_url_hash(with_param)

    def test_unicode_url(self):
        h = compute_url_hash("https://example.com/שלום")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# VideoStatus
# ---------------------------------------------------------------------------

class TestVideoStatus:
    def test_all_expected_values_exist(self):
        values = {s.value for s in VideoStatus}
        assert values == {
            "discovered", "downloading", "downloaded",
            "transcribing", "transcribed", "summarizing",
            "complete", "failed",
        }

    def test_string_construction(self):
        assert VideoStatus("discovered") == VideoStatus.DISCOVERED
        assert VideoStatus("complete") == VideoStatus.COMPLETE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            VideoStatus("processing")

    def test_is_string_subclass(self):
        assert isinstance(VideoStatus.DISCOVERED, str)


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

class TestPlatform:
    def test_all_expected_values_exist(self):
        values = {p.value for p in Platform}
        assert values == {"youtube", "instagram", "tiktok", "facebook"}

    def test_string_construction(self):
        assert Platform("youtube") == Platform.YOUTUBE
        assert Platform("facebook") == Platform.FACEBOOK

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError):
            Platform("snapchat")

    def test_twitter_not_supported(self):
        with pytest.raises(ValueError):
            Platform("twitter")


# ---------------------------------------------------------------------------
# JobStatus / JobType
# ---------------------------------------------------------------------------

class TestJobStatus:
    def test_all_values(self):
        values = {s.value for s in JobStatus}
        assert values == {"pending", "running", "completed", "failed"}


class TestJobType:
    def test_all_values(self):
        values = {t.value for t in JobType}
        assert values == {"download", "transcribe", "summarize"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            JobType("upload")


# ---------------------------------------------------------------------------
# VALID_STATUS_TRANSITIONS
# ---------------------------------------------------------------------------

class TestValidStatusTransitions:
    def test_every_status_has_an_entry(self):
        for status in VideoStatus:
            assert status in VALID_STATUS_TRANSITIONS

    def test_discovered_can_download_or_fail(self):
        allowed = VALID_STATUS_TRANSITIONS[VideoStatus.DISCOVERED]
        assert VideoStatus.DOWNLOADING in allowed
        assert VideoStatus.FAILED in allowed

    def test_discovered_cannot_skip_to_downloaded(self):
        assert VideoStatus.DOWNLOADED not in VALID_STATUS_TRANSITIONS[VideoStatus.DISCOVERED]

    def test_complete_is_terminal(self):
        assert VALID_STATUS_TRANSITIONS[VideoStatus.COMPLETE] == frozenset()

    def test_failed_can_only_retry_to_discovered(self):
        allowed = VALID_STATUS_TRANSITIONS[VideoStatus.FAILED]
        assert allowed == frozenset({VideoStatus.DISCOVERED})

    def test_full_forward_chain_is_valid(self):
        chain = [
            VideoStatus.DISCOVERED,
            VideoStatus.DOWNLOADING,
            VideoStatus.DOWNLOADED,
            VideoStatus.TRANSCRIBING,
            VideoStatus.TRANSCRIBED,
            VideoStatus.SUMMARIZING,
            VideoStatus.COMPLETE,
        ]
        for current, nxt in zip(chain, chain[1:]):
            assert nxt in VALID_STATUS_TRANSITIONS[current]

    def test_no_status_can_transition_to_discovered_except_failed(self):
        can_reach_discovered = [
            s for s, transitions in VALID_STATUS_TRANSITIONS.items()
            if VideoStatus.DISCOVERED in transitions
        ]
        assert can_reach_discovered == [VideoStatus.FAILED]


# ---------------------------------------------------------------------------
# VideoCreate
# ---------------------------------------------------------------------------

class TestVideoCreate:
    def _make(self, **kwargs) -> VideoCreate:
        url = kwargs.pop("canonical_url", "https://youtube.com/watch?v=abc")
        defaults = dict(
            canonical_url=url,
            url_hash=compute_url_hash(url),
            original_url=url,
            platform=Platform.YOUTUBE,
        )
        defaults.update(kwargs)
        return VideoCreate(**defaults)

    def test_valid_construction(self):
        vc = self._make()
        assert vc.platform == Platform.YOUTUBE
        assert len(vc.url_hash) == 64

    def test_empty_canonical_url_raises(self):
        with pytest.raises(ValidationError, match="canonical_url"):
            VideoCreate(
                canonical_url="",
                url_hash="a" * 64,
                original_url="https://x.com",
                platform=Platform.YOUTUBE,
            )

    def test_whitespace_only_canonical_url_raises(self):
        with pytest.raises(ValidationError, match="canonical_url"):
            VideoCreate(
                canonical_url="   ",
                url_hash="a" * 64,
                original_url="https://x.com",
                platform=Platform.YOUTUBE,
            )

    def test_url_hash_too_short_raises(self):
        with pytest.raises(ValidationError, match="url_hash"):
            VideoCreate(
                canonical_url="https://x.com",
                url_hash="abc123",
                original_url="https://x.com",
                platform=Platform.YOUTUBE,
            )

    def test_url_hash_too_long_raises(self):
        with pytest.raises(ValidationError, match="url_hash"):
            VideoCreate(
                canonical_url="https://x.com",
                url_hash="a" * 65,
                original_url="https://x.com",
                platform=Platform.YOUTUBE,
            )

    def test_url_hash_non_hex_raises(self):
        with pytest.raises(ValidationError, match="url_hash"):
            VideoCreate(
                canonical_url="https://x.com",
                url_hash="z" * 64,
                original_url="https://x.com",
                platform=Platform.YOUTUBE,
            )

    def test_url_hash_uppercase_normalized_to_lowercase(self):
        url = "https://youtube.com/watch?v=x"
        vc = VideoCreate(
            canonical_url=url,
            url_hash=compute_url_hash(url).upper(),
            original_url=url,
            platform=Platform.YOUTUBE,
        )
        assert vc.url_hash == vc.url_hash.lower()

    def test_invalid_platform_string_raises(self):
        with pytest.raises(ValidationError):
            VideoCreate(
                canonical_url="https://x.com",
                url_hash="a" * 64,
                original_url="https://x.com",
                platform="snapchat",  # type: ignore[arg-type]
            )

    def test_optional_fields_default_none(self):
        url = "https://youtube.com/watch?v=opts"
        vc = VideoCreate(
            canonical_url=url,
            url_hash=compute_url_hash(url),
            original_url=url,
            platform=Platform.YOUTUBE,
        )
        assert vc.source_group is None
        assert vc.source_message_id is None

    def test_source_group_stored(self):
        url = "https://youtube.com/watch?v=grp"
        vc = VideoCreate(
            canonical_url=url,
            url_hash=compute_url_hash(url),
            original_url=url,
            platform=Platform.YOUTUBE,
            source_group="My Group",
        )
        assert vc.source_group == "My Group"

    def test_all_platforms_accepted(self):
        for platform in Platform:
            url = f"https://example.com/{platform.value}"
            vc = VideoCreate(
                canonical_url=url,
                url_hash=compute_url_hash(url),
                original_url=url,
                platform=platform,
            )
            assert vc.platform == platform


# ---------------------------------------------------------------------------
# VideoUpdate
# ---------------------------------------------------------------------------

class TestVideoUpdate:
    def test_empty_update_all_none(self):
        vu = VideoUpdate()
        assert vu.transcript is None
        assert vu.summary is None
        assert vu.detected_language is None
        assert vu.audio_path is None
        assert vu.error_message is None
        assert vu.title is None

    def test_partial_update(self):
        vu = VideoUpdate(transcript="Hello")
        assert vu.transcript == "Hello"
        assert vu.summary is None

    def test_model_dump_excludes_none(self):
        vu = VideoUpdate(transcript="T1")
        dumped = vu.model_dump(exclude_none=True)
        assert "transcript" in dumped
        assert "summary" not in dumped

    def test_all_fields_set(self):
        vu = VideoUpdate(
            title="My Title",
            transcript="T",
            summary="S",
            detected_language="en",
            audio_path="/a/b.mp3",
            error_message="err",
        )
        assert vu.title == "My Title"
        assert vu.transcript == "T"
        assert vu.summary == "S"
        assert vu.detected_language == "en"
        assert vu.audio_path == "/a/b.mp3"
        assert vu.error_message == "err"


# ---------------------------------------------------------------------------
# JobCreate
# ---------------------------------------------------------------------------

class TestJobCreate:
    def test_valid_download_job(self):
        jc = JobCreate(video_id=1, job_type=JobType.DOWNLOAD)
        assert jc.video_id == 1
        assert jc.job_type == JobType.DOWNLOAD

    def test_all_job_types_accepted(self):
        for jt in JobType:
            jc = JobCreate(video_id=1, job_type=jt)
            assert jc.job_type == jt

    def test_invalid_job_type_raises(self):
        with pytest.raises(ValidationError):
            JobCreate(video_id=1, job_type="upload")  # type: ignore[arg-type]

    def test_video_id_required(self):
        with pytest.raises(ValidationError):
            JobCreate(job_type=JobType.DOWNLOAD)  # type: ignore[call-arg]
