"""Tests for URL extraction from raw message text and registry."""
from __future__ import annotations

import pytest

from src.core.exceptions import PlatformNotSupportedError, URLCanonicalizationError
from src.core.models import Platform, compute_url_hash
from src.downloaders.registry import canonicalize_url, detect_platform, get_enabled_platforms
from src.scrapers.url_extractor import ExtractedURL, extract_single_url, extract_urls


# ---------------------------------------------------------------------------
# detect_platform
# ---------------------------------------------------------------------------

class TestDetectPlatform:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
        ("https://youtu.be/dQw4w9WgXcQ", Platform.YOUTUBE),
        ("https://www.youtube.com/shorts/abc123", Platform.YOUTUBE),
        ("https://www.instagram.com/reel/CxYZ123/", Platform.INSTAGRAM),
        ("https://www.instagram.com/p/CxYZ123/", Platform.INSTAGRAM),
        ("https://www.tiktok.com/@user/video/123456", Platform.TIKTOK),
        ("https://vm.tiktok.com/ZMxxxxxx/", Platform.TIKTOK),
        ("https://www.facebook.com/reel/1234567890", Platform.FACEBOOK),
        ("https://fb.watch/abc123/", Platform.FACEBOOK),
    ])
    def test_known_urls(self, url, expected):
        assert detect_platform(url) == expected

    def test_unknown_url_raises(self):
        with pytest.raises(PlatformNotSupportedError):
            detect_platform("https://www.example.com/video/123")

    def test_google_url_raises(self):
        with pytest.raises(PlatformNotSupportedError):
            detect_platform("https://www.google.com/search?q=video")

    def test_empty_string_raises(self):
        with pytest.raises(PlatformNotSupportedError):
            detect_platform("")

    def test_non_url_string_raises(self):
        with pytest.raises(PlatformNotSupportedError):
            detect_platform("just some text")


# ---------------------------------------------------------------------------
# canonicalize_url
# ---------------------------------------------------------------------------

class TestCanonicalizeUrl:
    def test_youtube_watch_canonicalized(self):
        platform, canonical = canonicalize_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share"
        )
        assert platform == Platform.YOUTUBE
        assert canonical == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_youtu_be_to_youtube(self):
        platform, canonical = canonicalize_url("https://youtu.be/dQw4w9WgXcQ")
        assert platform == Platform.YOUTUBE
        assert "v=dQw4w9WgXcQ" in canonical

    def test_instagram_reel(self):
        platform, canonical = canonicalize_url(
            "https://www.instagram.com/reel/CxYZ123/?igsh=abc"
        )
        assert platform == Platform.INSTAGRAM
        assert canonical == "https://www.instagram.com/reel/CxYZ123/"

    def test_tiktok_long_form(self):
        platform, canonical = canonicalize_url(
            "https://www.tiktok.com/@user/video/123?is_from_webapp=1"
        )
        assert platform == Platform.TIKTOK
        assert canonical == "https://www.tiktok.com/@user/video/123"

    def test_facebook_reel(self):
        platform, canonical = canonicalize_url(
            "https://www.facebook.com/reel/1234567890?__cft__=abc"
        )
        assert platform == Platform.FACEBOOK
        assert canonical == "https://www.facebook.com/reel/1234567890"

    def test_unsupported_url_raises_platform_not_supported(self):
        with pytest.raises(PlatformNotSupportedError):
            canonicalize_url("https://www.example.com/video/123")

    def test_youtube_playlist_url_raises_platform_not_supported(self):
        # Playlist-only URLs (no v=) don't match any registered YouTube video pattern
        with pytest.raises(PlatformNotSupportedError):
            canonicalize_url("https://www.youtube.com/watch?list=PL123")

    def test_instagram_profile_url_raises_platform_not_supported(self):
        # Profile URLs don't match any registered Instagram video pattern
        with pytest.raises(PlatformNotSupportedError):
            canonicalize_url("https://www.instagram.com/username/")


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------

class TestExtractUrls:
    def test_extracts_single_youtube_url(self):
        text = "Check this out: https://www.youtube.com/watch?v=dQw4w9WgXcQ great video"
        results = extract_urls(text)
        assert len(results) == 1
        assert results[0].platform == Platform.YOUTUBE
        assert "v=dQw4w9WgXcQ" in results[0].canonical_url

    def test_extracts_multiple_different_platforms(self):
        text = (
            "YouTube: https://youtu.be/dQw4w9WgXcQ "
            "Instagram: https://www.instagram.com/reel/CxYZ123/ "
            "TikTok: https://vm.tiktok.com/ZMxxxxxx/"
        )
        results = extract_urls(text)
        platforms = {r.platform for r in results}
        assert Platform.YOUTUBE in platforms
        assert Platform.INSTAGRAM in platforms
        assert Platform.TIKTOK in platforms

    def test_deduplicates_same_video_different_forms(self):
        text = (
            "https://youtu.be/dQw4w9WgXcQ "
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ "
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share"
        )
        results = extract_urls(text)
        assert len(results) == 1, "Same video in different URL forms should deduplicate"

    def test_empty_text_returns_empty_list(self):
        assert extract_urls("") == []

    def test_text_with_no_urls_returns_empty_list(self):
        assert extract_urls("Hello world! No links here.") == []

    def test_unsupported_platform_url_is_skipped(self):
        text = "Visit https://www.example.com/video/123 for more"
        results = extract_urls(text)
        assert results == []

    def test_non_video_url_is_skipped(self):
        text = "Visit https://www.youtube.com/channel/UC12345 for updates"
        results = extract_urls(text)
        assert results == []

    def test_trailing_punctuation_stripped(self):
        text = "Watch here: https://youtu.be/dQw4w9WgXcQ."
        results = extract_urls(text)
        assert len(results) == 1
        assert results[0].platform == Platform.YOUTUBE

    def test_trailing_comma_stripped(self):
        text = "First https://youtu.be/dQw4w9WgXcQ, then the next one"
        results = extract_urls(text)
        assert len(results) == 1

    def test_url_in_parentheses_stripped(self):
        text = "Check (https://youtu.be/dQw4w9WgXcQ)"
        results = extract_urls(text)
        assert len(results) == 1

    def test_returns_extracted_url_dataclass(self):
        text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        results = extract_urls(text)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, ExtractedURL)
        assert r.raw_url
        assert r.canonical_url
        assert r.platform == Platform.YOUTUBE
        assert len(r.url_hash) == 64  # SHA-256 hex digest

    def test_url_hash_matches_compute_url_hash(self):
        text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        results = extract_urls(text)
        r = results[0]
        assert r.url_hash == compute_url_hash(r.canonical_url)

    def test_non_string_input_raises(self):
        with pytest.raises(Exception):  # URLExtractionError
            extract_urls(12345)  # type: ignore[arg-type]

    def test_message_with_multiple_non_video_urls(self):
        text = (
            "Visit https://google.com and https://docs.python.org/3/ "
            "or https://pypi.org/project/requests/"
        )
        results = extract_urls(text)
        assert results == []

    def test_mixed_video_and_non_video_urls(self):
        text = (
            "Read https://docs.python.org/3/ and "
            "watch https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        results = extract_urls(text)
        assert len(results) == 1
        assert results[0].platform == Platform.YOUTUBE

    def test_facebook_reel_in_text(self):
        text = "Check https://www.facebook.com/reel/1234567890 it's great!"
        results = extract_urls(text)
        assert len(results) == 1
        assert results[0].platform == Platform.FACEBOOK

    def test_three_different_videos_all_extracted(self):
        text = (
            "A: https://www.youtube.com/watch?v=vid1abc1234 "
            "B: https://www.instagram.com/reel/REEL1234567/ "
            "C: https://www.facebook.com/reel/1111111111"
        )
        results = extract_urls(text)
        assert len(results) == 3

    def test_two_identical_urls_adjacent_dedup(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        text = f"{url} {url}"
        results = extract_urls(text)
        assert len(results) == 1

    def test_instagram_tv_url(self):
        text = "https://www.instagram.com/tv/ABCD12345/"
        results = extract_urls(text)
        assert len(results) == 1
        assert results[0].platform == Platform.INSTAGRAM


# ---------------------------------------------------------------------------
# extract_single_url
# ---------------------------------------------------------------------------

class TestExtractSingleUrl:
    def test_youtube_url_extracted(self):
        result = extract_single_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result.platform == Platform.YOUTUBE
        assert "v=dQw4w9WgXcQ" in result.canonical_url

    def test_url_with_trailing_whitespace_handled(self):
        result = extract_single_url("  https://youtu.be/dQw4w9WgXcQ  ")
        assert result.platform == Platform.YOUTUBE

    def test_url_with_trailing_dot_handled(self):
        result = extract_single_url("https://youtu.be/dQw4w9WgXcQ.")
        assert result.platform == Platform.YOUTUBE

    def test_unsupported_url_raises_platform_not_supported(self):
        with pytest.raises(PlatformNotSupportedError):
            extract_single_url("https://www.example.com/video/123")

    def test_bad_youtube_url_raises(self):
        with pytest.raises((PlatformNotSupportedError, URLCanonicalizationError)):
            extract_single_url("https://www.youtube.com/watch?list=PL123")

    def test_instagram_reel_url(self):
        result = extract_single_url("https://www.instagram.com/reel/CxYZ123/")
        assert result.platform == Platform.INSTAGRAM

    def test_tiktok_short_link(self):
        result = extract_single_url("https://vm.tiktok.com/ZMxxxxxx/")
        assert result.platform == Platform.TIKTOK

    def test_facebook_reel(self):
        result = extract_single_url("https://www.facebook.com/reel/1234567890")
        assert result.platform == Platform.FACEBOOK

    def test_returns_correct_hash(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = extract_single_url(url)
        _, canonical = canonicalize_url(url)
        assert result.url_hash == compute_url_hash(canonical)

    def test_hash_is_64_hex_chars(self):
        result = extract_single_url("https://youtu.be/dQw4w9WgXcQ")
        assert len(result.url_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.url_hash)


# ---------------------------------------------------------------------------
# get_enabled_platforms
# ---------------------------------------------------------------------------

class TestGetEnabledPlatforms:
    def test_returns_all_four_platforms(self):
        platforms = get_enabled_platforms()
        assert set(platforms) == {
            Platform.YOUTUBE,
            Platform.INSTAGRAM,
            Platform.TIKTOK,
            Platform.FACEBOOK,
        }

    def test_returns_list(self):
        assert isinstance(get_enabled_platforms(), list)
