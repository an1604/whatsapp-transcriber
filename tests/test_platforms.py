"""Tests for per-platform URL pattern matching and canonicalization."""
from __future__ import annotations

import pytest

from src.downloaders.platforms import facebook, instagram, tiktok, youtube


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

class TestYouTubePatterns:
    """Pattern matching — does the URL match at all."""

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/abc123",
        "https://youtube.com/shorts/abc123",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/v/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=abc&list=PL123&index=2",
    ])
    def test_patterns_match(self, url):
        assert any(p.search(url) for p in youtube.PATTERNS), f"No pattern matched: {url}"

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/channel/UC12345",
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/@username",
        "https://notytube.com/watch?v=abc",
        "https://www.facebook.com/watch?v=123",
    ])
    def test_patterns_do_not_match_non_video(self, url):
        assert not any(p.search(url) for p in youtube.PATTERNS), f"Pattern wrongly matched: {url}"


class TestYouTubeCanonicalize:
    def test_watch_url_strips_tracking(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&si=abc"
        result = youtube.canonicalize(url)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_youtu_be_shortlink(self):
        result = youtube.canonicalize("https://youtu.be/dQw4w9WgXcQ")
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_youtu_be_with_query_stripped(self):
        result = youtube.canonicalize("https://youtu.be/dQw4w9WgXcQ?si=XXXX&feature=shared")
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_shorts_url(self):
        result = youtube.canonicalize("https://www.youtube.com/shorts/abc123")
        assert result == "https://www.youtube.com/shorts/abc123"

    def test_shorts_strips_query(self):
        result = youtube.canonicalize("https://www.youtube.com/shorts/abc123?feature=share")
        assert result == "https://www.youtube.com/shorts/abc123"

    def test_embed_url(self):
        result = youtube.canonicalize("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_v_url(self):
        result = youtube.canonicalize("https://www.youtube.com/v/dQw4w9WgXcQ")
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_mobile_url(self):
        result = youtube.canonicalize("https://m.youtube.com/watch?v=dQw4w9WgXcQ&si=XXXX")
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_keeps_playlist_params(self):
        result = youtube.canonicalize(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&index=2&feature=share"
        )
        assert "v=dQw4w9WgXcQ" in result
        assert "list=PL123" in result
        assert "index=2" in result
        assert "feature" not in result

    def test_youtu_be_empty_slug_raises(self):
        with pytest.raises(ValueError, match="youtu.be"):
            youtube.canonicalize("https://youtu.be/")

    def test_shorts_empty_id_raises(self):
        with pytest.raises(ValueError):
            youtube.canonicalize("https://www.youtube.com/shorts/")

    def test_watch_missing_v_param_raises(self):
        with pytest.raises(ValueError, match="v="):
            youtube.canonicalize("https://www.youtube.com/watch?list=PL123")

    def test_unrecognized_youtube_url_raises(self):
        with pytest.raises(ValueError):
            youtube.canonicalize("https://www.youtube.com/channel/UC12345")

    def test_canonical_is_idempotent(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert youtube.canonicalize(url) == youtube.canonicalize(youtube.canonicalize(url))


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------

class TestInstagramPatterns:
    @pytest.mark.parametrize("url", [
        "https://www.instagram.com/reel/CxYZ123/",
        "https://instagram.com/reel/CxYZ123/",
        "https://www.instagram.com/p/CxYZ123/",
        "https://www.instagram.com/tv/CxYZ123/",
        "https://www.instagram.com/reels/CxYZ123/",
    ])
    def test_patterns_match(self, url):
        assert any(p.search(url) for p in instagram.PATTERNS), f"No pattern matched: {url}"

    @pytest.mark.parametrize("url", [
        "https://www.instagram.com/username/",
        "https://www.instagram.com/explore/tags/python/",
        "https://www.facebook.com/reel/123",
    ])
    def test_patterns_do_not_match_non_video(self, url):
        assert not any(p.search(url) for p in instagram.PATTERNS), f"Pattern wrongly matched: {url}"


class TestInstagramCanonicalize:
    def test_reel_url(self):
        result = instagram.canonicalize("https://www.instagram.com/reel/CxYZ123/")
        assert result == "https://www.instagram.com/reel/CxYZ123/"

    def test_reel_strips_query(self):
        result = instagram.canonicalize(
            "https://www.instagram.com/reel/CxYZ123/?igsh=abc&utm_source=ig"
        )
        assert result == "https://www.instagram.com/reel/CxYZ123/"

    def test_p_url(self):
        result = instagram.canonicalize("https://www.instagram.com/p/CxYZ123/")
        assert result == "https://www.instagram.com/p/CxYZ123/"

    def test_tv_url(self):
        result = instagram.canonicalize("https://www.instagram.com/tv/CxYZ123/")
        assert result == "https://www.instagram.com/tv/CxYZ123/"

    def test_reels_normalized_to_reel(self):
        result = instagram.canonicalize("https://www.instagram.com/reels/CxYZ123/")
        assert result == "https://www.instagram.com/reel/CxYZ123/"

    def test_profile_url_raises(self):
        with pytest.raises(ValueError):
            instagram.canonicalize("https://www.instagram.com/username/")

    def test_explore_url_raises(self):
        with pytest.raises(ValueError):
            instagram.canonicalize("https://www.instagram.com/explore/tags/python/")

    def test_too_few_path_parts_raises(self):
        with pytest.raises(ValueError, match="too few path"):
            instagram.canonicalize("https://www.instagram.com/")

    def test_canonical_is_idempotent(self):
        url = "https://www.instagram.com/reel/CxYZ123/"
        assert instagram.canonicalize(url) == instagram.canonicalize(instagram.canonicalize(url))


# ---------------------------------------------------------------------------
# TikTok
# ---------------------------------------------------------------------------

class TestTikTokPatterns:
    @pytest.mark.parametrize("url", [
        "https://www.tiktok.com/@username/video/1234567890",
        "https://vm.tiktok.com/ZMxxxxxx/",
        "https://vt.tiktok.com/ZMxxxxxx/",
        "https://www.tiktok.com/t/ZMxxxxxx/",
        "https://m.tiktok.com/v/1234567890",
    ])
    def test_patterns_match(self, url):
        assert any(p.search(url) for p in tiktok.PATTERNS), f"No pattern matched: {url}"

    @pytest.mark.parametrize("url", [
        "https://www.tiktok.com/@username",
        "https://www.tiktok.com/foryou",
        "https://www.facebook.com/reel/123",
    ])
    def test_patterns_do_not_match_non_video(self, url):
        assert not any(p.search(url) for p in tiktok.PATTERNS), f"Pattern wrongly matched: {url}"


class TestTikTokCanonicalize:
    def test_long_form_url(self):
        result = tiktok.canonicalize("https://www.tiktok.com/@user123/video/9876543210")
        assert result == "https://www.tiktok.com/@user123/video/9876543210"

    def test_long_form_strips_query(self):
        result = tiktok.canonicalize(
            "https://www.tiktok.com/@user123/video/9876543210?is_from_webapp=1"
        )
        assert result == "https://www.tiktok.com/@user123/video/9876543210"

    def test_vm_short_link_kept(self):
        result = tiktok.canonicalize("https://vm.tiktok.com/ZMxxxxxx/")
        assert result == "https://vm.tiktok.com/ZMxxxxxx"

    def test_vt_short_link_kept(self):
        result = tiktok.canonicalize("https://vt.tiktok.com/ZMxxxxxx/")
        assert result == "https://vt.tiktok.com/ZMxxxxxx"

    def test_t_share_link(self):
        result = tiktok.canonicalize("https://www.tiktok.com/t/ZMxxxxxx/")
        assert result == "https://www.tiktok.com/t/ZMxxxxxx"

    def test_mobile_v_link(self):
        result = tiktok.canonicalize("https://m.tiktok.com/v/1234567890?lang=en")
        assert result == "https://www.tiktok.com/video/1234567890"

    def test_vm_empty_slug_raises(self):
        with pytest.raises(ValueError, match="no slug"):
            tiktok.canonicalize("https://vm.tiktok.com/")

    def test_unrecognized_tiktok_url_raises(self):
        with pytest.raises(ValueError, match="TikTok"):
            tiktok.canonicalize("https://www.tiktok.com/@username")

    def test_canonical_is_idempotent(self):
        url = "https://www.tiktok.com/@user/video/123456"
        assert tiktok.canonicalize(url) == tiktok.canonicalize(tiktok.canonicalize(url))


# ---------------------------------------------------------------------------
# Facebook
# ---------------------------------------------------------------------------

class TestFacebookPatterns:
    @pytest.mark.parametrize("url", [
        "https://www.facebook.com/reel/1234567890",
        "https://www.facebook.com/watch?v=1234567890",
        "https://www.facebook.com/watch/",
        "https://fb.watch/abcdef123/",
        "https://www.facebook.com/username/videos/1234567890",
        "https://www.facebook.com/video.php?v=1234567890",
        "https://www.facebook.com/share/v/abcdef/",
        "https://www.facebook.com/share/r/abcdef/",
    ])
    def test_patterns_match(self, url):
        assert any(p.search(url) for p in facebook.PATTERNS), f"No pattern matched: {url}"

    @pytest.mark.parametrize("url", [
        "https://www.facebook.com/username/posts/1234",
        "https://www.facebook.com/groups/mygroup",
        "https://www.instagram.com/reel/abc",
    ])
    def test_patterns_do_not_match_non_video(self, url):
        assert not any(p.search(url) for p in facebook.PATTERNS), f"Pattern wrongly matched: {url}"


class TestFacebookCanonicalize:
    def test_reel_url(self):
        result = facebook.canonicalize("https://www.facebook.com/reel/1234567890")
        assert result == "https://www.facebook.com/reel/1234567890"

    def test_reel_strips_query(self):
        result = facebook.canonicalize(
            "https://www.facebook.com/reel/1234567890?__cft__=abc"
        )
        assert result == "https://www.facebook.com/reel/1234567890"

    def test_watch_with_v_param(self):
        result = facebook.canonicalize("https://www.facebook.com/watch?v=1234567890")
        assert result == "https://www.facebook.com/watch?v=1234567890"

    def test_watch_without_v_param(self):
        result = facebook.canonicalize("https://www.facebook.com/watch/")
        assert result == "https://www.facebook.com/watch/"

    def test_fb_watch_short_link(self):
        result = facebook.canonicalize("https://fb.watch/abcdef123/")
        assert result == "https://fb.watch/abcdef123"

    def test_fb_watch_empty_slug_raises(self):
        with pytest.raises(ValueError, match="no slug"):
            facebook.canonicalize("https://fb.watch/")

    def test_username_videos(self):
        result = facebook.canonicalize(
            "https://www.facebook.com/someuser/videos/1234567890"
        )
        assert result == "https://www.facebook.com/someuser/videos/1234567890"

    def test_video_php(self):
        result = facebook.canonicalize(
            "https://www.facebook.com/video.php?v=1234567890&__cft__=abc"
        )
        assert result == "https://www.facebook.com/video.php?v=1234567890"

    def test_share_v(self):
        result = facebook.canonicalize("https://www.facebook.com/share/v/abcdef/")
        assert result == "https://www.facebook.com/share/v/abcdef"

    def test_share_r(self):
        result = facebook.canonicalize("https://www.facebook.com/share/r/abcdef/")
        assert result == "https://www.facebook.com/share/r/abcdef"

    def test_unrecognized_facebook_url_raises(self):
        with pytest.raises(ValueError, match="Facebook"):
            facebook.canonicalize("https://www.facebook.com/someuser/posts/1234")

    def test_canonical_is_idempotent(self):
        url = "https://www.facebook.com/reel/1234567890"
        assert facebook.canonicalize(url) == facebook.canonicalize(facebook.canonicalize(url))
