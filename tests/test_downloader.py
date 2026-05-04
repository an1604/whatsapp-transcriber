"""Tests for the YtDlpDownloader — all network calls are mocked."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import DownloadError, PlatformNotSupportedError
from src.downloaders.ytdlp_downloader import YtDlpDownloader, _url_to_filename


# ---------------------------------------------------------------------------
# _url_to_filename helper
# ---------------------------------------------------------------------------

class TestUrlToFilename:
    def test_returns_16_hex_chars(self):
        name = _url_to_filename("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert len(name) == 16
        assert all(c in "0123456789abcdef" for c in name)

    def test_deterministic(self):
        url = "https://www.youtube.com/watch?v=abc"
        assert _url_to_filename(url) == _url_to_filename(url)

    def test_different_urls_different_names(self):
        a = _url_to_filename("https://youtu.be/aaa")
        b = _url_to_filename("https://youtu.be/bbb")
        assert a != b


# ---------------------------------------------------------------------------
# supports_url
# ---------------------------------------------------------------------------

class TestSupportsUrl:
    def test_youtube_supported(self):
        dl = YtDlpDownloader()
        assert dl.supports_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_instagram_supported(self):
        dl = YtDlpDownloader()
        assert dl.supports_url("https://www.instagram.com/reel/CxYZ123/") is True

    def test_tiktok_supported(self):
        dl = YtDlpDownloader()
        assert dl.supports_url("https://vm.tiktok.com/ZMxxxxxx/") is True

    def test_facebook_supported(self):
        dl = YtDlpDownloader()
        assert dl.supports_url("https://www.facebook.com/reel/1234567890") is True

    def test_unknown_url_not_supported(self):
        dl = YtDlpDownloader()
        assert dl.supports_url("https://www.example.com/video/123") is False

    def test_empty_string_not_supported(self):
        dl = YtDlpDownloader()
        assert dl.supports_url("") is False


# ---------------------------------------------------------------------------
# download_audio — output_dir validation
# ---------------------------------------------------------------------------

class TestDownloadAudioOutputDirValidation:
    async def test_nonexistent_output_dir_raises_download_error(self, tmp_path):
        dl = YtDlpDownloader()
        fake_dir = tmp_path / "nonexistent"
        with pytest.raises(DownloadError, match="does not exist"):
            await dl.download_audio("https://youtu.be/abc", fake_dir)

    async def test_file_instead_of_dir_raises_download_error(self, tmp_path):
        dl = YtDlpDownloader()
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("hello")
        with pytest.raises(DownloadError, match="not a directory"):
            await dl.download_audio("https://youtu.be/abc", not_a_dir)


# ---------------------------------------------------------------------------
# download_audio — mocked yt-dlp success path
# ---------------------------------------------------------------------------

class TestDownloadAudioSuccess:
    async def test_returns_path_of_downloaded_file(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        stem = _url_to_filename(url)
        expected_file = tmp_path / f"{stem}.mp3"

        def fake_sync(u, out_dir):
            expected_file.write_bytes(b"fake audio")
            return expected_file

        with patch.object(dl, "_download_sync", side_effect=fake_sync):
            result = await dl.download_audio(url, tmp_path)

        assert result == expected_file
        assert result.exists()

    async def test_download_sync_called_with_correct_args(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        stem = _url_to_filename(url)
        expected_file = tmp_path / f"{stem}.mp3"
        expected_file.write_bytes(b"x")

        calls = []

        def recording_sync(u, out_dir):
            calls.append((u, out_dir))
            return expected_file

        with patch.object(dl, "_download_sync", side_effect=recording_sync):
            await dl.download_audio(url, tmp_path)

        assert len(calls) == 1
        assert calls[0][0] == url
        assert calls[0][1] == tmp_path

    async def test_accepts_string_output_dir(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://youtu.be/abc"
        stem = _url_to_filename(url)
        expected_file = tmp_path / f"{stem}.mp3"
        expected_file.write_bytes(b"x")

        with patch.object(dl, "_download_sync", return_value=expected_file):
            result = await dl.download_audio(url, str(tmp_path))

        assert result == expected_file


# ---------------------------------------------------------------------------
# download_audio — mocked yt-dlp failure paths
# ---------------------------------------------------------------------------

class TestDownloadAudioFailure:
    async def test_download_error_propagated(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://youtu.be/private_video"

        with patch.object(
            dl,
            "_download_sync",
            side_effect=DownloadError("private video"),
        ):
            with pytest.raises(DownloadError, match="private video"):
                await dl.download_audio(url, tmp_path)

    async def test_unexpected_exception_wrapped_in_download_error(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://youtu.be/abc"

        with patch.object(
            dl,
            "_download_sync",
            side_effect=RuntimeError("unexpected"),
        ):
            with pytest.raises(DownloadError, match="Unexpected error"):
                await dl.download_audio(url, tmp_path)


# ---------------------------------------------------------------------------
# _download_sync — mocked yt_dlp import
# ---------------------------------------------------------------------------

class TestDownloadSync:
    def _make_ydl_mock(self, ret_code: int = 0, raise_exc=None):
        """Build a fake yt_dlp module with YoutubeDL mock."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.download = MagicMock(
            side_effect=raise_exc if raise_exc else None,
            return_value=ret_code,
        )

        mock_yt_dlp = MagicMock()
        mock_yt_dlp.YoutubeDL = MagicMock(return_value=mock_ydl_instance)

        class FakeDownloadError(Exception):
            pass

        mock_yt_dlp.utils.DownloadError = FakeDownloadError
        return mock_yt_dlp, mock_ydl_instance

    def test_success_returns_expected_file(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        stem = _url_to_filename(url)
        audio_file = tmp_path / f"{stem}.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_yt_dlp, _ = self._make_ydl_mock(ret_code=0)

        with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            result = dl._download_sync(url, tmp_path)

        assert result == audio_file

    def test_nonzero_return_raises_download_error(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://www.youtube.com/watch?v=abc"
        mock_yt_dlp, _ = self._make_ydl_mock(ret_code=1)

        with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            with pytest.raises(DownloadError, match="non-zero exit code"):
                dl._download_sync(url, tmp_path)

    def test_ytdlp_download_error_wrapped(self, tmp_path):
        dl = YtDlpDownloader()
        url = "https://www.youtube.com/watch?v=abc"

        class FakeDownloadError(Exception):
            pass

        mock_yt_dlp = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.download = MagicMock(side_effect=FakeDownloadError("403 Forbidden"))
        mock_yt_dlp.YoutubeDL = MagicMock(return_value=mock_ydl_instance)
        mock_yt_dlp.utils.DownloadError = FakeDownloadError

        with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            with pytest.raises(DownloadError, match="yt-dlp failed"):
                dl._download_sync(url, tmp_path)

    def test_missing_output_file_raises(self, tmp_path):
        """yt-dlp returns 0 but no file on disk — should raise DownloadError."""
        dl = YtDlpDownloader()
        url = "https://www.youtube.com/watch?v=abc"
        mock_yt_dlp, _ = self._make_ydl_mock(ret_code=0)
        # Note: we do NOT create any file in tmp_path

        with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            with pytest.raises(DownloadError, match="no output file found"):
                dl._download_sync(url, tmp_path)

    def test_fallback_to_alternate_extension(self, tmp_path):
        """If .mp3 is missing but another extension exists, use that."""
        dl = YtDlpDownloader()
        url = "https://www.youtube.com/watch?v=abc"
        stem = _url_to_filename(url)
        alt_file = tmp_path / f"{stem}.m4a"
        alt_file.write_bytes(b"audio")

        mock_yt_dlp, _ = self._make_ydl_mock(ret_code=0)

        with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            result = dl._download_sync(url, tmp_path)

        assert result == alt_file

    def test_yt_dlp_not_installed_raises(self, tmp_path):
        dl = YtDlpDownloader()
        with patch.dict("sys.modules", {"yt_dlp": None}):
            with pytest.raises(DownloadError, match="yt-dlp is not installed"):
                dl._download_sync("https://youtu.be/abc", tmp_path)

    def test_custom_audio_format_used(self, tmp_path):
        dl = YtDlpDownloader(audio_format="opus", audio_quality="128")
        url = "https://www.youtube.com/watch?v=abc"
        stem = _url_to_filename(url)
        audio_file = tmp_path / f"{stem}.opus"
        audio_file.write_bytes(b"audio")

        captured_opts = {}

        class CapturingYDL:
            def __init__(self, opts):
                captured_opts.update(opts)
                self._mock = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def download(self, urls):
                return 0

        mock_yt_dlp = MagicMock()
        mock_yt_dlp.YoutubeDL = CapturingYDL
        mock_yt_dlp.utils.DownloadError = Exception

        with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            result = dl._download_sync(url, tmp_path)

        pp = captured_opts["postprocessors"][0]
        assert pp["preferredcodec"] == "opus"
        assert pp["preferredquality"] == "128"
        assert result == audio_file

    def test_extra_ydl_opts_merged(self, tmp_path):
        extra = {"proxy": "http://myproxy:8080"}
        dl = YtDlpDownloader(extra_ydl_opts=extra)
        url = "https://www.youtube.com/watch?v=abc"
        stem = _url_to_filename(url)
        (tmp_path / f"{stem}.mp3").write_bytes(b"x")

        captured_opts = {}

        class CapturingYDL:
            def __init__(self, opts):
                captured_opts.update(opts)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def download(self, urls):
                return 0

        mock_yt_dlp = MagicMock()
        mock_yt_dlp.YoutubeDL = CapturingYDL
        mock_yt_dlp.utils.DownloadError = Exception

        with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            dl._download_sync(url, tmp_path)

        assert captured_opts.get("proxy") == "http://myproxy:8080"
