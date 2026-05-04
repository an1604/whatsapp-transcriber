from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from src.core.exceptions import DownloadError, PlatformNotSupportedError
from src.downloaders.base import AbstractDownloader
from src.downloaders.registry import detect_platform

logger = logging.getLogger(__name__)

# Audio format preference: best quality opus/m4a/mp3, no video
_YTDLP_FORMAT = "bestaudio/best"

_DEFAULT_POSTPROCESSORS = [
    {
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "192",
    }
]


def _url_to_filename(url: str) -> str:
    """Derive a stable filename stem from a URL hash (first 16 hex chars)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


class YtDlpDownloader(AbstractDownloader):
    """Downloads audio using yt-dlp as a Python library (not subprocess).

    audio_format: output codec passed to FFmpegExtractAudio postprocessor.
    audio_quality: bitrate string, e.g. "192".
    extra_ydl_opts: any additional yt-dlp options to merge in.

    Raises DownloadError on any failure. Never returns None.
    """

    def __init__(
        self,
        audio_format: str = "mp3",
        audio_quality: str = "192",
        extra_ydl_opts: dict | None = None,
    ) -> None:
        self._audio_format = audio_format
        self._audio_quality = audio_quality
        self._extra_ydl_opts = extra_ydl_opts or {}

    def supports_url(self, url: str) -> bool:
        """Return True if yt-dlp can handle this URL (platform is registered)."""
        try:
            detect_platform(url)
            return True
        except PlatformNotSupportedError:
            return False

    async def download_audio(self, url: str, output_dir: Path) -> Path:
        """Download the audio for url into output_dir and return the file path.

        Runs the blocking yt-dlp call in a thread pool executor so the event
        loop is not blocked during the download.

        Raises:
            DownloadError: on any yt-dlp or I/O failure.
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise DownloadError(
                f"output_dir does not exist: {output_dir!r}. "
                "Create it before calling download_audio."
            )
        if not output_dir.is_dir():
            raise DownloadError(
                f"output_dir is not a directory: {output_dir!r}"
            )

        loop = asyncio.get_event_loop()
        try:
            audio_path = await loop.run_in_executor(
                None, self._download_sync, url, output_dir
            )
        except DownloadError:
            raise
        except Exception as exc:
            raise DownloadError(
                f"Unexpected error downloading {url!r}: {exc}"
            ) from exc

        return audio_path

    def _download_sync(self, url: str, output_dir: Path) -> Path:
        """Blocking yt-dlp download. Called from a thread executor."""
        try:
            import yt_dlp  # noqa: PLC0415
        except ImportError as exc:
            raise DownloadError(
                "yt-dlp is not installed. Run: pip install yt-dlp"
            ) from exc

        stem = _url_to_filename(url)
        outtmpl = str(output_dir / f"{stem}.%(ext)s")

        ydl_opts: dict = {
            "format": _YTDLP_FORMAT,
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self._audio_format,
                    "preferredquality": self._audio_quality,
                }
            ],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        ydl_opts.update(self._extra_ydl_opts)

        logger.info("Starting download: %s → %s", url, output_dir)

        errors: list[str] = []

        class _ErrorLogger:
            def debug(self, msg: str) -> None:
                pass

            def warning(self, msg: str) -> None:
                pass

            def error(self, msg: str) -> None:
                errors.append(msg)

        ydl_opts["logger"] = _ErrorLogger()

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ret = ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:
            raise DownloadError(
                f"yt-dlp failed to download {url!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise DownloadError(
                f"Unexpected yt-dlp error for {url!r}: {exc}"
            ) from exc

        if ret != 0:
            detail = "; ".join(errors) if errors else "unknown reason"
            raise DownloadError(
                f"yt-dlp returned non-zero exit code {ret} for {url!r}: {detail}"
            )

        expected = output_dir / f"{stem}.{self._audio_format}"
        if not expected.exists():
            # yt-dlp may have used a different extension
            matches = list(output_dir.glob(f"{stem}.*"))
            if not matches:
                raise DownloadError(
                    f"yt-dlp reported success but no output file found "
                    f"for stem {stem!r} in {output_dir!r}"
                )
            expected = matches[0]

        logger.info("Download complete: %s", expected)
        return expected
