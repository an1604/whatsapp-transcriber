from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AbstractDownloader(ABC):
    """Interface for all audio downloaders.

    Each implementation downloads the audio track of a video from a URL
    and writes it to a file inside output_dir. The caller is responsible
    for managing the lifecycle of that file.

    Implementations MUST raise DownloadError (from src.core.exceptions) on
    any failure — no silent fallbacks, no returning None.
    """

    @abstractmethod
    async def download_audio(self, url: str, output_dir: Path) -> Path:
        """Download the audio of the video at url into output_dir.

        Returns the absolute Path of the downloaded audio file.

        Raises:
            DownloadError: if the download fails for any reason.
            PlatformNotSupportedError: if the URL is not supported.
        """

    @abstractmethod
    def supports_url(self, url: str) -> bool:
        """Return True if this downloader can handle the given URL."""
