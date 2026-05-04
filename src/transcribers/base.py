from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptionResult:
    """Output of a transcription run."""

    text: str
    detected_language: str
    segments: list[dict]


class AbstractTranscriber(ABC):
    """Interface for all transcription backends.

    Implementations MUST raise TranscriptionError (from src.core.exceptions)
    on any failure — no silent fallbacks, no returning partial results.
    """

    @abstractmethod
    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe the audio file at audio_path.

        Returns a TranscriptionResult with text, language, and segments.

        Raises:
            TranscriptionError: on any transcription failure.
            AudioNotFoundError: if audio_path does not exist on disk.
        """
