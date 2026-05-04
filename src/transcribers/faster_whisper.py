from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.core.exceptions import AudioNotFoundError, TranscriptionError
from src.transcribers.base import AbstractTranscriber, TranscriptionResult

logger = logging.getLogger(__name__)

_VALID_MODEL_SIZES = frozenset(
    {"tiny", "tiny.en", "base", "base.en", "small", "small.en",
     "medium", "medium.en", "large-v1", "large-v2", "large-v3",
     "large-v3-turbo", "distil-small.en", "distil-medium.en",
     "distil-large-v2", "distil-large-v3"}
)
_VALID_DEVICES = frozenset({"cpu", "cuda", "auto"})
_VALID_COMPUTE_TYPES = frozenset(
    {"int8", "int8_float16", "int8_bfloat16", "int16",
     "float16", "bfloat16", "float32", "default"}
)


class FasterWhisperTranscriber(AbstractTranscriber):
    """Transcribes audio using faster-whisper (CTranslate2 backend).

    Designed for CPU inference with quantized compute types (int8 recommended).
    Language detection is automatic when language='auto'.

    Raises TranscriptionError on any failure — never returns partial results.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "auto",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        if model_size not in _VALID_MODEL_SIZES:
            raise TranscriptionError(
                f"Unknown model_size={model_size!r}. "
                f"Valid options: {sorted(_VALID_MODEL_SIZES)}"
            )
        if device not in _VALID_DEVICES:
            raise TranscriptionError(
                f"Unknown device={device!r}. Valid options: {sorted(_VALID_DEVICES)}"
            )
        if compute_type not in _VALID_COMPUTE_TYPES:
            raise TranscriptionError(
                f"Unknown compute_type={compute_type!r}. "
                f"Valid options: {sorted(_VALID_COMPUTE_TYPES)}"
            )

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language: str | None = None if language == "auto" else language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._model = None  # lazy-loaded on first use

    def _load_model(self):
        """Load the faster-whisper model (blocking). Called lazily."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc

        logger.info(
            "Loading faster-whisper model %s on %s (%s)",
            self._model_size,
            self._device,
            self._compute_type,
        )
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load faster-whisper model {self._model_size!r}: {exc}"
            ) from exc

    def _transcribe_sync(self, audio_path: Path) -> TranscriptionResult:
        """Blocking transcription call — runs in executor."""
        self._load_model()

        logger.info("Transcribing: %s", audio_path)
        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                beam_size=self._beam_size,
                language=self._language,
                vad_filter=self._vad_filter,
            )
            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })
                full_text_parts.append(seg.text)
        except Exception as exc:
            raise TranscriptionError(
                f"faster-whisper failed to transcribe {audio_path!r}: {exc}"
            ) from exc

        detected_language = info.language if self._language is None else self._language
        full_text = " ".join(part.strip() for part in full_text_parts).strip()

        logger.info(
            "Transcription done: lang=%s, %d segments, %d chars",
            detected_language,
            len(segments),
            len(full_text),
        )
        return TranscriptionResult(
            text=full_text,
            detected_language=detected_language,
            segments=segments,
        )

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Async wrapper — runs the blocking transcription in a thread pool.

        Raises:
            AudioNotFoundError: if the file does not exist.
            TranscriptionError: on any transcription failure.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioNotFoundError(
                f"Audio file not found: {audio_path!r}"
            )
        if not audio_path.is_file():
            raise AudioNotFoundError(
                f"Audio path is not a file: {audio_path!r}"
            )

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, self._transcribe_sync, audio_path
            )
        except (TranscriptionError, AudioNotFoundError):
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Unexpected error transcribing {audio_path!r}: {exc}"
            ) from exc

        return result
