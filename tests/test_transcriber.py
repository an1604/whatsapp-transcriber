"""Comprehensive tests for the transcriber module (faster-whisper, mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import AudioNotFoundError, TranscriptionError
from src.transcribers.base import AbstractTranscriber, TranscriptionResult
from src.transcribers.faster_whisper import FasterWhisperTranscriber


# ---------------------------------------------------------------------------
# TranscriptionResult dataclass
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def test_basic_construction(self):
        r = TranscriptionResult(
            text="Hello world",
            detected_language="en",
            segments=[{"start": 0.0, "end": 1.5, "text": "Hello world"}],
        )
        assert r.text == "Hello world"
        assert r.detected_language == "en"
        assert len(r.segments) == 1

    def test_frozen(self):
        r = TranscriptionResult(text="x", detected_language="en", segments=[])
        with pytest.raises(Exception):  # FrozenInstanceError
            r.text = "y"  # type: ignore[misc]

    def test_empty_segments_allowed(self):
        r = TranscriptionResult(text="", detected_language="he", segments=[])
        assert r.segments == []


# ---------------------------------------------------------------------------
# AbstractTranscriber is abstract
# ---------------------------------------------------------------------------

class TestAbstractTranscriber:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AbstractTranscriber()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_transcribe(self):
        class Partial(AbstractTranscriber):
            pass

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_valid_concrete_subclass(self):
        class Concrete(AbstractTranscriber):
            async def transcribe(self, audio_path: Path) -> TranscriptionResult:
                return TranscriptionResult(text="", detected_language="en", segments=[])

        inst = Concrete()
        assert isinstance(inst, AbstractTranscriber)


# ---------------------------------------------------------------------------
# FasterWhisperTranscriber — constructor validation
# ---------------------------------------------------------------------------

class TestFasterWhisperConstructor:
    def test_defaults_accepted(self):
        t = FasterWhisperTranscriber()
        assert t._model_size == "small"
        assert t._device == "cpu"
        assert t._compute_type == "int8"
        assert t._language is None  # "auto" → None

    def test_explicit_language_stored(self):
        t = FasterWhisperTranscriber(language="he")
        assert t._language == "he"

    def test_auto_language_stored_as_none(self):
        t = FasterWhisperTranscriber(language="auto")
        assert t._language is None

    def test_invalid_model_size_raises(self):
        with pytest.raises(TranscriptionError, match="model_size"):
            FasterWhisperTranscriber(model_size="superduper")

    def test_invalid_device_raises(self):
        with pytest.raises(TranscriptionError, match="device"):
            FasterWhisperTranscriber(device="tpu")

    def test_invalid_compute_type_raises(self):
        with pytest.raises(TranscriptionError, match="compute_type"):
            FasterWhisperTranscriber(compute_type="float128")

    @pytest.mark.parametrize("size", [
        "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo",
        "distil-large-v3",
    ])
    def test_all_valid_model_sizes_accepted(self, size):
        t = FasterWhisperTranscriber(model_size=size)
        assert t._model_size == size

    @pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
    def test_all_valid_devices_accepted(self, device):
        t = FasterWhisperTranscriber(device=device)
        assert t._device == device

    @pytest.mark.parametrize("ct", ["int8", "float16", "float32", "bfloat16"])
    def test_valid_compute_types_accepted(self, ct):
        t = FasterWhisperTranscriber(compute_type=ct)
        assert t._compute_type == ct

    def test_model_not_loaded_at_init(self):
        t = FasterWhisperTranscriber()
        assert t._model is None


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_model_loaded_on_first_call(self):
        t = FasterWhisperTranscriber()
        mock_whisper_model = MagicMock()
        mock_module = MagicMock()
        mock_module.WhisperModel = MagicMock(return_value=mock_whisper_model)

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            t._load_model()

        assert t._model is mock_whisper_model
        mock_module.WhisperModel.assert_called_once_with(
            "small", device="cpu", compute_type="int8"
        )

    def test_model_loaded_only_once(self):
        t = FasterWhisperTranscriber()
        mock_module = MagicMock()
        mock_module.WhisperModel = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            t._load_model()
            t._load_model()

        mock_module.WhisperModel.assert_called_once()

    def test_faster_whisper_not_installed_raises(self):
        t = FasterWhisperTranscriber()
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with pytest.raises(TranscriptionError, match="not installed"):
                t._load_model()

    def test_model_constructor_exception_raises_transcription_error(self):
        t = FasterWhisperTranscriber()
        mock_module = MagicMock()
        mock_module.WhisperModel = MagicMock(side_effect=RuntimeError("OOM"))

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            with pytest.raises(TranscriptionError, match="Failed to load"):
                t._load_model()

    def test_after_failed_load_model_is_none(self):
        t = FasterWhisperTranscriber()
        mock_module = MagicMock()
        mock_module.WhisperModel = MagicMock(side_effect=RuntimeError("fail"))

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            try:
                t._load_model()
            except TranscriptionError:
                pass
        assert t._model is None


# ---------------------------------------------------------------------------
# _transcribe_sync — mocked model
# ---------------------------------------------------------------------------

def _make_segment(start: float, end: float, text: str):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def _make_info(language: str = "en"):
    info = MagicMock()
    info.language = language
    return info


class TestTranscribeSync:
    def _transcriber_with_mock_model(self, segments, language="en"):
        """Return a transcriber whose model is already set to a mock."""
        t = FasterWhisperTranscriber()
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(
            return_value=(iter(segments), _make_info(language))
        )
        t._model = mock_model
        return t

    def test_basic_transcription(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        segs = [_make_segment(0.0, 1.0, " Hello"), _make_segment(1.0, 2.0, " world")]
        t = self._transcriber_with_mock_model(segs, "en")

        result = t._transcribe_sync(audio)

        assert result.text == "Hello world"
        assert result.detected_language == "en"
        assert len(result.segments) == 2

    def test_detected_language_from_info(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = self._transcriber_with_mock_model([_make_segment(0, 1, "שלום")], "he")

        result = t._transcribe_sync(audio)
        assert result.detected_language == "he"

    def test_fixed_language_overrides_info(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber(language="he")
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(
            return_value=(iter([_make_segment(0, 1, "hello")]), _make_info("en"))
        )
        t._model = mock_model

        result = t._transcribe_sync(audio)
        assert result.detected_language == "he"

    def test_segments_have_correct_structure(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        segs = [_make_segment(0.5, 1.5, " test text")]
        t = self._transcriber_with_mock_model(segs, "en")

        result = t._transcribe_sync(audio)
        assert result.segments[0] == {"start": 0.5, "end": 1.5, "text": "test text"}

    def test_empty_audio_no_segments(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = self._transcriber_with_mock_model([], "en")

        result = t._transcribe_sync(audio)
        assert result.text == ""
        assert result.segments == []

    def test_transcribe_exception_raises_transcription_error(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber()
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=RuntimeError("decode error"))
        t._model = mock_model

        with pytest.raises(TranscriptionError, match="faster-whisper failed"):
            t._transcribe_sync(audio)

    def test_vad_filter_passed_to_model(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber(vad_filter=False)
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(
            return_value=(iter([]), _make_info("en"))
        )
        t._model = mock_model

        t._transcribe_sync(audio)
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("vad_filter") is False

    def test_beam_size_passed_to_model(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber(beam_size=3)
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(
            return_value=(iter([]), _make_info("en"))
        )
        t._model = mock_model

        t._transcribe_sync(audio)
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("beam_size") == 3

    def test_language_none_passed_to_model_for_auto(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber(language="auto")
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(
            return_value=(iter([]), _make_info("fr"))
        )
        t._model = mock_model

        t._transcribe_sync(audio)
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("language") is None


# ---------------------------------------------------------------------------
# transcribe (async) — file validation + thread executor
# ---------------------------------------------------------------------------

class TestTranscribeAsync:
    async def test_missing_file_raises_audio_not_found(self, tmp_path):
        t = FasterWhisperTranscriber()
        missing = tmp_path / "nope.mp3"

        with pytest.raises(AudioNotFoundError, match="not found"):
            await t.transcribe(missing)

    async def test_directory_instead_of_file_raises(self, tmp_path):
        t = FasterWhisperTranscriber()
        with pytest.raises(AudioNotFoundError, match="not a file"):
            await t.transcribe(tmp_path)

    async def test_successful_transcription_returns_result(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")

        t = FasterWhisperTranscriber()
        expected = TranscriptionResult(
            text="Hello world",
            detected_language="en",
            segments=[{"start": 0.0, "end": 1.5, "text": "Hello world"}],
        )
        with patch.object(t, "_transcribe_sync", return_value=expected):
            result = await t.transcribe(audio)

        assert result.text == "Hello world"
        assert result.detected_language == "en"

    async def test_transcription_error_propagated(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber()

        with patch.object(
            t, "_transcribe_sync", side_effect=TranscriptionError("bad audio")
        ):
            with pytest.raises(TranscriptionError, match="bad audio"):
                await t.transcribe(audio)

    async def test_unexpected_exception_wrapped(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber()

        with patch.object(
            t, "_transcribe_sync", side_effect=MemoryError("OOM")
        ):
            with pytest.raises(TranscriptionError, match="Unexpected error"):
                await t.transcribe(audio)

    async def test_accepts_string_path(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber()
        expected = TranscriptionResult(text="hi", detected_language="en", segments=[])

        with patch.object(t, "_transcribe_sync", return_value=expected):
            result = await t.transcribe(str(audio))  # type: ignore[arg-type]

        assert result.text == "hi"

    async def test_transcribe_sync_called_with_path_object(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        t = FasterWhisperTranscriber()
        expected = TranscriptionResult(text="x", detected_language="en", segments=[])
        calls = []

        def recording_sync(p):
            calls.append(p)
            return expected

        with patch.object(t, "_transcribe_sync", side_effect=recording_sync):
            await t.transcribe(audio)

        assert len(calls) == 1
        assert isinstance(calls[0], Path)
