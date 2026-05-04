"""Tests for the summarizer module (prompt_loader + LiteLLMSummarizer, fully mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ConfigurationError, SummarizationError
from src.summarizers.base import AbstractSummarizer, SummaryResult
from src.summarizers.litellm_summarizer import LiteLLMSummarizer
from src.summarizers.prompt_loader import load_prompt, render_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_prompt(path: Path, content: str = "Summarize {transcript} in {language}.") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _make_llm_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# SummaryResult dataclass
# ---------------------------------------------------------------------------

class TestSummaryResult:
    def test_basic_construction(self):
        r = SummaryResult(
            summary="Key point A. Key point B.",
            model="ollama/llama3.1:8b",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert r.summary == "Key point A. Key point B."
        assert r.model == "ollama/llama3.1:8b"
        assert r.prompt_tokens == 100
        assert r.completion_tokens == 50

    def test_frozen(self):
        r = SummaryResult(summary="x", model="m", prompt_tokens=0, completion_tokens=0)
        with pytest.raises(Exception):
            r.summary = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AbstractSummarizer is abstract
# ---------------------------------------------------------------------------

class TestAbstractSummarizer:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractSummarizer()  # type: ignore[abstract]

    def test_concrete_subclass_valid(self):
        class Concrete(AbstractSummarizer):
            async def summarize(self, transcript: str, language: str) -> SummaryResult:
                return SummaryResult(summary="", model="m", prompt_tokens=0, completion_tokens=0)

        assert isinstance(Concrete(), AbstractSummarizer)


# ---------------------------------------------------------------------------
# load_prompt
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_loads_valid_file(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt", "Hello {transcript}")
        assert load_prompt(f) == "Hello {transcript}"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not found"):
            load_prompt(tmp_path / "missing.txt")

    def test_directory_path_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not a file"):
            load_prompt(tmp_path)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="empty"):
            load_prompt(f)

    def test_whitespace_only_file_raises(self, tmp_path):
        f = tmp_path / "ws.txt"
        f.write_text("   \n\t  ", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="empty"):
            load_prompt(f)

    def test_strips_leading_trailing_whitespace(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("  hello {transcript}  \n", encoding="utf-8")
        assert load_prompt(f) == "hello {transcript}"

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("ok {transcript}", encoding="utf-8")
        result = load_prompt(str(f))
        assert result == "ok {transcript}"

    def test_reads_utf8_content(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("סכם את {transcript}", encoding="utf-8")
        assert load_prompt(f) == "סכם את {transcript}"


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------

class TestRenderPrompt:
    def test_substitutes_transcript(self):
        result = render_prompt("Say: {transcript}", "hello world", "en")
        assert result == "Say: hello world"

    def test_substitutes_language(self):
        result = render_prompt("Answer in {language}.", "text", "he")
        assert "in he." in result

    def test_substitutes_both(self):
        result = render_prompt(
            "Summarize {transcript} in {language}.", "my text", "en"
        )
        assert result == "Summarize my text in en."

    def test_no_placeholders_raises(self):
        with pytest.raises(ConfigurationError, match="neither"):
            render_prompt("No placeholders here.", "text", "en")

    def test_only_transcript_placeholder_ok(self):
        result = render_prompt("Summarize: {transcript}", "text", "en")
        assert "text" in result

    def test_only_language_placeholder_ok(self):
        result = render_prompt("Reply in {language}.", "anything", "fr")
        assert "fr" in result

    def test_empty_transcript_substituted(self):
        result = render_prompt("Summarize {transcript}!", "", "en")
        assert result == "Summarize !"

    def test_multiple_occurrences_both_replaced(self):
        result = render_prompt(
            "{transcript} and {transcript} in {language}", "X", "en"
        )
        assert result == "X and X in en"


# ---------------------------------------------------------------------------
# LiteLLMSummarizer — constructor validation
# ---------------------------------------------------------------------------

class TestLiteLLMSummarizerConstructor:
    def test_valid_construction(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="ollama/llama3.1:8b", prompt_path=f)
        assert s._model == "ollama/llama3.1:8b"
        assert s._temperature == 0.3

    def test_blank_model_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        with pytest.raises(SummarizationError, match="model"):
            LiteLLMSummarizer(model="  ", prompt_path=f)

    def test_temperature_below_zero_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        with pytest.raises(SummarizationError, match="temperature"):
            LiteLLMSummarizer(model="m", prompt_path=f, temperature=-0.1)

    def test_temperature_above_two_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        with pytest.raises(SummarizationError, match="temperature"):
            LiteLLMSummarizer(model="m", prompt_path=f, temperature=2.1)

    def test_zero_max_tokens_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        with pytest.raises(SummarizationError, match="max_tokens"):
            LiteLLMSummarizer(model="m", prompt_path=f, max_tokens=0)

    def test_negative_max_tokens_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        with pytest.raises(SummarizationError, match="max_tokens"):
            LiteLLMSummarizer(model="m", prompt_path=f, max_tokens=-1)

    def test_zero_timeout_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        with pytest.raises(SummarizationError, match="timeout"):
            LiteLLMSummarizer(model="m", prompt_path=f, timeout=0)

    def test_missing_prompt_file_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not found"):
            LiteLLMSummarizer(model="m", prompt_path=tmp_path / "missing.txt")

    def test_prompt_loaded_at_construction(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt", "Say {transcript} in {language}.")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        assert "{transcript}" in s._prompt_template

    def test_temperature_zero_accepted(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, temperature=0.0)
        assert s._temperature == 0.0

    def test_temperature_two_accepted(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, temperature=2.0)
        assert s._temperature == 2.0

    def test_none_max_tokens_accepted(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, max_tokens=None)
        assert s._max_tokens is None

    def test_api_base_stored(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, api_base="http://localhost:11434")
        assert s._api_base == "http://localhost:11434"


# ---------------------------------------------------------------------------
# LiteLLMSummarizer.summarize — input validation
# ---------------------------------------------------------------------------

class TestSummarizeInputValidation:
    async def test_empty_transcript_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        with pytest.raises(SummarizationError, match="empty"):
            await s.summarize("", "en")

    async def test_whitespace_only_transcript_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        with pytest.raises(SummarizationError, match="empty"):
            await s.summarize("   \n  ", "en")

    async def test_empty_language_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        with pytest.raises(SummarizationError, match="language"):
            await s.summarize("some text", "")

    async def test_whitespace_language_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        with pytest.raises(SummarizationError, match="language"):
            await s.summarize("some text", "  ")


# ---------------------------------------------------------------------------
# LiteLLMSummarizer.summarize — mocked LiteLLM calls
# ---------------------------------------------------------------------------

class TestSummarizeSuccess:
    async def test_returns_summary_result(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="ollama/llama3.1:8b", prompt_path=f)
        response = _make_llm_response("Key points here.", 80, 20)

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=response)
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = await s.summarize("Full transcript text here.", "en")

        assert isinstance(result, SummaryResult)
        assert result.summary == "Key points here."
        assert result.model == "ollama/llama3.1:8b"
        assert result.prompt_tokens == 80
        assert result.completion_tokens == 20

    async def test_strips_whitespace_from_summary(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        response = _make_llm_response("  padded summary  \n")

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=response)
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = await s.summarize("text", "en")

        assert result.summary == "padded summary"

    async def test_api_base_passed_to_litellm(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(
            model="m", prompt_path=f, api_base="http://localhost:11434"
        )
        response = _make_llm_response("ok")
        captured_kwargs: dict = {}

        async def capturing_completion(**kwargs):
            captured_kwargs.update(kwargs)
            return response

        mock_litellm = MagicMock()
        mock_litellm.acompletion = capturing_completion
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await s.summarize("text", "en")

        assert captured_kwargs.get("api_base") == "http://localhost:11434"

    async def test_max_tokens_passed_when_set(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, max_tokens=256)
        response = _make_llm_response("ok")
        captured_kwargs: dict = {}

        async def capturing(**kwargs):
            captured_kwargs.update(kwargs)
            return response

        mock_litellm = MagicMock()
        mock_litellm.acompletion = capturing
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await s.summarize("text", "en")

        assert captured_kwargs.get("max_tokens") == 256

    async def test_max_tokens_not_passed_when_none(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, max_tokens=None)
        response = _make_llm_response("ok")
        captured_kwargs: dict = {}

        async def capturing(**kwargs):
            captured_kwargs.update(kwargs)
            return response

        mock_litellm = MagicMock()
        mock_litellm.acompletion = capturing
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await s.summarize("text", "en")

        assert "max_tokens" not in captured_kwargs

    async def test_temperature_passed(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, temperature=0.7)
        response = _make_llm_response("ok")
        captured_kwargs: dict = {}

        async def capturing(**kwargs):
            captured_kwargs.update(kwargs)
            return response

        mock_litellm = MagicMock()
        mock_litellm.acompletion = capturing
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await s.summarize("text", "en")

        assert captured_kwargs.get("temperature") == 0.7

    async def test_messages_contain_rendered_prompt(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt", "Sum {transcript} in {language}.")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        response = _make_llm_response("ok")
        captured_kwargs: dict = {}

        async def capturing(**kwargs):
            captured_kwargs.update(kwargs)
            return response

        mock_litellm = MagicMock()
        mock_litellm.acompletion = capturing
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await s.summarize("my transcript", "he")

        msgs = captured_kwargs.get("messages", [])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "my transcript" in msgs[0]["content"]
        assert "he" in msgs[0]["content"]

    async def test_zero_usage_tokens_default(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        response = _make_llm_response("ok")
        response.usage = None  # no usage info

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=response)
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = await s.summarize("text", "en")

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


# ---------------------------------------------------------------------------
# LiteLLMSummarizer.summarize — failure paths
# ---------------------------------------------------------------------------

class TestSummarizeFailure:
    async def test_litellm_not_installed_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(SummarizationError, match="not installed"):
                await s.summarize("text", "en")

    async def test_acompletion_exception_raises_summarization_error(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(side_effect=ConnectionError("refused"))
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with pytest.raises(SummarizationError, match="LiteLLM call failed"):
                await s.summarize("text", "en")

    async def test_empty_response_content_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        response = _make_llm_response("")

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=response)
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with pytest.raises(SummarizationError, match="empty summary"):
                await s.summarize("text", "en")

    async def test_whitespace_only_response_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        response = _make_llm_response("   \n  ")

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=response)
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with pytest.raises(SummarizationError, match="empty summary"):
                await s.summarize("text", "en")

    async def test_malformed_response_shape_raises(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f)
        bad_response = MagicMock()
        bad_response.choices = []  # empty list

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=bad_response)
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with pytest.raises(SummarizationError, match="Unexpected response shape"):
                await s.summarize("text", "en")

    async def test_timeout_passed_to_acompletion(self, tmp_path):
        f = _write_prompt(tmp_path / "p.txt")
        s = LiteLLMSummarizer(model="m", prompt_path=f, timeout=30.0)
        response = _make_llm_response("ok")
        captured_kwargs: dict = {}

        async def capturing(**kwargs):
            captured_kwargs.update(kwargs)
            return response

        mock_litellm = MagicMock()
        mock_litellm.acompletion = capturing
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await s.summarize("text", "en")

        assert captured_kwargs.get("timeout") == 30.0
