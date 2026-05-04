from __future__ import annotations

import logging
from pathlib import Path

from src.core.exceptions import SummarizationError
from src.summarizers.base import AbstractSummarizer, SummaryResult
from src.summarizers.prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)


class LiteLLMSummarizer(AbstractSummarizer):
    """Summarizes transcripts via LiteLLM (supports Ollama, OpenAI, Anthropic, etc.).

    The prompt template is loaded once at construction time.
    Supports any model string that LiteLLM understands (e.g. "ollama/llama3.1:8b").

    Raises SummarizationError on any LLM or network failure. Never falls back
    to an empty or partial result.
    """

    def __init__(
        self,
        model: str,
        prompt_path: str | Path,
        api_base: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not model.strip():
            raise SummarizationError("model must not be empty or blank")
        if not 0.0 <= temperature <= 2.0:
            raise SummarizationError(
                f"temperature must be between 0.0 and 2.0, got {temperature}"
            )
        if max_tokens is not None and max_tokens <= 0:
            raise SummarizationError(
                f"max_tokens must be a positive integer, got {max_tokens}"
            )
        if timeout <= 0:
            raise SummarizationError(
                f"timeout must be positive, got {timeout}"
            )

        self._model = model
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._prompt_template = load_prompt(prompt_path)  # raises ConfigurationError

    async def summarize(self, transcript: str, language: str) -> SummaryResult:
        """Call the LLM to summarize transcript, returning a SummaryResult.

        Raises SummarizationError on any failure.
        """
        if not transcript.strip():
            raise SummarizationError(
                "transcript is empty — nothing to summarize"
            )
        if not language.strip():
            raise SummarizationError(
                "language must not be empty"
            )

        prompt = render_prompt(self._prompt_template, transcript, language)

        logger.info("Summarizing with model=%s, lang=%s", self._model, language)

        try:
            import litellm  # noqa: PLC0415
        except ImportError as exc:
            raise SummarizationError(
                "litellm is not installed. Run: pip install litellm"
            ) from exc

        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "timeout": self._timeout,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise SummarizationError(
                f"LiteLLM call failed for model {self._model!r}: {exc}"
            ) from exc

        try:
            summary_text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise SummarizationError(
                f"Unexpected response shape from LiteLLM: {exc}. "
                f"Raw response: {response!r}"
            ) from exc

        if not summary_text or not summary_text.strip():
            raise SummarizationError(
                f"LiteLLM returned an empty summary for model {self._model!r}"
            )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        logger.info(
            "Summary done: %d prompt tokens, %d completion tokens",
            prompt_tokens,
            completion_tokens,
        )

        return SummaryResult(
            summary=summary_text.strip(),
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
