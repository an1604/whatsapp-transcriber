from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryResult:
    """Output of a summarization run."""

    summary: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class AbstractSummarizer(ABC):
    """Interface for all LLM summarization backends.

    Implementations MUST raise SummarizationError (from src.core.exceptions)
    on any failure — no silent fallbacks, no empty-string returns for errors.
    """

    @abstractmethod
    async def summarize(self, transcript: str, language: str) -> SummaryResult:
        """Summarize the transcript text.

        Args:
            transcript: The full transcription text to summarize.
            language: Detected language code (e.g. "en", "he") — used to
                      instruct the model to reply in the same language.

        Returns a SummaryResult with the summary and token counts.

        Raises:
            SummarizationError: on any LLM or network failure.
        """
