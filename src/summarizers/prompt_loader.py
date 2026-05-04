from __future__ import annotations

from pathlib import Path

from src.core.exceptions import ConfigurationError


def load_prompt(prompt_path: str | Path) -> str:
    """Load a prompt template from disk.

    Raises ConfigurationError if the file is missing, unreadable, or empty.
    Never returns an empty string — an empty prompt is always a configuration
    error that must surface immediately.
    """
    path = Path(prompt_path)

    if not path.exists():
        raise ConfigurationError(
            f"Prompt file not found: {path!r}. "
            "Check summarizer.prompt_path in config.yaml."
        )
    if not path.is_file():
        raise ConfigurationError(
            f"Prompt path is not a file: {path!r}"
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read prompt file {path!r}: {exc}"
        ) from exc

    stripped = text.strip()
    if not stripped:
        raise ConfigurationError(
            f"Prompt file is empty: {path!r}. "
            "Provide a non-empty system prompt."
        )

    return stripped


def render_prompt(template: str, transcript: str, language: str) -> str:
    """Substitute {transcript} and {language} placeholders in the template.

    Raises ConfigurationError if the template contains neither placeholder
    (it almost certainly means the wrong file was loaded).
    """
    if "{transcript}" not in template and "{language}" not in template:
        raise ConfigurationError(
            "Prompt template contains neither {transcript} nor {language} "
            "placeholders. Verify the prompt file content."
        )

    return template.replace("{transcript}", transcript).replace("{language}", language)
