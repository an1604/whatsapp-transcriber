from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator

from src.core.exceptions import ConfigurationError
from src.core.models import Platform

_VALID_TRANSCRIBER_TYPES: frozenset[str] = frozenset({"faster_whisper", "whisper_api"})
_VALID_SUMMARIZER_TYPES: frozenset[str] = frozenset({"litellm"})
_VALID_PLATFORMS: frozenset[str] = frozenset(p.value for p in Platform)


class WhatsAppConfig(BaseModel):
    sidecar_url: str
    community_ids: list[str]
    scrape_interval_minutes: int

    @field_validator("scrape_interval_minutes")
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("scrape_interval_minutes must be a positive integer")
        return v


class PlatformsConfig(BaseModel):
    enabled: list[str]

    @field_validator("enabled")
    @classmethod
    def _validate_platforms(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("platforms.enabled must contain at least one platform")
        unknown = set(v) - _VALID_PLATFORMS
        if unknown:
            raise ValueError(
                f"Unknown platform(s): {sorted(unknown)}. "
                f"Valid options: {sorted(_VALID_PLATFORMS)}"
            )
        # deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for item in v:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped


class TranscriberParams(BaseModel):
    model_size: str
    device: str
    compute_type: str
    language: str


class TranscriberConfig(BaseModel):
    type: str
    params: TranscriberParams

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _VALID_TRANSCRIBER_TYPES:
            raise ValueError(
                f"Unknown transcriber type: {v!r}. "
                f"Valid options: {sorted(_VALID_TRANSCRIBER_TYPES)}"
            )
        return v


class SummarizerParams(BaseModel):
    model: str
    api_base: str
    temperature: float

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0 (inclusive)")
        return v


class SummarizerConfig(BaseModel):
    type: str
    params: SummarizerParams
    prompt_path: str

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _VALID_SUMMARIZER_TYPES:
            raise ValueError(
                f"Unknown summarizer type: {v!r}. "
                f"Valid options: {sorted(_VALID_SUMMARIZER_TYPES)}"
            )
        return v


class DeduplicationConfig(BaseModel):
    enabled: bool
    resolve_redirects: bool


class StorageConfig(BaseModel):
    database_url: str
    audio_cache_dir: str
    audio_retention_days: Optional[int]

    @field_validator("database_url")
    @classmethod
    def _database_url_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("storage.database_url must not be empty or blank")
        return v

    @field_validator("audio_retention_days")
    @classmethod
    def _retention_must_be_positive_or_null(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(
                "storage.audio_retention_days must be a positive integer or null "
                "(null means keep forever)"
            )
        return v


class WebConfig(BaseModel):
    host: str
    port: int

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("web.port must be between 1 and 65535")
        return v


class AppConfig(BaseModel):
    whatsapp: WhatsAppConfig
    platforms: PlatformsConfig
    transcriber: TranscriberConfig
    summarizer: SummarizerConfig
    deduplication: DeduplicationConfig
    storage: StorageConfig
    web: WebConfig


# ---------------------------------------------------------------------------
# Environment variable override support (for Docker / 12-factor deployments)
# ---------------------------------------------------------------------------

# Maps env var name → dotted path into the raw config dict.
# Leaf values are always strings from the environment; int coercion happens
# for fields whose last key is "port" or "scrape_interval_minutes".
_ENV_OVERRIDE_MAP: dict[str, tuple[str, ...]] = {
    "DATABASE_URL": ("storage", "database_url"),
    "AUDIO_CACHE_DIR": ("storage", "audio_cache_dir"),
    "OLLAMA_BASE_URL": ("summarizer", "params", "api_base"),
    "WEB_HOST": ("web", "host"),
    "WEB_PORT": ("web", "port"),
    "WHISPER_MODEL_SIZE": ("transcriber", "params", "model_size"),
}

_INT_LEAF_KEYS: frozenset[str] = frozenset({"port", "scrape_interval_minutes"})


def _apply_env_overrides(raw: dict, env: dict[str, str]) -> None:
    """Mutate *raw* in-place, applying every matching env var override."""
    for env_key, path in _ENV_OVERRIDE_MAP.items():
        if env_key not in env:
            continue
        value_str = env[env_key].strip()
        node = raw
        for part in path[:-1]:
            node = node.setdefault(part, {})
        leaf_key = path[-1]
        if leaf_key in _INT_LEAF_KEYS:
            try:
                node[leaf_key] = int(value_str)
            except ValueError:
                raise ConfigurationError(
                    f"Environment variable {env_key} must be an integer, got {value_str!r}"
                )
        else:
            node[leaf_key] = value_str


def load_config_with_env_overrides(
    config_path: Path,
    env: dict[str, str] | None = None,
) -> AppConfig:
    """Load config from *config_path* and apply environment variable overrides.

    ``env`` defaults to ``os.environ`` when *None*. Pass an explicit mapping in
    tests to avoid touching the real environment.

    Raises ConfigurationError for any problem — same contract as load_config().
    """
    if env is None:
        env = dict(os.environ)

    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigurationError(f"Config path is not a file: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in config file {config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Config file must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__!r}"
        )

    _apply_env_overrides(raw, env)

    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"Config validation failed: {exc}") from exc


def load_config(config_path: Path) -> AppConfig:
    """Load and validate the application config from a YAML file.

    Raises ConfigurationError for any problem — missing file, bad YAML,
    or schema validation failure. Never returns a partially-valid config.
    """
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigurationError(f"Config path is not a file: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in config file {config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Config file must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__!r}"
        )

    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"Config validation failed: {exc}") from exc
