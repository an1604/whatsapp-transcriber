from __future__ import annotations

import pytest
import pytest_asyncio
import yaml

from src.core.database import Database


@pytest_asyncio.fixture
async def db() -> Database:
    """Fresh in-memory SQLite database per test — guarantees full isolation."""
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_tables()
    yield database
    await database.drop_tables()
    await database.close()


@pytest_asyncio.fixture
async def session(db: Database):
    """Open session tied to the test-scoped in-memory db."""
    async with db._session_factory() as s:
        yield s


@pytest.fixture
def valid_config_dict() -> dict:
    return {
        "whatsapp": {
            "sidecar_url": "http://localhost:3000",
            "community_ids": [],
            "scrape_interval_minutes": 60,
        },
        "platforms": {
            "enabled": ["youtube", "instagram", "tiktok", "facebook"],
        },
        "transcriber": {
            "type": "faster_whisper",
            "params": {
                "model_size": "small",
                "device": "cpu",
                "compute_type": "int8",
                "language": "auto",
            },
        },
        "summarizer": {
            "type": "litellm",
            "params": {
                "model": "ollama/llama3.1:8b",
                "api_base": "http://localhost:11434",
                "temperature": 0.3,
            },
            "prompt_path": "config/prompts/summarize_default.txt",
        },
        "deduplication": {
            "enabled": True,
            "resolve_redirects": True,
        },
        "storage": {
            "database_url": "sqlite+aiosqlite:///data/app.db",
            "audio_cache_dir": "data/audio_cache",
            "audio_retention_days": None,
        },
        "web": {
            "host": "0.0.0.0",
            "port": 8000,
        },
    }


@pytest.fixture
def valid_config_file(valid_config_dict: dict, tmp_path):
    """Write a valid config.yaml to a temp directory and return its Path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(valid_config_dict), encoding="utf-8")
    return config_file


def write_config(tmp_path, data: dict):
    """Helper: write a config dict as YAML and return the Path."""
    f = tmp_path / "config.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    return f
