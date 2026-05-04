"""Tests for Step 8: Docker Compose infrastructure and env-var config overrides."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.config import (
    AppConfig,
    load_config_with_env_overrides,
    _ENV_OVERRIDE_MAP,
)
from src.core.exceptions import ConfigurationError

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
DOCKER_CONFIG = CONFIG_DIR / "config.docker.yaml"
PROMPT_FILE = CONFIG_DIR / "prompts" / "summarize_default.txt"
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE_FILE = ROOT / "docker-compose.yml"
ENTRYPOINT = ROOT / "scripts" / "entrypoint.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


def _base_env() -> dict[str, str]:
    """Minimal env that satisfies the docker config as-is (no overrides)."""
    return {}


# ---------------------------------------------------------------------------
# docker-compose.yml structure
# ---------------------------------------------------------------------------

class TestDockerComposeFile:
    def test_file_exists(self):
        assert COMPOSE_FILE.exists(), "docker-compose.yml must exist at project root"

    def test_is_valid_yaml(self):
        compose = _load_compose()
        assert isinstance(compose, dict)

    def test_has_services_key(self):
        compose = _load_compose()
        assert "services" in compose

    def test_has_app_service(self):
        compose = _load_compose()
        assert "app" in compose["services"]

    def test_has_ollama_service(self):
        compose = _load_compose()
        assert "ollama" in compose["services"]

    def test_app_exposes_port_8000(self):
        compose = _load_compose()
        ports = compose["services"]["app"].get("ports", [])
        assert any("8000" in str(p) for p in ports), "app must expose port 8000"

    def test_ollama_exposes_port_11434(self):
        compose = _load_compose()
        ports = compose["services"]["ollama"].get("ports", [])
        assert any("11434" in str(p) for p in ports), "ollama must expose port 11434"

    def test_app_depends_on_ollama(self):
        compose = _load_compose()
        depends = compose["services"]["app"].get("depends_on", [])
        if isinstance(depends, dict):
            assert "ollama" in depends
        else:
            assert "ollama" in depends

    def test_volumes_key_exists(self):
        compose = _load_compose()
        assert "volumes" in compose

    def test_data_volume_defined(self):
        compose = _load_compose()
        assert "data" in compose["volumes"]

    def test_ollama_data_volume_defined(self):
        compose = _load_compose()
        assert "ollama_data" in compose["volumes"]

    def test_app_has_healthcheck(self):
        compose = _load_compose()
        assert "healthcheck" in compose["services"]["app"], "app service must have healthcheck"

    def test_app_has_restart_policy(self):
        compose = _load_compose()
        assert "restart" in compose["services"]["app"]

    def test_app_environment_has_database_url(self):
        compose = _load_compose()
        env = compose["services"]["app"].get("environment", {})
        if isinstance(env, list):
            keys = [e.split("=")[0] for e in env]
        else:
            keys = list(env.keys())
        assert "DATABASE_URL" in keys

    def test_app_environment_has_ollama_url(self):
        compose = _load_compose()
        env = compose["services"]["app"].get("environment", {})
        if isinstance(env, list):
            values = [e for e in env if "OLLAMA" in e]
        else:
            values = [k for k in env if "OLLAMA" in k]
        assert values, "app environment must reference Ollama"

    def test_app_mounts_data_volume(self):
        compose = _load_compose()
        volumes = compose["services"]["app"].get("volumes", [])
        assert any("data" in str(v) for v in volumes)

    def test_ollama_uses_official_image(self):
        compose = _load_compose()
        image = compose["services"]["ollama"].get("image", "")
        assert "ollama" in image.lower()

    def test_ollama_mounts_ollama_data_volume(self):
        compose = _load_compose()
        volumes = compose["services"]["ollama"].get("volumes", [])
        assert any("ollama_data" in str(v) for v in volumes)

    def test_ollama_has_restart_policy(self):
        compose = _load_compose()
        assert "restart" in compose["services"]["ollama"]

    def test_has_whatsapp_sidecar_service(self):
        compose = _load_compose()
        assert "whatsapp-sidecar" in compose["services"]

    def test_sidecar_exposes_port_3000(self):
        compose = _load_compose()
        ports = compose["services"]["whatsapp-sidecar"].get("ports", [])
        assert any("3000" in str(p) for p in ports)

    def test_sidecar_depends_on_app(self):
        compose = _load_compose()
        depends = compose["services"]["whatsapp-sidecar"].get("depends_on", [])
        if isinstance(depends, dict):
            assert "app" in depends
        else:
            assert "app" in depends

    def test_sidecar_mounts_wa_session_volume(self):
        compose = _load_compose()
        vols = compose["services"]["whatsapp-sidecar"].get("volumes", [])
        assert any("wa_session" in str(v) for v in vols)

    def test_wa_session_volume_defined(self):
        compose = _load_compose()
        assert "wa_session" in compose["volumes"]

    def test_sidecar_has_pipeline_url_env(self):
        compose = _load_compose()
        env = compose["services"]["whatsapp-sidecar"].get("environment", {})
        if isinstance(env, list):
            keys = [e.split("=")[0] for e in env]
        else:
            keys = list(env.keys())
        assert "PIPELINE_URL" in keys


# ---------------------------------------------------------------------------
# Sidecar Dockerfile structure
# ---------------------------------------------------------------------------

SIDECAR_DOCKERFILE = ROOT / "whatsapp-sidecar" / "Dockerfile"


class TestSidecarDockerfile:
    def _content(self) -> str:
        return SIDECAR_DOCKERFILE.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert SIDECAR_DOCKERFILE.exists()

    def test_uses_node_base_image(self):
        assert "node:" in self._content().lower()

    def test_exposes_port_3000(self):
        assert "EXPOSE 3000" in self._content()

    def test_has_healthcheck(self):
        assert "HEALTHCHECK" in self._content()

    def test_has_cmd(self):
        assert "CMD" in self._content()

    def test_has_volume_for_wa_session(self):
        assert "VOLUME" in self._content()

    def test_copies_src_directory(self):
        content = self._content()
        assert "COPY src/" in content or "COPY src " in content


# ---------------------------------------------------------------------------
# Dockerfile structure
# ---------------------------------------------------------------------------

class TestDockerfile:
    def _lines(self) -> list[str]:
        return DOCKERFILE.read_text(encoding="utf-8").splitlines()

    def test_file_exists(self):
        assert DOCKERFILE.exists(), "Dockerfile must exist at project root"

    def test_has_from_instruction(self):
        lines = self._lines()
        from_lines = [l for l in lines if l.strip().upper().startswith("FROM ")]
        assert from_lines, "Dockerfile must have at least one FROM instruction"

    def test_uses_python_base_image(self):
        lines = self._lines()
        from_lines = [l for l in lines if l.strip().upper().startswith("FROM ")]
        assert any("python" in l.lower() for l in from_lines)

    def test_has_two_build_stages(self):
        lines = self._lines()
        from_lines = [l for l in lines if l.strip().upper().startswith("FROM ")]
        assert len(from_lines) >= 2, "Must use multi-stage build (builder + runtime)"

    def test_first_stage_named_builder(self):
        lines = self._lines()
        from_lines = [l.strip() for l in lines if l.strip().upper().startswith("FROM ")]
        assert any("builder" in l.lower() for l in from_lines)

    def test_second_stage_named_runtime(self):
        lines = self._lines()
        from_lines = [l.strip() for l in lines if l.strip().upper().startswith("FROM ")]
        assert any("runtime" in l.lower() for l in from_lines)

    def test_exposes_port_8000(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "EXPOSE 8000" in content or "EXPOSE\n8000" in content

    def test_has_volume_instruction(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "VOLUME" in content

    def test_copies_src_directory(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY src/" in content or "COPY src " in content

    def test_uses_ffmpeg(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "ffmpeg" in content.lower()

    def test_has_healthcheck(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content

    def test_has_entrypoint(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "ENTRYPOINT" in content

    def test_has_cmd(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "CMD" in content

    def test_copies_from_builder_stage(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY --from=builder" in content


# ---------------------------------------------------------------------------
# Docker-specific config file
# ---------------------------------------------------------------------------

class TestDockerConfig:
    def test_config_file_exists(self):
        assert DOCKER_CONFIG.exists(), "config/config.docker.yaml must exist"

    def test_config_is_valid_yaml(self):
        data = yaml.safe_load(DOCKER_CONFIG.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_config_passes_appconfig_validation(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert isinstance(cfg, AppConfig)

    def test_database_url_uses_data_volume(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert "/data" in cfg.storage.database_url

    def test_ollama_service_name_in_api_base(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert "ollama" in cfg.summarizer.params.api_base

    def test_audio_cache_in_data_volume(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert "/data" in cfg.storage.audio_cache_dir

    def test_web_host_is_all_interfaces(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert cfg.web.host == "0.0.0.0"

    def test_web_port_is_8000(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert cfg.web.port == 8000

    def test_transcriber_uses_cpu(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert cfg.transcriber.params.device == "cpu"

    def test_audio_retention_set(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert cfg.storage.audio_retention_days is not None


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvVarOverrides:
    def test_no_overrides_loads_yaml_values(self):
        cfg = load_config_with_env_overrides(DOCKER_CONFIG, env={})
        assert "ollama:11434" in cfg.summarizer.params.api_base

    def test_database_url_override(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"DATABASE_URL": "sqlite+aiosqlite:///custom/test.db"}
        )
        assert cfg.storage.database_url == "sqlite+aiosqlite:///custom/test.db"

    def test_audio_cache_dir_override(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"AUDIO_CACHE_DIR": "/mnt/external/audio"}
        )
        assert cfg.storage.audio_cache_dir == "/mnt/external/audio"

    def test_ollama_base_url_override(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"OLLAMA_BASE_URL": "http://my-ollama-host:11434"}
        )
        assert cfg.summarizer.params.api_base == "http://my-ollama-host:11434"

    def test_web_host_override(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"WEB_HOST": "127.0.0.1"}
        )
        assert cfg.web.host == "127.0.0.1"

    def test_web_port_override(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"WEB_PORT": "9000"}
        )
        assert cfg.web.port == 9000

    def test_web_port_parsed_as_int(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"WEB_PORT": "8080"}
        )
        assert isinstance(cfg.web.port, int)
        assert cfg.web.port == 8080

    def test_invalid_web_port_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="WEB_PORT"):
            load_config_with_env_overrides(
                DOCKER_CONFIG, env={"WEB_PORT": "not_a_number"}
            )

    def test_out_of_range_web_port_raises(self):
        with pytest.raises(ConfigurationError):
            load_config_with_env_overrides(
                DOCKER_CONFIG, env={"WEB_PORT": "99999"}
            )

    def test_whisper_model_size_override(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"WHISPER_MODEL_SIZE": "large-v3"}
        )
        assert cfg.transcriber.params.model_size == "large-v3"

    def test_multiple_overrides_applied_together(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG,
            env={
                "DATABASE_URL": "sqlite+aiosqlite:///tmp/multi.db",
                "WEB_PORT": "7777",
                "WEB_HOST": "127.0.0.1",
            },
        )
        assert cfg.storage.database_url == "sqlite+aiosqlite:///tmp/multi.db"
        assert cfg.web.port == 7777
        assert cfg.web.host == "127.0.0.1"

    def test_unknown_env_var_is_ignored(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"TOTALLY_UNKNOWN_VAR": "surprise"}
        )
        assert isinstance(cfg, AppConfig)

    def test_whitespace_in_env_value_is_stripped(self):
        cfg = load_config_with_env_overrides(
            DOCKER_CONFIG, env={"WEB_HOST": "  192.168.1.1  "}
        )
        assert cfg.web.host == "192.168.1.1"

    def test_whitespace_only_value_stripped_then_validated(self):
        with pytest.raises(ConfigurationError):
            load_config_with_env_overrides(
                DOCKER_CONFIG, env={"DATABASE_URL": "   "}
            )

    def test_override_does_not_affect_original_yaml_on_disk(self):
        original = yaml.safe_load(DOCKER_CONFIG.read_text(encoding="utf-8"))
        load_config_with_env_overrides(
            DOCKER_CONFIG, env={"DATABASE_URL": "sqlite+aiosqlite:///mutated.db"}
        )
        after = yaml.safe_load(DOCKER_CONFIG.read_text(encoding="utf-8"))
        assert original["storage"]["database_url"] == after["storage"]["database_url"]

    def test_env_override_map_covers_all_key_surfaces(self):
        keys = list(_ENV_OVERRIDE_MAP.keys())
        assert "DATABASE_URL" in keys
        assert "OLLAMA_BASE_URL" in keys
        assert "WEB_PORT" in keys


# ---------------------------------------------------------------------------
# Prompt and entrypoint files
# ---------------------------------------------------------------------------

class TestSupportFiles:
    def test_prompt_file_exists(self):
        assert PROMPT_FILE.exists(), f"Prompt file must exist: {PROMPT_FILE}"

    def test_prompt_has_transcript_placeholder(self):
        content = PROMPT_FILE.read_text(encoding="utf-8")
        assert "{transcript}" in content

    def test_prompt_has_language_placeholder(self):
        content = PROMPT_FILE.read_text(encoding="utf-8")
        assert "{language}" in content

    def test_prompt_is_not_empty(self):
        content = PROMPT_FILE.read_text(encoding="utf-8").strip()
        assert len(content) > 20

    def test_entrypoint_script_exists(self):
        assert ENTRYPOINT.exists(), f"Entrypoint script must exist: {ENTRYPOINT}"

    def test_entrypoint_is_shell_script(self):
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert content.startswith("#!/")

    def test_entrypoint_waits_for_ollama(self):
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "ollama" in content.lower()

    def test_entrypoint_uses_exec(self):
        content = ENTRYPOINT.read_text(encoding="utf-8")
        assert "exec" in content

    def test_dockerignore_exists(self):
        assert (ROOT / ".dockerignore").exists()

    def test_dockerignore_excludes_tests(self):
        content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert "tests" in content

    def test_dockerignore_excludes_git(self):
        content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert ".git" in content
