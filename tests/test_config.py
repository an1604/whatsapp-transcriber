from __future__ import annotations

import copy

import pytest
import yaml

from src.core.config import AppConfig, load_config
from src.core.exceptions import ConfigurationError
from tests.conftest import write_config


class TestLoadConfig:
    def test_load_valid_config(self, valid_config_file):
        config = load_config(valid_config_file)
        assert isinstance(config, AppConfig)

    def test_config_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(missing)

    def test_config_path_is_directory(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not a file"):
            load_config(tmp_path)

    def test_config_invalid_yaml_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed bracket: {", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_config(bad)

    def test_config_yaml_list_at_root_raises(self, tmp_path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="YAML mapping"):
            load_config(bad)

    def test_config_yaml_scalar_at_root_raises(self, tmp_path):
        bad = tmp_path / "scalar.yaml"
        bad.write_text("just a string\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="YAML mapping"):
            load_config(bad)

    def test_config_missing_top_level_section_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        del d["transcriber"]
        with pytest.raises(ConfigurationError, match="validation failed"):
            load_config(write_config(tmp_path, d))

    def test_config_missing_nested_field_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        del d["transcriber"]["params"]["model_size"]
        with pytest.raises(ConfigurationError, match="validation failed"):
            load_config(write_config(tmp_path, d))

    def test_config_extra_top_level_keys_are_ignored(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["unknown_section"] = {"foo": "bar"}
        # Pydantic v2 ignores extra fields by default
        config = load_config(write_config(tmp_path, d))
        assert isinstance(config, AppConfig)


class TestWhatsAppConfig:
    def test_valid_values(self, valid_config_file):
        cfg = load_config(valid_config_file)
        assert cfg.whatsapp.sidecar_url == "http://localhost:3000"
        assert cfg.whatsapp.community_ids == []
        assert cfg.whatsapp.scrape_interval_minutes == 60

    def test_community_ids_populated(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["whatsapp"]["community_ids"] = ["120363xxx@g.us", "120363yyy@g.us"]
        cfg = load_config(write_config(tmp_path, d))
        assert len(cfg.whatsapp.community_ids) == 2

    def test_scrape_interval_zero_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["whatsapp"]["scrape_interval_minutes"] = 0
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_scrape_interval_negative_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["whatsapp"]["scrape_interval_minutes"] = -10
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_scrape_interval_one_is_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["whatsapp"]["scrape_interval_minutes"] = 1
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.whatsapp.scrape_interval_minutes == 1


class TestPlatformsConfig:
    def test_all_four_platforms_loaded(self, valid_config_file):
        cfg = load_config(valid_config_file)
        assert set(cfg.platforms.enabled) == {"youtube", "instagram", "tiktok", "facebook"}

    def test_single_platform_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["platforms"]["enabled"] = ["youtube"]
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.platforms.enabled == ["youtube"]

    def test_unknown_platform_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["platforms"]["enabled"] = ["youtube", "snapchat"]
        with pytest.raises(ConfigurationError, match="Unknown platform"):
            load_config(write_config(tmp_path, d))

    def test_multiple_unknown_platforms_listed_in_error(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["platforms"]["enabled"] = ["snapchat", "twitter"]
        with pytest.raises(ConfigurationError, match="Unknown platform"):
            load_config(write_config(tmp_path, d))

    def test_empty_platforms_list_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["platforms"]["enabled"] = []
        with pytest.raises(ConfigurationError, match="at least one"):
            load_config(write_config(tmp_path, d))

    def test_duplicate_platforms_are_deduplicated(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["platforms"]["enabled"] = ["youtube", "youtube", "facebook", "facebook"]
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.platforms.enabled.count("youtube") == 1
        assert cfg.platforms.enabled.count("facebook") == 1

    def test_dedup_preserves_order(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["platforms"]["enabled"] = ["facebook", "youtube", "facebook"]
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.platforms.enabled[0] == "facebook"
        assert cfg.platforms.enabled[1] == "youtube"


class TestTranscriberConfig:
    def test_valid_faster_whisper(self, valid_config_file):
        cfg = load_config(valid_config_file)
        assert cfg.transcriber.type == "faster_whisper"
        assert cfg.transcriber.params.device == "cpu"
        assert cfg.transcriber.params.compute_type == "int8"
        assert cfg.transcriber.params.language == "auto"

    def test_unknown_type_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["transcriber"]["type"] = "google_stt"
        with pytest.raises(ConfigurationError, match="Unknown transcriber type"):
            load_config(write_config(tmp_path, d))

    def test_whisper_api_type_is_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["transcriber"]["type"] = "whisper_api"
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.transcriber.type == "whisper_api"


class TestSummarizerConfig:
    def test_valid_litellm(self, valid_config_file):
        cfg = load_config(valid_config_file)
        assert cfg.summarizer.type == "litellm"
        assert cfg.summarizer.params.temperature == pytest.approx(0.3)
        assert cfg.summarizer.params.model == "ollama/llama3.1:8b"

    def test_unknown_type_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["summarizer"]["type"] = "langchain"
        with pytest.raises(ConfigurationError, match="Unknown summarizer type"):
            load_config(write_config(tmp_path, d))

    def test_temperature_above_max_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["summarizer"]["params"]["temperature"] = 2.01
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_temperature_below_min_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["summarizer"]["params"]["temperature"] = -0.01
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_temperature_boundary_zero_is_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["summarizer"]["params"]["temperature"] = 0.0
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.summarizer.params.temperature == 0.0

    def test_temperature_boundary_two_is_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["summarizer"]["params"]["temperature"] = 2.0
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.summarizer.params.temperature == 2.0

    def test_prompt_path_preserved(self, valid_config_file):
        cfg = load_config(valid_config_file)
        assert cfg.summarizer.prompt_path == "config/prompts/summarize_default.txt"


class TestStorageConfig:
    def test_audio_retention_null_means_keep_forever(self, valid_config_file):
        cfg = load_config(valid_config_file)
        assert cfg.storage.audio_retention_days is None

    def test_positive_retention_days_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["storage"]["audio_retention_days"] = 30
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.storage.audio_retention_days == 30

    def test_retention_one_day_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["storage"]["audio_retention_days"] = 1
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.storage.audio_retention_days == 1

    def test_retention_zero_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["storage"]["audio_retention_days"] = 0
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_retention_negative_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["storage"]["audio_retention_days"] = -7
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_empty_database_url_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["storage"]["database_url"] = ""
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_blank_database_url_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["storage"]["database_url"] = "   "
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))


class TestWebConfig:
    def test_valid_defaults(self, valid_config_file):
        cfg = load_config(valid_config_file)
        assert cfg.web.host == "0.0.0.0"
        assert cfg.web.port == 8000

    def test_port_zero_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["web"]["port"] = 0
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_port_above_max_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["web"]["port"] = 65536
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))

    def test_port_boundary_1_is_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["web"]["port"] = 1
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.web.port == 1

    def test_port_boundary_65535_is_valid(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["web"]["port"] = 65535
        cfg = load_config(write_config(tmp_path, d))
        assert cfg.web.port == 65535

    def test_negative_port_raises(self, valid_config_dict, tmp_path):
        d = copy.deepcopy(valid_config_dict)
        d["web"]["port"] = -1
        with pytest.raises(ConfigurationError):
            load_config(write_config(tmp_path, d))
