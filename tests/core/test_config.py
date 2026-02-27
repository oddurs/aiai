"""Tests for aiai_core.config."""

import os
import tempfile
from pathlib import Path

import yaml
from aiai_core.config import (
    load_evolution_config,
    load_models_config,
    load_safety_config,
)
from aiai_core.types import Complexity, ModelTier


def _write_yaml(dir_path: Path, filename: str, data: dict) -> None:  # type: ignore[type-arg]
    """Write a YAML file to a temporary config directory."""
    path = dir_path / filename
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestLoadModelsConfig:
    def test_load_real_config(self) -> None:
        """Load the actual config/models.yaml from the repo."""
        config = load_models_config()
        assert config.provider == "openrouter"
        assert ModelTier.NANO in config.tiers
        assert ModelTier.MAX in config.tiers
        assert config.routing[Complexity.TRIVIAL] == ModelTier.NANO
        assert config.routing[Complexity.CRITICAL] == ModelTier.MAX
        assert config.cost.daily_budget_usd == 50.0

    def test_tier_for_complexity(self) -> None:
        config = load_models_config()
        tier = config.tier_for_complexity(Complexity.SIMPLE)
        assert tier.name == ModelTier.FAST
        assert len(tier.models) > 0

    def test_load_custom_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "provider": "test",
                "tiers": {
                    "nano": {
                        "models": ["test/model-a"],
                        "max_tokens": 512,
                        "temperature": 0.0,
                    },
                },
                "routing": {"trivial": "nano"},
                "cost": {
                    "log_file": "test.jsonl",
                    "warn_threshold_usd": 0.5,
                    "daily_budget_usd": 10.0,
                },
            }
            _write_yaml(Path(tmpdir), "models.yaml", data)
            os.environ["AIAI_CONFIG_DIR"] = tmpdir
            try:
                config = load_models_config()
                assert config.provider == "test"
                assert config.cost.daily_budget_usd == 10.0
            finally:
                del os.environ["AIAI_CONFIG_DIR"]


class TestLoadSafetyConfig:
    def test_load_real_config(self) -> None:
        config = load_safety_config()
        assert "CLAUDE.md" in config.protected_files
        assert config.max_failures_per_task == 5
        assert config.max_cost_per_task_usd == 5.0

    def test_defaults_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AIAI_CONFIG_DIR"] = tmpdir
            try:
                config = load_safety_config()
                assert config.protected_files == []
                assert config.max_failures_per_task == 5
            finally:
                del os.environ["AIAI_CONFIG_DIR"]


class TestLoadEvolutionConfig:
    def test_load_real_config(self) -> None:
        config = load_evolution_config()
        assert config.enabled is False
        assert config.max_hypotheses_per_day == 3
        assert config.auto_revert_on_regression is True

    def test_defaults_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AIAI_CONFIG_DIR"] = tmpdir
            try:
                config = load_evolution_config()
                assert config.enabled is False
            finally:
                del os.environ["AIAI_CONFIG_DIR"]
