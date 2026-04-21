"""Gateway configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
All settings can be overridden via environment variables prefixed with GATEWAY_.
"""

from __future__ import annotations

import logging
import os
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env file early so YAML ${ } interpolation can reference env vars
load_dotenv()

logger = logging.getLogger(__name__)

# ── Project root detection ──────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Pre-compiled regex for env var interpolation (avoid per-call compilation)
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


class Environment(StrEnum):
    """Application environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class GatewaySettings(BaseSettings):
    """Central configuration for the Inference Gateway.

    All fields can be overridden by environment variables
    prefixed with ``GATEWAY_`` (e.g. ``GATEWAY_PORT=9090``).
    """

    # ── Server ──────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8080, description="Server port")
    workers: int = Field(default=1, description="Number of Uvicorn workers")
    env: Environment = Field(default=Environment.DEVELOPMENT, description="Runtime environment")
    log_level: str = Field(default="info", description="Logging level")

    # ── Auth ────────────────────────────────────────────────
    auth_enabled: bool = Field(default=True, description="Enable API key authentication")

    # ── Rate Limiting ───────────────────────────────────────
    default_rate_limit_rpm: int = Field(
        default=60, description="Default rate limit (requests per minute)"
    )

    # ── Config file paths ───────────────────────────────────
    routes_config: str = Field(default="configs/routes.yaml", description="Backend routes config")
    ab_tests_config: str = Field(
        default="configs/ab_tests.yaml", description="A/B test config"
    )
    tenants_config: str = Field(default="configs/tenants.yaml", description="Tenants config")

    model_config = {
        "env_prefix": "GATEWAY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        return v.lower()

    # ── Helpers ─────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION

    def resolve_config_path(self, relative_path: str) -> Path:
        """Resolve a config path relative to the project root."""
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return _PROJECT_ROOT / path


def _interpolate_env_vars(data: Any) -> Any:
    """Recursively replace ``${VAR_NAME}`` placeholders in YAML values
    with the corresponding environment variable.
    """
    if isinstance(data, str):
        def _replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name, "")
            if not value:
                logger.debug("Environment variable %s is not set (referenced in config)", var_name)
            return value
        return _ENV_VAR_PATTERN.sub(_replacer, data)
    if isinstance(data, dict):
        return {k: _interpolate_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate_env_vars(item) for item in data]
    return data


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file with environment variable interpolation.

    Supports ``${ENV_VAR}`` syntax in string values.

    Returns empty dict if file doesn't exist or parsing fails.
    Logs warnings instead of crashing — the gateway should start
    even with missing/corrupt configs (graceful degradation).
    """
    if not path.exists():
        logger.warning("Config file not found: %s (using defaults)", path)
        return {}
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            logger.warning("Config file is empty: %s", path)
            return {}
        if not isinstance(raw, dict):
            logger.error("Config file %s must be a YAML mapping, got %s", path, type(raw).__name__)
            return {}
        return _interpolate_env_vars(raw)
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML config %s: %s", path, e)
        return {}
    except OSError as e:
        logger.error("Failed to read config file %s: %s", path, e)
        return {}


@lru_cache(maxsize=1)
def get_settings() -> GatewaySettings:
    """Return the singleton settings instance (cached)."""
    return GatewaySettings()
