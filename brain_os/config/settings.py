"""
BrAIn OS — Configuration Settings

Type-safe configuration via Pydantic BaseSettings.
Loads from environment variables and .env file with validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """Redis connection configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = "redis://localhost:6379/0"
    password: str = ""


class ChromaDBSettings(BaseSettings):
    """ChromaDB vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="CHROMADB_")

    host: str = "localhost"
    port: int = 8000
    persist_dir: str = "./data/chromadb"


class LLMSettings(BaseSettings):
    """LLM backend configuration — supports Ollama (local) and Groq (cloud) via litellm."""

    model_config = SettingsConfigDict(env_prefix="")

    # Primary: Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Fallback: Groq (cloud)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"

    # Which backend to use by default
    llm_default_backend: str = "ollama"

    @property
    def default_backend(self) -> str:
        """Alias for default backend name."""
        return self.llm_default_backend

    @property
    def primary_model(self) -> str:
        """Get the litellm model string for the primary backend."""
        if self.llm_default_backend == "ollama":
            return f"ollama/{self.ollama_model}"
        return f"groq/{self.groq_model}"

    @property
    def fallback_model(self) -> str | None:
        """Get the litellm model string for the fallback backend."""
        if self.llm_default_backend == "ollama" and self.groq_api_key:
            return f"groq/{self.groq_model}"
        if self.llm_default_backend == "groq":
            return f"ollama/{self.ollama_model}"
        return None


class EmbeddingSettings(BaseSettings):
    """Embedding and reranking model configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SchedulerSettings(BaseSettings):
    """Async scheduler configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    max_concurrent_agents: int = Field(default=5, ge=1, le=20)
    agent_timeout_seconds: int = Field(default=120, ge=10)
    priority_aging_seconds: int = Field(default=30, ge=5)


class MemorySettings(BaseSettings):
    """Memory hierarchy configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    l1_cache_ttl_seconds: int = Field(default=300, ge=10)
    l1_cache_max_items: int = Field(default=100, ge=10)
    l2_session_ttl_seconds: int = Field(default=3600, ge=60)


class SecuritySettings(BaseSettings):
    """Security and rate limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    rate_limit_requests_per_minute: int = Field(default=60, ge=1)
    rate_limit_burst: int = Field(default=10, ge=1)


class ObservabilitySettings(BaseSettings):
    """OpenTelemetry and logging configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "brain-os"
    log_level: str = "INFO"
    log_format: str = "json"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v


class BrainOSSettings(BaseSettings):
    """
    Root configuration for BrAIn OS.

    All sub-configurations are composed here for single-point access:
        settings = BrainOSSettings()
        settings.redis.url
        settings.llm.primary_model
        settings.scheduler.max_concurrent_agents
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurations
    redis: RedisSettings = Field(default_factory=RedisSettings)
    chromadb: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: APISettings = Field(default_factory=APISettings)

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    @property
    def agent_capabilities_path(self) -> Path:
        """Path to the agent capabilities YAML file."""
        return self.project_root / "brain_os" / "config" / "agent_capabilities.yaml"


# --- Singleton Access ---
_settings: BrainOSSettings | None = None


def get_settings() -> BrainOSSettings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = BrainOSSettings()
    return _settings
