"""Pydantic models for YAML configuration files.

These models validate and type the YAML configs loaded from:
  - configs/routes.yaml
  - configs/ab_tests.yaml
  - configs/tenants.yaml
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Backend configuration ───────────────────────────────────


class BackendConfig(BaseModel):
    """Configuration for a single inference backend."""

    name: str = Field(..., description="Unique backend identifier")
    type: Literal["ollama", "vllm", "sglang", "openai_compatible"] = Field(
        ..., description="Backend type"
    )
    base_url: str = Field(..., description="Backend base URL")
    api_key: str = Field(default="", description="API key (or env var reference)")
    models: list[str] = Field(default_factory=list, description="Supported model identifiers")
    health_endpoint: str = Field(default="/health", description="Health check endpoint path")
    priority: int = Field(default=10, description="Priority in fallback chain (lower = higher)")
    tags: list[str] = Field(default_factory=list, description="Metadata tags")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    max_retries: int = Field(default=2, description="Max retries on transient errors")


class RoutingConfig(BaseModel):
    """Top-level routing configuration."""

    default_backend: str = Field(default="groq", description="Default backend name")
    fallback_order: list[str] = Field(
        default_factory=list, description="Ordered list of backend names for fallback"
    )


class RoutesConfig(BaseModel):
    """Root schema for routes.yaml."""

    backends: list[BackendConfig] = Field(default_factory=list)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


# ── A/B test configuration ──────────────────────────────────


class ABTestVariant(BaseModel):
    """A single variant in an A/B test."""

    backend: str = Field(..., description="Backend name for this variant")
    weight: int = Field(..., ge=0, le=100, description="Traffic weight (0-100)")


class ABTestConfig(BaseModel):
    """Configuration for a single A/B test."""

    name: str = Field(..., description="Unique test identifier")
    enabled: bool = Field(default=False, description="Whether this test is active")
    model: str = Field(default="*", description="Model pattern this test applies to")
    sticky: bool = Field(
        default=False, description="Use consistent hashing for same tenant → same variant"
    )
    variants: list[ABTestVariant] = Field(
        ..., min_length=2, description="At least 2 variants required"
    )


class ABTestsConfig(BaseModel):
    """Root schema for ab_tests.yaml."""

    ab_tests: list[ABTestConfig] = Field(default_factory=list)


# ── Tenant configuration ────────────────────────────────────


class TenantConfig(BaseModel):
    """Configuration for a single API tenant."""

    name: str = Field(..., description="Human-readable tenant name")
    api_key: str = Field(..., description="Bearer token for authentication")
    rate_limit_rpm: int = Field(default=60, ge=1, description="Requests per minute limit")
    allowed_backends: str | list[str] = Field(
        default="*", description="Allowed backends ('*' = all, or list of names)"
    )
    tier: Literal["free", "pro", "enterprise"] = Field(
        default="free", description="Tenant tier"
    )

    def can_access_backend(self, backend_name: str) -> bool:
        """Check if this tenant is allowed to use a specific backend."""
        if self.allowed_backends == "*":
            return True
        if isinstance(self.allowed_backends, list):
            return backend_name in self.allowed_backends
        return False


class TenantsConfig(BaseModel):
    """Root schema for tenants.yaml."""

    tenants: list[TenantConfig] = Field(default_factory=list)
