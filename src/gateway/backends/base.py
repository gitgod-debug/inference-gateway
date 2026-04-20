"""Base abstractions for inference backends.

Implements:
  - ModelBackend Protocol  (Strategy pattern)
  - BackendFactory         (Factory pattern)
  - BackendRegistry        (Singleton registry)

Design Patterns:
  - **Strategy**: ModelBackend defines the interface; each backend adapter
    provides its own implementation of complete/stream/health.
  - **Factory**: BackendFactory creates the right adapter from config type.
  - **Singleton**: BackendRegistry is the single source of truth for all
    initialized backends and their health status.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncIterator, Protocol, runtime_checkable

import httpx

from gateway.models.config_models import BackendConfig

if TYPE_CHECKING:
    from gateway.models.request import ChatRequest
    from gateway.models.response import ChatChunk, ChatResponse

logger = logging.getLogger(__name__)


# ── Strategy Pattern: Backend Protocol ──────────────────────


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol defining the interface all backends must implement.

    This is the Strategy pattern — the routing engine works with
    this interface and doesn't know which concrete backend it's using.
    """

    @property
    def name(self) -> str:
        """Unique backend identifier."""
        ...

    @property
    def backend_type(self) -> str:
        """Backend type (ollama, vllm, sglang, openai_compatible)."""
        ...

    @property
    def models(self) -> list[str]:
        """List of model identifiers this backend serves."""
        ...

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming completion request."""
        ...

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Send a streaming completion request, yielding chunks."""
        ...

    async def health(self) -> bool:
        """Check if the backend is healthy and reachable."""
        ...

    async def list_models(self) -> list[str]:
        """Return list of available model identifiers."""
        ...

    async def close(self) -> None:
        """Clean up resources (HTTP clients, connections)."""
        ...


# ── Template Method: Base HTTP Backend ──────────────────────


class BaseHTTPBackend:
    """Base class providing shared HTTP client logic for all backends.

    Implements the Template Method pattern — subclasses override
    specific methods while reusing the common HTTP plumbing.
    """

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._name = config.name
        self._type = config.type
        self._models = list(config.models)
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key
        self._health_endpoint = config.health_endpoint

        # Shared async HTTP client with connection pooling
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def backend_type(self) -> str:
        return self._type

    @property
    def models(self) -> list[str]:
        return self._models

    async def health(self) -> bool:
        """Default health check — GET the health endpoint."""
        try:
            resp = await self._client.get(self._health_endpoint, timeout=5.0)
            return resp.status_code < 500
        except (httpx.HTTPError, Exception):
            return False

    async def list_models(self) -> list[str]:
        """Return configured model list (override for dynamic discovery)."""
        return self._models

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self._name!r} url={self._base_url!r}>"


# ── Factory Pattern: Create backends from config ────────────


class BackendFactory:
    """Factory that creates the correct backend adapter from config type.

    Usage:
        backend = BackendFactory.create(backend_config)
    """

    _registry: dict[str, type[BaseHTTPBackend]] = {}

    @classmethod
    def register(cls, backend_type: str) -> callable:
        """Decorator to register a backend class for a given type.

        Example:
            @BackendFactory.register("ollama")
            class OllamaBackend(BaseHTTPBackend): ...
        """
        def decorator(backend_cls: type[BaseHTTPBackend]) -> type[BaseHTTPBackend]:
            cls._registry[backend_type] = backend_cls
            return backend_cls
        return decorator

    @classmethod
    def create(cls, config: BackendConfig) -> BaseHTTPBackend:
        """Create a backend instance from configuration."""
        backend_cls = cls._registry.get(config.type)
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend type: {config.type!r}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return backend_cls(config)


# ── Singleton: Backend Registry ─────────────────────────────


class BackendRegistry:
    """Singleton registry holding all initialized backend instances.

    Provides lookup by name, model, and health status tracking.
    Implements the Observer pattern — health checker updates status here,
    and the load balancer/router reads it.
    """

    def __init__(self) -> None:
        self._backends: dict[str, BaseHTTPBackend] = {}
        self._health_status: dict[str, bool] = {}
        self._model_to_backends: dict[str, list[str]] = {}

    def register(self, backend: BaseHTTPBackend) -> None:
        """Register a backend instance."""
        self._backends[backend.name] = backend
        self._health_status[backend.name] = True  # Assume healthy until checked

        # Build model → backend index
        for model in backend.models:
            if model not in self._model_to_backends:
                self._model_to_backends[model] = []
            self._model_to_backends[model].append(backend.name)

        logger.info(
            "Registered backend: %s (%s) with models: %s",
            backend.name, backend.backend_type, backend.models,
        )

    def get(self, name: str) -> BaseHTTPBackend | None:
        """Get a backend by name."""
        return self._backends.get(name)

    def get_backends_for_model(self, model: str) -> list[BaseHTTPBackend]:
        """Get all backends that serve a given model."""
        backend_names = self._model_to_backends.get(model, [])
        return [
            self._backends[name]
            for name in backend_names
            if name in self._backends
        ]

    def get_healthy_backends(self) -> list[BaseHTTPBackend]:
        """Get all currently healthy backends."""
        return [
            b for name, b in self._backends.items()
            if self._health_status.get(name, False)
        ]

    def update_health(self, name: str, healthy: bool) -> None:
        """Update the health status of a backend (Observer pattern)."""
        prev = self._health_status.get(name)
        self._health_status[name] = healthy
        if prev != healthy:
            status = "UP" if healthy else "DOWN"
            logger.warning("Backend %s is now %s", name, status)

    def is_healthy(self, name: str) -> bool:
        """Check if a backend is currently marked healthy."""
        return self._health_status.get(name, False)

    @property
    def all_backends(self) -> list[BaseHTTPBackend]:
        """Return all registered backends."""
        return list(self._backends.values())

    @property
    def all_names(self) -> list[str]:
        """Return all registered backend names."""
        return list(self._backends.keys())

    @property
    def health_status(self) -> dict[str, bool]:
        """Return a copy of the current health status map."""
        return dict(self._health_status)

    async def close_all(self) -> None:
        """Close all backend HTTP clients."""
        for backend in self._backends.values():
            await backend.close()
        self._backends.clear()
        self._health_status.clear()
        self._model_to_backends.clear()
