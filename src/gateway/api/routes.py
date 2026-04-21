"""API route handlers for the inference gateway.

Endpoints:
  POST /v1/chat/completions  — Main inference endpoint
  GET  /v1/models            — List available models
  GET  /health               — Gateway health
  GET  /health/backends      — Per-backend health status

Production considerations:
  - All routes have try-except around every operation
  - Streaming errors record circuit breaker failures
  - Client disconnects during streaming are handled gracefully
  - No backend error details leaked in production error responses
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gateway.middleware.metrics import (
    ACTIVE_REQUESTS,
    record_request,
    record_tokens,
)
from gateway.models.request import ChatRequest  # noqa: TC001
from gateway.models.response import (
    ChatResponse,
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
)
from gateway.streaming.sse import create_sse_response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: Request, body: ChatRequest) -> Any:
    """OpenAI-compatible chat completion endpoint.

    Supports both streaming (SSE) and non-streaming responses.
    Routes requests through the middleware pipeline → router → backend.
    """
    # Guard: if startup failed, request_router may be None
    if not hasattr(request.app.state, "request_router") or request.app.state.request_router is None:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error={"message": "Gateway is starting up or misconfigured",
                       "type": "service_error", "code": "gateway_unavailable"}
            ).model_dump(),
        )

    app_state = request.app.state
    request_router = app_state.request_router
    fallback_chain = app_state.fallback_chain
    tenant = getattr(request.state, "tenant", None)
    request_id = getattr(request.state, "request_id", "")

    start_time = time.monotonic()
    backend = None

    try:
        # Resolve which backend to use
        backend = request_router.resolve(
            model=body.model, tenant=tenant, request_id=request_id,
        )
        ACTIVE_REQUESTS.labels(backend=backend.name).inc()

        if body.stream:
            # Streaming response via SSE
            chunk_iter = backend.stream(body)
            return await create_sse_response(
                request=request,
                chunk_iterator=chunk_iter,
                model=body.model,
                backend_name=backend.name,
                fallback_chain=fallback_chain,
            )
        else:
            # Non-streaming response
            response = await backend.complete(body)

            # Record metrics
            duration = time.monotonic() - start_time
            tenant_name = tenant.name if tenant else "anonymous"
            record_request(backend.name, body.model, 200, tenant_name, duration)

            if response.usage:
                record_tokens(
                    body.model, response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            # Record success for circuit breaker
            if fallback_chain:
                fallback_chain.record_result(backend.name, success=True)

            return response

    except ValueError as e:
        # No backend found
        duration = time.monotonic() - start_time
        tenant_name = tenant.name if tenant else "anonymous"
        record_request("none", body.model, 503, tenant_name, duration)
        logger.warning("No backend for model=%s tenant=%s: %s", body.model, tenant_name, e)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error={"message": str(e), "type": "backend_error", "code": "no_backend"}
            ).model_dump(),
        )
    except httpx.HTTPStatusError as e:
        # Backend returned an HTTP error (4xx/5xx)
        duration = time.monotonic() - start_time
        tenant_name = tenant.name if tenant else "anonymous"
        backend_name = backend.name if backend else "unknown"
        status_code = e.response.status_code

        record_request(backend_name, body.model, status_code, tenant_name, duration)
        if fallback_chain:
            fallback_chain.record_result(backend_name, success=False)

        logger.error("Backend %s HTTP %d: %s", backend_name, status_code,
                     e.response.text[:500] if e.response else "no body")

        return JSONResponse(
            status_code=502,
            content=ErrorResponse(
                error={"message": f"Backend returned HTTP {status_code}",
                       "type": "backend_error", "code": "backend_http_error",
                       "backend": backend_name}
            ).model_dump(),
        )
    except httpx.TimeoutException:
        # Backend timeout
        duration = time.monotonic() - start_time
        tenant_name = tenant.name if tenant else "anonymous"
        backend_name = backend.name if backend else "unknown"

        record_request(backend_name, body.model, 504, tenant_name, duration)
        if fallback_chain:
            fallback_chain.record_result(backend_name, success=False)

        logger.error("Backend %s timeout after %.1fs", backend_name, duration)

        return JSONResponse(
            status_code=504,
            content=ErrorResponse(
                error={"message": f"Backend timed out after {duration:.1f}s",
                       "type": "timeout_error", "code": "backend_timeout",
                       "backend": backend_name}
            ).model_dump(),
        )
    except httpx.HTTPError as e:
        # Network/connection error
        duration = time.monotonic() - start_time
        tenant_name = tenant.name if tenant else "anonymous"
        backend_name = backend.name if backend else "unknown"

        record_request(backend_name, body.model, 502, tenant_name, duration)
        if fallback_chain:
            fallback_chain.record_result(backend_name, success=False)

        logger.error("Backend %s connection error: %s", backend_name, e)

        return JSONResponse(
            status_code=502,
            content=ErrorResponse(
                error={"message": "Backend connection failed",
                       "type": "connection_error", "code": "backend_unreachable",
                       "backend": backend_name}
            ).model_dump(),
        )
    except Exception as e:
        # Catch-all for any unexpected error
        duration = time.monotonic() - start_time
        tenant_name = tenant.name if tenant else "anonymous"
        backend_name = backend.name if backend else "unknown"

        record_request(backend_name, body.model, 500, tenant_name, duration)
        if fallback_chain:
            fallback_chain.record_result(backend_name, success=False)

        logger.error("Unexpected error in chat_completions: %s", e, exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error={"message": "Internal gateway error",
                       "type": "internal_error", "code": "unexpected_error"}
            ).model_dump(),
        )
    finally:
        if backend:
            ACTIVE_REQUESTS.labels(backend=backend.name).dec()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """List all available models across all backends."""
    try:
        registry = request.app.state.backend_registry
        models = []
        for backend in registry.all_backends:
            if registry.is_healthy(backend.name):
                for model_id in backend.models:
                    models.append(ModelInfo(
                        id=model_id, owned_by=backend.name, backend=backend.name,
                    ))
        return ModelListResponse(data=models)
    except Exception as e:
        logger.error("Error listing models: %s", e, exc_info=True)
        return ModelListResponse(data=[])


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Gateway health check."""
    try:
        registry = request.app.state.backend_registry
        healthy_count = len(registry.get_healthy_backends())
        total_count = len(registry.all_backends)
        status = "healthy" if healthy_count > 0 else "degraded"
        return {
            "status": status,
            "version": request.app.version,
            "backends": {"healthy": healthy_count, "total": total_count},
        }
    except Exception as e:
        logger.error("Health check error: %s", e)
        return {"status": "error", "backends": {"healthy": 0, "total": 0}}


@router.get("/health/backends")
async def backend_health(request: Request) -> dict:
    """Detailed per-backend health status."""
    try:
        registry = request.app.state.backend_registry
        cb_manager = request.app.state.circuit_breaker_manager

        backends = []
        for backend in registry.all_backends:
            backends.append({
                "name": backend.name,
                "type": backend.backend_type,
                "healthy": registry.is_healthy(backend.name),
                "circuit_breaker": cb_manager.states.get(backend.name, "closed"),
                "models": backend.models,
            })

        return {"backends": backends}
    except Exception as e:
        logger.error("Backend health check error: %s", e)
        return {"backends": []}
