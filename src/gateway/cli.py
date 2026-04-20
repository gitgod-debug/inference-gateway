"""CLI interface using Typer.

Commands:
  inference-gateway serve             — Start the server
  inference-gateway health            — Check backend health
  inference-gateway config validate   — Validate YAML configs
  inference-gateway models            — List available models

Production note: Uses typer.echo for user-facing output (CLI tools should
print to stdout for human consumption). Internal errors use logging.
"""

from __future__ import annotations

import logging
import sys

import typer
import uvicorn

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="inference-gateway",
    help="🚀 Unified Multi-Model Inference Gateway",
    add_completion=False,
)

config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8080, help="Port number"),
    workers: int = typer.Option(1, help="Number of workers"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    log_level: str = typer.Option("info", help="Log level"),
) -> None:
    """Start the Inference Gateway server."""
    import os
    os.environ["GATEWAY_HOST"] = host
    os.environ["GATEWAY_PORT"] = str(port)
    os.environ["GATEWAY_LOG_LEVEL"] = log_level

    typer.echo(f"🚀 Starting Inference Gateway on {host}:{port}")

    try:
        uvicorn.run(
            "gateway.app:create_app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level,
            factory=True,
            # ── Production safety ──
            timeout_graceful_shutdown=10,  # Drain in-flight requests on SIGTERM (10s)
            timeout_keep_alive=5,          # Close idle keep-alive connections after 5s
            limit_max_requests=10000,      # Restart worker after 10K requests (memory leak safety)
            h11_max_incomplete_event_size=1048576,  # Max 1MB for headers (prevent header injection)
        )
    except KeyboardInterrupt:
        typer.echo("\n🛑 Gateway stopped")
    except Exception as e:
        logger.error("Server failed to start: %s", e, exc_info=True)
        typer.echo(f"❌ Server failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def health() -> None:
    """Check health of all configured backends."""
    import asyncio
    from gateway.config import get_settings, load_yaml_config
    from gateway.backends.base import BackendFactory, BackendRegistry
    from gateway.models.config_models import RoutesConfig

    # Import backends to register them
    import gateway.backends.ollama  # noqa: F401
    import gateway.backends.vllm  # noqa: F401
    import gateway.backends.sglang  # noqa: F401
    import gateway.backends.openai_compatible  # noqa: F401

    async def _check() -> None:
        settings = get_settings()
        try:
            routes_data = load_yaml_config(settings.resolve_config_path(settings.routes_config))
            routes_config = RoutesConfig(**routes_data)
        except Exception as e:
            typer.echo(f"❌ Failed to load routes config: {e}", err=True)
            raise typer.Exit(code=1)

        registry = BackendRegistry()
        for cfg in routes_config.backends:
            try:
                backend = BackendFactory.create(cfg)
                registry.register(backend)
            except Exception as e:
                typer.echo(f"  ❌ {cfg.name}: failed to init ({e})")

        typer.echo("🏥 Backend Health Check\n")
        for backend in registry.all_backends:
            try:
                healthy = await backend.health()
                icon = "✅" if healthy else "❌"
                typer.echo(f"  {icon} {backend.name} ({backend.backend_type}) → {backend._base_url}")
            except Exception as e:
                typer.echo(f"  ❌ {backend.name}: error ({e})")

        await registry.close_all()

    try:
        asyncio.run(_check())
    except Exception as e:
        logger.error("Health check command failed: %s", e, exc_info=True)
        typer.echo(f"❌ Health check failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def models() -> None:
    """List all available models across backends."""
    from gateway.config import get_settings, load_yaml_config
    from gateway.models.config_models import RoutesConfig

    try:
        settings = get_settings()
        routes_data = load_yaml_config(settings.resolve_config_path(settings.routes_config))
        routes_config = RoutesConfig(**routes_data)
    except Exception as e:
        typer.echo(f"❌ Failed to load config: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("📋 Configured Models\n")
    for backend in routes_config.backends:
        typer.echo(f"  {backend.name} ({backend.type}):")
        for model in backend.models:
            typer.echo(f"    • {model}")
        typer.echo()


@config_app.command("validate")
def config_validate() -> None:
    """Validate all YAML configuration files."""
    from gateway.config import get_settings, load_yaml_config
    from gateway.models.config_models import ABTestsConfig, RoutesConfig, TenantsConfig

    settings = get_settings()
    errors = []
    passed = 0

    # Validate routes
    try:
        data = load_yaml_config(settings.resolve_config_path(settings.routes_config))
        RoutesConfig(**data)
        typer.echo(f"  ✅ {settings.routes_config}")
        passed += 1
    except Exception as e:
        errors.append(f"  ❌ {settings.routes_config}: {e}")

    # Validate A/B tests
    try:
        data = load_yaml_config(settings.resolve_config_path(settings.ab_tests_config))
        ABTestsConfig(**data)
        typer.echo(f"  ✅ {settings.ab_tests_config}")
        passed += 1
    except Exception as e:
        errors.append(f"  ❌ {settings.ab_tests_config}: {e}")

    # Validate tenants
    try:
        data = load_yaml_config(settings.resolve_config_path(settings.tenants_config))
        TenantsConfig(**data)
        typer.echo(f"  ✅ {settings.tenants_config}")
        passed += 1
    except Exception as e:
        errors.append(f"  ❌ {settings.tenants_config}: {e}")

    if errors:
        typer.echo(f"\n❌ {len(errors)} validation error(s):")
        for err in errors:
            typer.echo(err)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"\n✅ All {passed} configs valid!")


if __name__ == "__main__":
    app()
