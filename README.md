# 🚀 Inference Gateway

**Unified Multi-Model Inference Gateway** — Production-grade API gateway with OpenAI-compatible `/v1/chat/completions` across all backends.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com)
[![CI](https://github.com/gitgod-debug/inference-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/gitgod-debug/inference-gateway/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏛️ Architecture

```
Client Request (OpenAI-compatible)
    │
    ▼
┌──────────────────────────────────────┐
│         FastAPI Gateway              │
│  ┌────────────────────────────────┐  │
│  │     Middleware Pipeline        │  │
│  │  1. Request ID injection       │  │
│  │  2. Auth (API key → tenant)    │  │
│  │  3. Rate Limiter (per-tenant)  │  │
│  │  4. Request Logger (structlog) │  │
│  │  5. Metrics Collector (prom)   │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │     Router Engine              │  │
│  │  ├─ A/B Test Router            │  │
│  │  ├─ Canary Router              │  │
│  │  ├─ Weighted Load Balancer     │  │
│  │  └─ Fallback Chain             │  │
│  └────────────────────────────────┘  │
└──────────┬───────────────────────────┘
           │
    ┌──────┼──────┬──────────┐
    ▼      ▼      ▼          ▼
 ┌──────┐┌──────┐┌──────┐┌──────────────┐
 │ Groq ││Open- ││Gemini││ Any OpenAI-  │
 │(free)││Router││(free)││ compatible   │
 │      ││(free)││      ││ (BYOK)       │
 └──────┘└──────┘└──────┘└──────────────┘
```

## ✨ Features

- **OpenAI-Compatible API** — Drop-in replacement. Works with any OpenAI client.
- **Multi-Backend Support** — Groq, OpenRouter, Gemini, Ollama, vLLM, SGLang, and any OpenAI-compatible API.
- **Zero Cost** — Default backends are 100% free (Groq, OpenRouter, Google Gemini).
- **BYOK** — Users bring their own API keys for paid providers (OpenAI, Anthropic, etc.).
- **A/B Testing** — Weighted traffic splitting with sticky sessions.
- **Canary Deployments** — Gradual rollout with automatic rollback.
- **Circuit Breaker** — Prevents cascading failures with automatic recovery.
- **Fallback Chain** — Automatic failover across backends.
- **Rate Limiting** — Per-tenant token bucket rate limiter.
- **Multi-Tenant** — API key authentication with per-tenant rate limits.
- **SSE Streaming** — Full Server-Sent Events streaming support.
- **Prometheus Metrics** — Request rates, latencies, token usage, backend health.
- **Grafana Dashboard** — Auto-provisioned premium monitoring dashboard.
- **CLI** — `inference-gateway serve`, `health`, `models`, `config validate`.

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/harshit/inference-gateway.git
cd inference-gateway
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Add your API keys:
#   GROQ_API_KEY=       → https://console.groq.com
#   OPENROUTER_API_KEY= → https://openrouter.ai/keys
#   GEMINI_API_KEY=     → https://aistudio.google.com/apikey
```

### 3. Run

```bash
inference-gateway serve
```

### 4. Test

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer ig-demo-key-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 🐳 Docker

```bash
docker compose up -d          # Gateway + Prometheus + Grafana
docker compose --profile ollama up -d  # + Ollama backend
```

- **Gateway**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / gateway2026)

## 📁 Project Structure

```
inference-gateway/
├── src/gateway/
│   ├── app.py              # FastAPI app factory
│   ├── config.py            # Pydantic Settings
│   ├── cli.py               # Typer CLI
│   ├── models/              # Request/Response schemas
│   ├── backends/            # Backend adapters (Strategy pattern)
│   ├── routing/             # Router engine (Chain of Responsibility)
│   ├── middleware/          # Auth, Rate limit, Logging, Metrics
│   ├── streaming/           # SSE handler
│   └── health/              # Health checker, Circuit breaker
├── configs/                 # YAML configuration
├── monitoring/              # Prometheus + Grafana
├── tests/                   # Comprehensive test suite
└── benchmarks/              # Locust load tests
```

## 🧩 Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| Strategy | `ModelBackend` Protocol | Swap backends without changing routing logic |
| Factory | `BackendFactory` | Create backends from YAML config |
| Chain of Responsibility | Router pipeline | A/B → Canary → Model → Fallback |
| Circuit Breaker | `CircuitBreaker` | Prevent cascading failures |
| Observer | Health → Load balancer | React to health status changes |
| Singleton | `BackendRegistry` | Single source of truth for backends |
| Template Method | `BaseHTTPBackend` | Shared HTTP logic, override specifics |
| Builder | `create_app()` | Configurable application assembly |

## 📊 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Gateway health |
| GET | `/health/backends` | Per-backend health status |
| GET | `/metrics` | Prometheus metrics |

## 🧪 Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/ -v --cov=src/gateway  # With coverage
inference-gateway config validate    # Validate configs
```

## 📄 License

MIT
