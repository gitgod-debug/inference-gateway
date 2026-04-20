# ============================================================
# Inference Gateway — Multi-stage Dockerfile
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Security: non-root user
RUN groupadd -r gateway && useradd -r -g gateway -d /app gateway

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
COPY src/ src/
COPY configs/ configs/

# Ownership
RUN chown -R gateway:gateway /app

USER gateway

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); assert r.status_code == 200"

EXPOSE 8080

ENTRYPOINT ["inference-gateway"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
