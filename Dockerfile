# Packages the Streamlit app for a reproducible run. This is NOT the
# sandbox-isolation container ADR-001 named as a possible fallback - that
# ADR's decision (no whole-app container as a security boundary) is
# untouched. The one thing this solves is "works on my machine": a fresh
# clone needing uv, Python 3.13, and a manual `brew install libomp` for
# LightGBM (ADR-003's consequence) becomes `docker compose up` instead.
#
# The local qwen2.5-coder:14b model stays on the HOST via Ollama, not in
# this image or a sidecar container - Docker Desktop on macOS does not pass
# the GPU/Metal through to a container, so a 14b model would run CPU-only
# and be too slow to use. See docker-compose.yml for how the app reaches it.
FROM python:3.13-slim

# libgomp1: LightGBM's Linux wheel links against it at import time (the
# macOS/Homebrew equivalent, libomp, was ADR-003's own recorded gotcha).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first, so an unchanged pyproject.toml/uv.lock reuses this
# Docker layer even when application code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

EXPOSE 8501

# --server.address=0.0.0.0: Streamlit's own default only binds localhost
# inside the container, which the host could never reach.
CMD ["uv", "run", "streamlit", "run", "src/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
