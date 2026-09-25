# Reproducible runtime for the benchmark. CPU-only by default; on a CUDA host,
# run with `--gpus all` and torch will pick the device up.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra inference --extra publish --no-dev

COPY data ./data

# The image carries no .git, so the build names its commit for every recorded result:
#   docker build --build-arg LRB_SOURCE_COMMIT=$(git rev-parse HEAD) .
ARG LRB_SOURCE_COMMIT=""
ENV LRB_SOURCE_COMMIT=${LRB_SOURCE_COMMIT}

ENV PATH="/opt/venv/bin:${PATH}"
ENTRYPOINT ["lrb"]
CMD ["--help"]
