# Build either `transformers` (the default) or `sglang`. The extras pin different
# Torch/Transformers stacks, so each runtime gets its own image target.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH="/opt/venv/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

RUN groupadd --gid 1000 lrb \
    && useradd --uid 1000 --gid lrb --create-home --shell /usr/sbin/nologin lrb \
    && mkdir -p /cache/huggingface /app/results \
    && chown -R lrb:lrb /cache/huggingface /app/results

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data

# A commit SHA is provenance, not a credential. Pass credentials at container runtime.
ARG LRB_SOURCE_COMMIT=""
ENV LRB_SOURCE_COMMIT=${LRB_SOURCE_COMMIT}

VOLUME ["/cache/huggingface", "/app/results"]

# SGLang's CUDA kernels need a CUDA development toolchain at runtime for JIT builds.
# Keep its CUDA/PyTorch stack isolated from the Transformers image above.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 AS sglang-cuda-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 lrb \
    && useradd --uid 1000 --gid lrb --create-home --shell /usr/sbin/nologin lrb \
    && mkdir -p /cache/huggingface /app/results \
    && chown -R lrb:lrb /cache/huggingface /app/results

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
ARG LRB_SOURCE_COMMIT=""
ENV LRB_SOURCE_COMMIT=${LRB_SOURCE_COMMIT}
VOLUME ["/cache/huggingface", "/app/results"]

FROM sglang-cuda-base AS sglang
RUN uv sync --frozen --extra speculative --extra publish --no-dev
USER lrb
ENTRYPOINT ["lrb"]
CMD ["--help"]

# Keep this as the final stage so `docker build .` selects the Transformers runtime.
FROM base AS transformers
RUN uv sync --frozen --extra inference --extra publish --no-dev
USER lrb
ENTRYPOINT ["lrb"]
CMD ["--help"]
