# Build either `transformers` (the default) or `sglang`. The extras pin different
# Torch/Transformers stacks, so each runtime gets its own image target.
# Base images are pinned by digest (tag kept in the comment); Dependabot's docker
# ecosystem proposes digest bumps. When bumping UV_VERSION, update the uv digest too.
ARG UV_VERSION=0.11.16

# ghcr.io/astral-sh/uv:0.11.16
FROM ghcr.io/astral-sh/uv:${UV_VERSION}@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d AS uv

# python:3.12-slim
FROM python@sha256:1eb6b7d4b76454b1de8317863ac3213b678c337b27e604a4e3fb70bddbb2bad7 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH="/opt/venv/bin:${PATH}"

COPY --from=uv /uv /usr/local/bin/uv

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
# nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04
FROM nvidia/cuda@sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd AS sglang-cuda-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && if ! getent group lrb >/dev/null; then groupadd --force --gid 1000 lrb; fi \
    && if ! id -u lrb >/dev/null 2>&1; then useradd --gid lrb --create-home --shell /usr/sbin/nologin lrb; fi \
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
