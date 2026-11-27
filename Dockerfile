# syntax=docker/dockerfile:1

# The API image: the runtime dependencies (with CPU-only PyTorch), the so-tagger package and
# both Taggers of one Release, downloaded at build time. The image is immutable, tied to that
# Release, and never reaches the network at runtime.
#
#   docker build --build-arg RELEASE_TAG=v0.3.0 -t so-tagger:v0.3.0 .

FROM python:3.12-slim-trixie AS base

FROM ghcr.io/astral-sh/uv:0.13.1 AS uv

# --- Install the runtime dependencies and the package into a virtual environment ------------
FROM base AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# Dependencies first, in their own layer: it is reused as long as uv.lock does not change.
# --extra transformer adds torch and transformers for the Transformer tagger. uv.lock pins torch
# to the PyTorch CPU wheel index (pyproject.toml, [tool.uv.sources]): no CUDA wheel is ever
# installed. --no-dev leaves out the dev group; the docs and explore groups are never installed.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra transformer --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra transformer --no-editable \
    && test -z "$(find /app/.venv \( -name 'nvidia*' -o -name 'libcudart*' \) -print -quit)"
# Trim what serving never runs (about 280 MB): the third-party test suites, the C++ headers and
# test binaries of torch. torch_shm_manager is kept: torch looks for it at import time.
RUN site=/app/.venv/lib/python3.12/site-packages \
    && find "$site" -depth -type d -name tests -exec rm -rf {} + \
    && rm -rf "$site/torch/test" "$site/torch/include" \
    && find "$site/torch/bin" -mindepth 1 -maxdepth 1 ! -name torch_shm_manager -exec rm -rf {} +

# --- Download and extract the Release assets --------------------------------------------------
FROM base AS taggers
ARG RELEASE_TAG
ARG REPOSITORY=thomas-hinton/stackoverflow-tag-prediction
# The repository is public: its Release assets download over plain HTTPS, without a token.
ARG RELEASE_BASE_URL=https://github.com/${REPOSITORY}/releases/download
# ADD fails the build with the URL when an asset is missing from the Release (HTTP 404).
# Files are made world-readable: save_pretrained writes model.safetensors as 0600, which the
# non-root runtime user could not read.
ADD ${RELEASE_BASE_URL}/${RELEASE_TAG}/linear-tagger.tar.gz /tmp/linear-tagger.tar.gz
ADD ${RELEASE_BASE_URL}/${RELEASE_TAG}/transformer-tagger.tar.gz /tmp/transformer-tagger.tar.gz
RUN test -n "${RELEASE_TAG}" || { echo "RELEASE_TAG build argument is required." >&2; exit 1; } \
    && for kind in linear transformer; do \
         mkdir -p "/taggers/${kind}" \
         && tar -xzf "/tmp/${kind}-tagger.tar.gz" -C "/taggers/${kind}" --no-same-owner \
         && test -f "/taggers/${kind}/metadata.json" \
         || { echo "${kind}-tagger.tar.gz of ${RELEASE_TAG} holds no Tagger." >&2; exit 1; }; \
       done \
    && test -f /taggers/transformer/model/config.json \
    && chmod -R a+rX,go-w /taggers

# --- Runtime -----------------------------------------------------------------------------------
FROM base AS runtime
ARG RELEASE_TAG
ARG REPOSITORY=thomas-hinton/stackoverflow-tag-prediction
ARG COMMIT=unknown
LABEL org.opencontainers.image.title="so-tagger" \
      org.opencontainers.image.description="Suggests Tags for a StackOverflow Question from its Title." \
      org.opencontainers.image.source="https://github.com/${REPOSITORY}" \
      org.opencontainers.image.version="${RELEASE_TAG}" \
      org.opencontainers.image.revision="${COMMIT}"

RUN groupadd --system --gid 10001 tagger \
    && useradd --system --uid 10001 --gid tagger --no-create-home --shell /usr/sbin/nologin tagger

COPY --from=builder /app/.venv /app/.venv
COPY --from=taggers /taggers /app/taggers

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SO_TAGGER_HOST=0.0.0.0 \
    SO_TAGGER_PORT=8000 \
    SO_TAGGER_LINEAR_DIR=/app/taggers/linear \
    SO_TAGGER_TRANSFORMER_DIR=/app/taggers/transformer \
    SO_TAGGER_TORCH_THREADS=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/huggingface

WORKDIR /app
USER tagger:tagger
EXPOSE 8000

# 200 only when the Linear tagger is loaded; urlopen raises on any other status. Both Taggers
# load before the server listens, so the start period covers loading the Transformer tagger.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['SO_TAGGER_PORT']}/health\", timeout=4)"]

ENTRYPOINT ["so-tagger", "serve"]
