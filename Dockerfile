# syntax=docker/dockerfile:1

# The API image: the runtime dependencies, the so-tagger package and the Tagger(s) of one
# Release, downloaded at build time. The image is immutable and tied to that Release.
#
#   docker build --build-arg RELEASE_TAG=v0.1.0 -t so-tagger:v0.1.0 .

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
# --no-dev leaves out the dev group; the explore group and the transformer extra are never
# installed unless asked for.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- Download and extract the Release assets --------------------------------------------------
FROM base AS taggers
ARG RELEASE_TAG
ARG REPOSITORY=thomas-hinton/stackoverflow-tag-prediction
# The repository is public: its Release assets download over plain HTTPS, without a token.
ARG RELEASE_BASE_URL=https://github.com/${REPOSITORY}/releases/download
ADD ${RELEASE_BASE_URL}/${RELEASE_TAG}/linear-tagger.tar.gz /tmp/linear-tagger.tar.gz
RUN test -n "${RELEASE_TAG}" \
    && mkdir -p /taggers/linear \
    && tar -xzf /tmp/linear-tagger.tar.gz -C /taggers/linear --no-same-owner \
    && test -f /taggers/linear/metadata.json

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
    SO_TAGGER_LINEAR_DIR=/app/taggers/linear

WORKDIR /app
USER tagger:tagger
EXPOSE 8000

# 200 only when the Linear tagger is loaded; urlopen raises on any other status.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['SO_TAGGER_PORT']}/health\", timeout=4)"]

ENTRYPOINT ["so-tagger", "serve"]
