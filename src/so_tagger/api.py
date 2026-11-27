"""HTTP API: any client gets a Suggestion list for a Title from a loaded Tagger.

``create_app`` builds the application from a registry of already loaded Taggers, so tests
inject fixture Taggers without touching the disk. ``load_taggers`` builds that registry from
Tagger directories at startup; the ``serve`` CLI command wires both together.
"""

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from so_tagger import __version__
from so_tagger.suggestion_policy import MAX_SUGGESTIONS, MIN_SUGGESTIONS
from so_tagger.taggers import DEV_VERSION, TAGGER_KINDS, Tagger, load_tagger
from so_tagger.taggers.base import METADATA_FILE

DEFAULT_TAGGER: Final = "linear"
MAX_TITLE_LENGTH = 300
"""Longest accepted Title, in characters after stripping. Real Titles are far shorter."""

TaggerKind = Literal["linear", "transformer"]
"""The known Tagger kinds; a test keeps it in sync with ``TAGGER_KINDS``."""

logger = logging.getLogger(__name__)

API_DESCRIPTION = f"""\
Suggests Tags for a StackOverflow Question from its Title.

`POST /suggest` returns a Suggestion list: {MIN_SUGGESTIONS} to {MAX_SUGGESTIONS} Supported tags
ranked by descending Score, where a Score is the estimated probability that the Tag applies.
Tags whose Score passes the Tagger's tuned Score cut-off are kept, never fewer than
{MIN_SUGGESTIONS} and never more than {MAX_SUGGESTIONS}.

`GET /health` tells which Taggers are loaded and which version of each is served.
"""


class SuggestRequest(BaseModel):
    """A Title to suggest Tags for, and the Tagger to ask."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"title": "How do I parse JSON in Python?", "tagger": "linear"}]
        }
    )

    title: str = Field(
        description=(
            "Title of the StackOverflow Question. Leading and trailing whitespace is removed; "
            f"the rest must hold 1 to {MAX_TITLE_LENGTH} characters."
        ),
        json_schema_extra={"minLength": 1, "maxLength": MAX_TITLE_LENGTH},
    )
    tagger: TaggerKind = Field(
        default=DEFAULT_TAGGER,
        description=f"Tagger to ask: {' or '.join(TAGGER_KINDS)}. Defaults to {DEFAULT_TAGGER}.",
    )

    @field_validator("title")
    @classmethod
    def _stripped_title_of_valid_length(cls, title: str) -> str:
        stripped = title.strip()
        if not stripped:
            raise ValueError("The Title must not be empty or only whitespace.")
        if len(stripped) > MAX_TITLE_LENGTH:
            raise ValueError(
                f"The Title must be at most {MAX_TITLE_LENGTH} characters long "
                f"(got {len(stripped)})."
            )
        return stripped


class SuggestionModel(BaseModel):
    """A Supported tag proposed for the Title, with its Score."""

    tag: str = Field(description="A Supported tag.", examples=["python"])
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated probability that the Tag applies to the Title, in [0, 1].",
        examples=[0.97],
    )


class SuggestResponse(BaseModel):
    """The Suggestion list for one Title, and the Tagger that produced it."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "tagger": "linear",
                    "version": "v0.3.0",
                    "suggestions": [
                        {"tag": "python", "score": 0.97},
                        {"tag": "json", "score": 0.91},
                    ],
                }
            ]
        }
    )

    tagger: TaggerKind = Field(description="Tagger that produced the Suggestions.")
    version: str = Field(
        description=(
            "Version of the Tagger that answered: the Release tag it was packaged for, "
            f"or '{DEV_VERSION}' outside a Release."
        )
    )
    suggestions: list[SuggestionModel] = Field(
        min_length=MIN_SUGGESTIONS,
        max_length=MAX_SUGGESTIONS,
        description=(
            f"{MIN_SUGGESTIONS} to {MAX_SUGGESTIONS} Suggestions ranked by descending Score."
        ),
    )


class TaggerStatus(BaseModel):
    """Whether one Tagger kind is loaded, and what it serves."""

    loaded: bool = Field(description="Whether this Tagger answers `POST /suggest`.")
    version: str | None = Field(
        description=(
            f"Version of the loaded Tagger: its Release tag, or '{DEV_VERSION}' outside a "
            "Release; null when not loaded."
        )
    )
    n_supported_tags: int = Field(
        ge=0, description="Number of Supported tags the Tagger can suggest; 0 when not loaded."
    )


class HealthResponse(BaseModel):
    """Service status: which Taggers are loaded."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "taggers": {
                        "linear": {"loaded": True, "version": "v0.3.0", "n_supported_tags": 86},
                        "transformer": {
                            "loaded": True,
                            "version": "v0.3.0",
                            "n_supported_tags": 86,
                        },
                    },
                }
            ]
        }
    )

    status: Literal["ok", "unavailable"] = Field(
        description=f"'ok' when the default Tagger ({DEFAULT_TAGGER}) is loaded."
    )
    taggers: dict[str, TaggerStatus] = Field(description="Status of every known Tagger kind.")


class ErrorResponse(BaseModel):
    detail: str = Field(description="What went wrong.")


def create_app(taggers: Mapping[str, Tagger]) -> FastAPI:
    """Build the API around already loaded Taggers, keyed by Tagger kind.

    A known kind missing from ``taggers`` is reported as not loaded: requests for it get a
    503. Each Tagger reports its own version, read from its metadata: the Release tag it was
    packaged for, or ``dev``.
    """
    unknown = set(taggers) - set(TAGGER_KINDS)
    if unknown:
        raise ValueError(f"Unknown Tagger kinds: {', '.join(sorted(unknown))}.")
    for kind, tagger in taggers.items():
        if tagger.kind != kind:
            raise ValueError(f"The Tagger registered as '{kind}' is a '{tagger.kind}' Tagger.")

    app = FastAPI(
        title="StackOverflow Tag Suggestion API",
        version=__version__,
        description=API_DESCRIPTION,
    )

    @app.post(
        "/suggest",
        response_model=SuggestResponse,
        summary="Suggest Tags for a Title",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ErrorResponse,
                "description": "The requested Tagger is known but not loaded on this server.",
            }
        },
    )
    def suggest(request: SuggestRequest) -> SuggestResponse:
        """Return the ranked Suggestion list for a Title from the requested Tagger."""
        start = time.perf_counter()
        tagger = taggers.get(request.tagger)
        if tagger is None:
            _log_request(request, None, start)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"The {request.tagger} Tagger is not loaded on this server.",
            )
        suggestions = tagger.suggest(request.title)
        _log_request(request, len(suggestions), start, tagger.metadata.version)
        return SuggestResponse(
            tagger=request.tagger,
            version=tagger.metadata.version,
            suggestions=[SuggestionModel(tag=s.tag, score=s.score) for s in suggestions],
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Report which Taggers are loaded",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": HealthResponse,
                "description": f"The default Tagger ({DEFAULT_TAGGER}) is not loaded.",
            }
        },
    )
    def health() -> JSONResponse:
        """Status of every known Tagger kind; 503 when the default Tagger is not loaded."""
        healthy = DEFAULT_TAGGER in taggers
        body = HealthResponse(
            status="ok" if healthy else "unavailable",
            taggers={kind: _tagger_status(taggers.get(kind)) for kind in TAGGER_KINDS},
        )
        return JSONResponse(
            body.model_dump(),
            status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return app


def _tagger_status(tagger: Tagger | None) -> TaggerStatus:
    if tagger is None:
        return TaggerStatus(loaded=False, version=None, n_supported_tags=0)
    return TaggerStatus(
        loaded=True, version=tagger.metadata.version, n_supported_tags=len(tagger.supported_tags)
    )


def _log_request(
    request: SuggestRequest, n_suggestions: int | None, start: float, version: str | None = None
) -> None:
    """One structured line per request, with the latency of the Tagger that answered.

    The Title itself is only logged at debug level.
    """
    fields = {
        "event": "suggest",
        "tagger": request.tagger,
        "version": version,
        "title_length": len(request.title),
        "n_suggestions": n_suggestions,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    logger.info(json.dumps(fields))
    logger.debug("Title: %s", request.title)


def load_taggers(directories: Mapping[str, Path]) -> dict[str, Tagger]:
    """Load the Tagger saved in each directory, keyed by Tagger kind.

    A directory without a saved Tagger means that kind is not loaded: it is skipped with a
    warning instead of stopping the service. A directory holding a broken Tagger, or one of
    another kind, still raises ``TaggerError``.
    """
    loaded: dict[str, Tagger] = {}
    for kind, directory in directories.items():
        if not (directory / METADATA_FILE).is_file():
            logger.warning("No %s Tagger in %s: it is not loaded.", kind, directory)
            continue
        loaded[kind] = load_tagger(directory, kind=kind)
        logger.info("Loaded the %s Tagger from %s.", kind, directory)
    return loaded
