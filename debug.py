"""Optional debug artifacts shared by all agents in a research run."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel

import config

logger = logging.getLogger(__name__)
_run_directory: ContextVar[Path | None] = ContextVar("research_run_directory", default=None)


# --- Artifact writing ---
# Models and provider objects -> JSON -> atomic file replacement


def _json_default(value):
    """Convert models, datetimes, and SDK objects for the artifact JSON encoder.

    Args:
        value: Object not handled directly by the standard JSON encoder.

    Returns:
        A dictionary, ISO datetime string, or fallback string representation.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, datetime):
        return value.isoformat()
    return vars(value) if hasattr(value, "__dict__") else str(value)


def save_output(filename: str, output) -> Path | None:
    """Write an artifact without changing the research outcome on failure.

    Args:
        filename: Artifact path relative to the active run directory.
        output: Result or provider data to serialize as JSON.

    Returns:
        The written path, or None when capture is inactive or saving fails.
    """
    directory = _run_directory.get()
    if directory is None:
        return None
    path = directory / filename
    try:
        # Replace the destination only after serialization and writing succeed.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        temporary.replace(path)
        return path
    except (OSError, TypeError, ValueError) as exc:
        # Debug storage failures must not interrupt research.
        logger.warning("Could not save %s (%s)", path, exc)
        return None


# --- Run capture ---
# Share a directory -> retain outputs and failures -> finalize the root run


@contextmanager
def capture_run(question: str, raw: dict, output_file: str):
    """Share one artifact folder across nested agents and retain failed runs.

    Exceptions and cancellations from the wrapped run propagate after capture.

    Args:
        question: Research question recorded in run metadata.
        raw: Mutable provider-response store to flush when the root run exits.
        output_file: Relative artifact path used to record a run failure.

    Yields:
        The shared run directory, or None when saving is disabled.
    """
    directory = _run_directory.get()
    root = directory is None
    # Saving is optional; nested agents reuse an existing capture directory.
    if root and not config.SAVE_RUNS:
        yield None
        return

    started = datetime.now(UTC)
    timer = perf_counter()
    token = None
    if root:
        # Only the outermost capture creates and owns the run directory.
        identifier = config.run_id.get()
        if identifier == "-":
            identifier = uuid4().hex[:12]
        directory = config.RUNS_DIR / f"{started:%Y%m%dT%H%M%S%fZ}_{identifier}"
        token = _run_directory.set(directory)
    record = {"question": question, "started_at": started.isoformat(), "status": "success",
              "model": config.MODEL_CONFIG, "run_id": directory.name.rsplit("_", 1)[-1]}
    try:
        yield directory
    except BaseException as exc:
        # Retain failures and cancellations before propagating them.
        record.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        save_output(output_file, {"status": "error", "error": record["error"]})
        raise
    finally:
        if root:
            # Flush available raw responses and restore context even if saving fails.
            try:
                record.update(finished_at=datetime.now(UTC).isoformat(), elapsed_seconds=perf_counter() - timer)
                save_output("raw.json", raw)
                if save_output("run.json", record):
                    logger.info("Research run saved to %s", directory)
            finally:
                _run_directory.reset(token)
