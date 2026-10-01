"""Portable archives of one evaluation's complete recorded evidence."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from cua_speedrun.service.evaluations import (
    EvaluationServiceError,
    TERMINAL_STAGES,
    get_evaluation,
)


_ALREADY_COMPRESSED = frozenset({
    ".bz2", ".gif", ".gz", ".jpeg", ".jpg", ".mp4", ".png", ".webp",
    ".xz", ".zip",
})


def evaluation_run_directory(
    session_factory,
    run_id: int,
    user_id: int,
    *,
    installation_root: Path,
) -> Path:
    """Resolve one owned, finished evaluation's artifact directory."""
    payload = get_evaluation(session_factory, run_id, user_id)
    if payload["stage"] not in TERMINAL_STAGES:
        raise EvaluationServiceError(
            409,
            f"evaluation {run_id} is still {payload['stage']}; "
            "export it after it stops",
        )
    stored = payload.get("run_dir")
    if not stored:
        raise EvaluationServiceError(409, f"evaluation {run_id} has no artifacts yet")
    run_dir = Path(stored).expanduser()
    if not run_dir.is_absolute():
        run_dir = installation_root / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise EvaluationServiceError(
            410, f"evaluation {run_id} artifacts are no longer available"
        )
    return run_dir


def write_run_archive(run_dir: Path, destination: Path) -> int:
    """Write all regular run artifacts under a single top-level directory.

    Symlinks are deliberately omitted: evaluation evidence is self-contained,
    and following a link could export data outside the run artifact boundary.
    Returns the number of archived files.
    """
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    try:
        destination.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "the export archive cannot be written inside the run directory"
        )

    files = []
    for root, directory_names, file_names in os.walk(run_dir, followlinks=False):
        directory = Path(root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not (directory / name).is_symlink()
        )
        files.extend(
            directory / name
            for name in sorted(file_names)
            if (directory / name).is_file()
            and not (directory / name).is_symlink()
        )
    if not files:
        raise ValueError(f"run artifact directory is empty: {run_dir}")

    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in files:
            archive.write(
                path,
                Path(run_dir.name) / path.relative_to(run_dir),
                compress_type=(
                    zipfile.ZIP_STORED
                    if path.suffix.lower() in _ALREADY_COMPRESSED
                    else zipfile.ZIP_DEFLATED
                ),
            )
    return len(files)


def remove_file(path: str | os.PathLike[str]) -> None:
    """Best-effort cleanup callback for a streamed temporary archive."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


__all__ = [
    "evaluation_run_directory",
    "remove_file",
    "write_run_archive",
]
