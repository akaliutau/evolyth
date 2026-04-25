from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable

from .rp import ResearchProblem

IGNORE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "runs", "data"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}


def create_workspace(
    *,
    arena_root: str | Path,
    source_rp: ResearchProblem,
    run_id: str,
    parent_model_path: str | Path | None = None,
) -> ResearchProblem:
    """Create an isolated RP workspace for one mutation/run.

    The canonical RP folder remains read-only from the evolver's point of view.
    Claude Code or another agent edits only this throwaway workspace.
    """
    root = Path(arena_root).expanduser().resolve()
    workspace = root / "workspaces" / run_id
    if workspace.exists():
        shutil.rmtree(workspace)

    shutil.copytree(
        source_rp.path,
        workspace,
        ignore=shutil.ignore_patterns(*IGNORE_DIRS, "*.pyc", "*.pyo"),
    )

    # Avoid copying large datasets when possible. If the copytree created a data
    # directory because it was small enough to copy, leave it alone. Otherwise,
    # symlink the source data directory for local runs.
    src_data = source_rp.path / "data"
    dst_data = workspace / "data"
    if src_data.exists() and not dst_data.exists():
        try:
            os.symlink(src_data, dst_data, target_is_directory=True)
        except OSError:
            # Symlinks may be unavailable on some platforms. The RP can still run
            # synthetic/dry-run modes or receive --data-dir explicitly.
            pass

    if parent_model_path:
        src = Path(parent_model_path)
        if src.exists():
            shutil.copy2(src, workspace / source_rp.mutable_file)

    return ResearchProblem.load(workspace)


def snapshot_hashes(root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    hashes: dict[str, str] = {}
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        hashes[rel] = _sha256(path)
    return hashes


def changed_files(before: dict[str, str], root: str | Path) -> list[str]:
    after = snapshot_hashes(root)
    keys = sorted(set(before) | set(after))
    return [k for k in keys if before.get(k) != after.get(k)]


def validate_only_allowed_changed(before: dict[str, str], root: str | Path, allowed: Iterable[str]) -> list[str]:
    allowed_set = {Path(p).as_posix() for p in allowed}
    changed = changed_files(before, root)
    illegal = [p for p in changed if p not in allowed_set]
    if illegal:
        raise RuntimeError(f"Illegal RP edits outside allowed files: {illegal}")
    return changed


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if parts & IGNORE_DIRS:
            continue
        if path.suffix in IGNORE_SUFFIXES:
            continue
        yield path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
