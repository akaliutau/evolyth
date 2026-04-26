from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Iterable


SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def load_dotenv_files(paths: Iterable[str | Path], *, override: bool = False) -> list[Path]:
    """Load KEY=VALUE pairs from .env files into os.environ.

    Existing variables win by default so a shell/CI secret cannot be accidentally
    replaced by a project file. Empty existing values are treated as unset unless
    override=True is passed; this lets a real .env key fix a blank inherited var.
    """
    loaded: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        _load_one(path, override=override)
        loaded.append(path)
    return loaded


def default_dotenv_paths(*, cli_file: str | Path, cwd: str | Path | None = None, rp_path: str | Path | None = None) -> list[Path]:
    """Return .env candidates in least-to-most-local order.

    Later files can fill keys that earlier files did not set. Because override is
    normally false, already exported shell variables still take precedence.
    """
    paths: list[Path] = []
    cli_dir = Path(cli_file).resolve().parent
    paths.append(cli_dir / ".env")
    if cwd is not None:
        paths.append(Path(cwd).expanduser().resolve() / ".env")
    if rp_path is not None:
        paths.append(Path(rp_path).expanduser().resolve() / ".env")
    return paths


def redacted_env(keys: Iterable[str] | None = None) -> dict[str, str]:
    """Small diagnostic helper for logs/tests; never returns secret values."""
    selected = keys or os.environ.keys()
    out: dict[str, str] = {}
    for key in selected:
        if key in os.environ:
            out[key] = "<set>" if _looks_sensitive(key) else os.environ[key]
    return out


def _load_one(path: Path, *, override: bool) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw)
        if parsed is None:
            continue
        key, value = parsed
        existing = os.environ.get(key)
        if override or existing is None or existing == "":
            os.environ[key] = value


def _parse_dotenv_line(raw: str) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    value = value.strip()
    if value and value[0] in {"'", '"'}:
        try:
            parts = shlex.split(value, comments=False, posix=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = value.strip('"').strip("'")
    else:
        # Keep inline # when escaped or part of a token simple; this is enough
        # for normal API-key .env files without pulling in python-dotenv.
        hash_index = value.find(" #")
        if hash_index >= 0:
            value = value[:hash_index].rstrip()
    return key, value


def _looks_sensitive(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SENSITIVE_ENV_MARKERS)
