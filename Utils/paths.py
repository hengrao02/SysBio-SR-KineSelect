from __future__ import annotations

from pathlib import Path
import re
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATH_PATTERN = re.compile(r"/(?:data|home|tmp)/[^\s\"'),\]]+")


def repository_relative(path: str | Path | None) -> str | None:
    """Return a stable repository-relative label without leaking local roots."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def sanitize_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return repository_relative(value)
    if isinstance(value, dict):
        return {str(key): sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_paths(item) for item in value]
    if isinstance(value, str):
        text = value.replace(str(REPOSITORY_ROOT), "<repo>")
        return _LOCAL_PATH_PATTERN.sub("<local-path>", text)
    return value
