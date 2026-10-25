from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    """Reject archive/API paths that could escape the destination directory."""
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    if path.parts and ":" in path.parts[0]:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "file"
