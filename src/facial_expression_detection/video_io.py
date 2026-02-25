"""Video discovery helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .constants import ALLOWED_VIDEO_EXTENSIONS


def iter_video_paths(
    root_dir: Path,
    allowed_exts: Iterable[str] = ALLOWED_VIDEO_EXTENSIONS,
) -> Iterator[Path]:
    """Yield video files recursively in deterministic order."""
    normalized_exts = tuple(ext.lower() for ext in allowed_exts)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(normalized_exts):
                yield Path(dirpath) / filename
