"""Logging setup helpers."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool) -> None:
    """Configure process-wide logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
        force=True,
    )
