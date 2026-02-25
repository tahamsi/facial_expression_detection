"""CLI entrypoint for face-detector backend sanity checks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_DETECTOR_OUTPUT,
    DEFAULT_SANITY_SAMPLES_PER_VIDEO,
)
from .detector_sanity import run_detector_sanity_check
from .logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for backend sanity-check workflow."""
    parser = argparse.ArgumentParser(description="Face-detection backend sanity check")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Root folder containing videos (default: ./data)",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_DETECTOR_OUTPUT),
        help="CSV output path for backend sanity check",
    )
    parser.add_argument(
        "--samples-per-video",
        type=int,
        default=DEFAULT_SANITY_SAMPLES_PER_VIDEO,
        help="How many frames to sample from each video",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run backend sanity-check CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    run_detector_sanity_check(
        data_dir=Path(args.data_dir),
        output_csv=Path(args.output_csv),
        samples_per_video=args.samples_per_video,
        show_progress=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
