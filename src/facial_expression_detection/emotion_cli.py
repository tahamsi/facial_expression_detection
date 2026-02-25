"""CLI entrypoint for emotion CSV export."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .constants import DEFAULT_DATA_DIR, DEFAULT_EMOTION_OUTPUT
from .emotion_pipeline import run_emotion_export, single_frame_emotion
from .logging_utils import configure_logging
from .runtime import setup_gpu_logging
from .video_io import iter_video_paths

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for emotion workflow."""
    parser = argparse.ArgumentParser(
        description="Emotion analysis over videos using DeepFace emotion attributes."
    )
    parser.add_argument("--single-frame-test", action="store_true")
    parser.add_argument("--video", help="Path to a video for single-frame test")
    parser.add_argument("--backend", default="retinaface")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--align", action="store_true")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Root folder containing videos (default: ./data)",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_EMOTION_OUTPUT),
        help="CSV output path for batch emotion export",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def _resolve_single_frame_video(video_arg: str | None, data_dir: Path) -> Path:
    """Resolve single-frame input video path."""
    if video_arg:
        return Path(video_arg)

    videos = list(iter_video_paths(data_dir))
    if not videos:
        raise FileNotFoundError(f"No videos found under: {data_dir}")
    return videos[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Run emotion CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    setup_gpu_logging()

    data_dir = Path(args.data_dir)
    if args.single_frame_test:
        video_path = _resolve_single_frame_video(
            video_arg=args.video,
            data_dir=data_dir,
        )
        result = single_frame_emotion(
            video_path=video_path,
            backend=args.backend,
            frame_index=args.frame_index,
            align=args.align,
        )
        LOGGER.info(
            "Single-frame result for %s: %s",
            video_path,
            json.dumps(result, sort_keys=True, default=str),
        )
        return 0

    # Keep prior batch behavior: alignment is enabled for full exports.
    run_emotion_export(
        data_dir=data_dir,
        output_csv=Path(args.output_csv),
        align=True,
        show_progress=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
