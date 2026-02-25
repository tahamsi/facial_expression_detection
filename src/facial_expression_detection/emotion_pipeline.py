"""Emotion-analysis pipeline used by the CLI wrapper."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .constants import DEFAULT_EMOTION_BACKENDS, EMOTION_ANALYZER_NAME
from .runtime import (
    is_backend_available,
    require_cv2,
    require_deepface,
    suppress_third_party_output,
)
from .video_io import iter_video_paths

LOGGER = logging.getLogger(__name__)

EMOTION_COLUMNS = [
    "video_name",
    "video_relpath",
    "backend",
    "emotion_analyzer",
    "supported_model",
    "dominant_emotion",
    "sampled_frames",
    "analyzed_frames",
]


def _normalize_analysis_result(analysis: Any) -> dict[str, Any]:
    """Normalize DeepFace return shape to a single dictionary."""
    if isinstance(analysis, list):
        if not analysis:
            return {}
        first = analysis[0]
        return first if isinstance(first, dict) else {}
    return analysis if isinstance(analysis, dict) else {}


def _select_dominant_emotion(emotion_counts: Counter[str]) -> str | None:
    """Return deterministic winner (frequency desc, emotion name asc)."""
    if not emotion_counts:
        return None
    return sorted(emotion_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def dominant_emotion_for_video(
    video_path: Path,
    backend: str,
    align: bool = True,
) -> tuple[str | None, int, int]:
    """Compute dominant emotion for one video/backend combination."""
    cv2 = require_cv2()
    deepface = require_deepface()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_interval = max(1, int(round(fps)))
    frame_index = 0

    sampled_frames = 0
    analyzed_frames = 0
    emotion_counts: Counter[str] = Counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            sampled_frames += 1
            try:
                with suppress_third_party_output():
                    analysis = deepface.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend=backend,
                        align=align,
                    )
                result = _normalize_analysis_result(analysis)
                dominant_emotion = result.get("dominant_emotion")
                if isinstance(dominant_emotion, str) and dominant_emotion:
                    emotion_counts[dominant_emotion] += 1
                    analyzed_frames += 1
            except Exception as exc:
                LOGGER.debug(
                    "Skipping frame %s for %s with backend %s: %s",
                    frame_index,
                    video_path,
                    backend,
                    exc,
                )

        frame_index += 1

    cap.release()
    return _select_dominant_emotion(emotion_counts), sampled_frames, analyzed_frames


def single_frame_emotion(
    video_path: Path,
    backend: str,
    frame_index: int = 0,
    align: bool = False,
) -> dict[str, Any]:
    """Analyze one frame and return the raw DeepFace emotion dictionary."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    cv2 = require_cv2()
    deepface = require_deepface()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    with suppress_third_party_output():
        analysis = deepface.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=backend,
            align=align,
        )
    return _normalize_analysis_result(analysis)


def run_emotion_export(
    data_dir: Path,
    output_csv: Path,
    backends: Sequence[str] = DEFAULT_EMOTION_BACKENDS,
    align: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run batch emotion export over all videos and detector backends."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    videos = list(iter_video_paths(data_dir))
    if not videos:
        LOGGER.warning("No videos found under: %s", data_dir)
        empty_df = pd.DataFrame(columns=EMOTION_COLUMNS)
        empty_df.to_csv(output_csv, index=False)
        return empty_df

    available_backends = [
        backend for backend in backends if is_backend_available(backend)
    ]
    unavailable_backends = [
        backend for backend in backends if backend not in available_backends
    ]
    if unavailable_backends:
        LOGGER.warning(
            "Skipping unavailable backends: %s",
            ", ".join(unavailable_backends),
        )

    if not available_backends:
        raise RuntimeError("No usable backends are available.")

    rows: list[dict[str, Any]] = []
    total = len(videos) * len(available_backends)
    with tqdm(
        total=total,
        desc="Emotion export",
        unit="combo",
        disable=not show_progress,
    ) as pbar:
        for video_path in videos:
            video_name = video_path.name
            video_relpath = str(video_path.relative_to(data_dir))
            for backend in available_backends:
                dominant_emotion, sampled_frames, analyzed_frames = (
                    dominant_emotion_for_video(
                        video_path=video_path,
                        backend=backend,
                        align=align,
                    )
                )
                rows.append(
                    {
                        "video_name": video_name,
                        "video_relpath": video_relpath,
                        "backend": backend,
                        "emotion_analyzer": EMOTION_ANALYZER_NAME,
                        # Backward-compatibility alias for downstream scripts.
                        "supported_model": EMOTION_ANALYZER_NAME,
                        "dominant_emotion": dominant_emotion,
                        "sampled_frames": sampled_frames,
                        "analyzed_frames": analyzed_frames,
                    }
                )
                pbar.update(1)

    frame = pd.DataFrame(rows, columns=EMOTION_COLUMNS)
    frame.to_csv(output_csv, index=False)
    LOGGER.info("Saved emotion results to %s", output_csv)
    return frame
