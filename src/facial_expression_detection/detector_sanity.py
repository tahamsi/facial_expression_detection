"""Face-detector backend sanity-check pipeline."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .constants import DEFAULT_DETECTOR_BACKENDS, DEFAULT_SANITY_SAMPLES_PER_VIDEO
from .runtime import (
    is_backend_available,
    require_cv2,
    require_deepface,
    suppress_third_party_output,
)
from .video_io import iter_video_paths

LOGGER = logging.getLogger(__name__)

DETECTOR_COLUMNS = [
    "video_name",
    "video_relpath",
    "backend",
    "frame_index",
    "faces_detected",
    "success",
]


def sample_frame_indices(total_frames: int, samples: int) -> list[int]:
    """Return deterministic frame indices spaced through the video."""
    if samples < 1:
        raise ValueError("samples must be greater than 0")

    normalized_total = max(1, total_frames)
    if samples == 1:
        return [0]

    step = max(1, normalized_total // samples)
    return [min(index * step, normalized_total - 1) for index in range(samples)]


def sample_video_frames(video_path: Path, samples: int) -> list[tuple[int, Any]]:
    """Extract up to `samples` frames from a video."""
    cv2 = require_cv2()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_frame_indices(total_frames=total_frames, samples=samples)

    frames: list[tuple[int, Any]] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((index, frame))

    cap.release()
    return frames


def detect_faces_in_frame(frame: Any, backend: str) -> int:
    """Return number of faces found in one frame for one backend."""
    deepface = require_deepface()
    try:
        with suppress_third_party_output():
            faces = deepface.extract_faces(
                img_path=frame,
                detector_backend=backend,
                enforce_detection=False,
            )
        return len(faces)
    except Exception as exc:
        LOGGER.debug("Face detection failed for backend %s: %s", backend, exc)
        return 0


def run_detector_sanity_check(
    data_dir: Path,
    output_csv: Path,
    backends: Sequence[str] = DEFAULT_DETECTOR_BACKENDS,
    samples_per_video: int = DEFAULT_SANITY_SAMPLES_PER_VIDEO,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run backend-only face-detection sanity check and export CSV."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    videos = list(iter_video_paths(data_dir))
    if not videos:
        LOGGER.warning("No videos found under: %s", data_dir)
        empty_df = pd.DataFrame(columns=DETECTOR_COLUMNS)
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
    total = len(videos) * len(available_backends) * samples_per_video

    with tqdm(
        total=total,
        desc="Detector sanity",
        unit="frame",
        disable=not show_progress,
    ) as pbar:
        for video_path in videos:
            video_name = video_path.name
            video_relpath = str(video_path.relative_to(data_dir))
            frames = sample_video_frames(
                video_path=video_path,
                samples=samples_per_video,
            )

            if not frames:
                for backend in available_backends:
                    rows.append(
                        {
                            "video_name": video_name,
                            "video_relpath": video_relpath,
                            "backend": backend,
                            "frame_index": None,
                            "faces_detected": 0,
                            "success": False,
                        }
                    )
                    pbar.update(samples_per_video)
                continue

            for backend in available_backends:
                for frame_index, frame in frames:
                    faces_detected = detect_faces_in_frame(frame=frame, backend=backend)
                    rows.append(
                        {
                            "video_name": video_name,
                            "video_relpath": video_relpath,
                            "backend": backend,
                            "frame_index": frame_index,
                            "faces_detected": faces_detected,
                            "success": faces_detected > 0,
                        }
                    )
                    pbar.update(1)

                if len(frames) < samples_per_video:
                    pbar.update(samples_per_video - len(frames))

    frame = pd.DataFrame(rows, columns=DETECTOR_COLUMNS)
    frame.to_csv(output_csv, index=False)
    LOGGER.info("Saved detector sanity results to %s", output_csv)
    return frame
