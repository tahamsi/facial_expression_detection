"""Project constants used by both CLI pipelines."""

from __future__ import annotations

from pathlib import Path

ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

DEFAULT_DATA_DIR = Path.cwd() / "data"
DEFAULT_EMOTION_OUTPUT = Path("emotion_results.csv")
DEFAULT_DETECTOR_OUTPUT = Path("face_detection_sanity.csv")

# Emotion analysis compares detector backends only. DeepFace uses its internal
# emotion attribute head for the `actions=["emotion"]` analysis.
EMOTION_ANALYZER_NAME = "deepface_emotion_attribute_head"

DEFAULT_EMOTION_BACKENDS = (
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "yunet",
    "centerface",
)

DEFAULT_DETECTOR_BACKENDS = (
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yolov11s",
    "yolov11n",
    "yolov11m",
    "yunet",
    "centerface",
)

DEFAULT_SANITY_SAMPLES_PER_VIDEO = 3
