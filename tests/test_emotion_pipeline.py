"""Pipeline tests for emotion export."""

from __future__ import annotations

import contextlib
from collections import Counter
from pathlib import Path

import facial_expression_detection.emotion_pipeline as emotion_pipeline


def test_select_dominant_emotion_tie_breaker() -> None:
    counts = Counter({"happy": 2, "angry": 2, "sad": 1})
    assert emotion_pipeline._select_dominant_emotion(counts) == "angry"


def test_run_emotion_export_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for filename in ["b.mp4", "a.mp4"]:
        (data_dir / filename).touch()

    def fake_backend_available(backend: str) -> bool:
        return backend != "dlib"

    def fake_dominant(video_path: Path, backend: str, align: bool = True):
        del backend, align
        return ("happy" if video_path.name == "a.mp4" else "sad"), 3, 3

    monkeypatch.setattr(
        emotion_pipeline,
        "is_backend_available",
        fake_backend_available,
    )
    monkeypatch.setattr(emotion_pipeline, "dominant_emotion_for_video", fake_dominant)

    output_csv = tmp_path / "emotion.csv"
    frame = emotion_pipeline.run_emotion_export(
        data_dir=data_dir,
        output_csv=output_csv,
        backends=("opencv", "dlib"),
        show_progress=False,
    )

    assert output_csv.exists()
    assert frame["video_name"].tolist() == ["a.mp4", "b.mp4"]
    assert frame["backend"].tolist() == ["opencv", "opencv"]
    assert set(frame["supported_model"]) == {emotion_pipeline.EMOTION_ANALYZER_NAME}


def test_single_frame_emotion_uses_mocked_deepface(monkeypatch, tmp_path: Path) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self._released = False

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: int) -> None:
            del prop, value

        def read(self):
            return True, object()

        def release(self) -> None:
            self._released = True

    fake_capture = FakeCapture()

    class FakeCV2:
        CAP_PROP_POS_FRAMES = 1

        @staticmethod
        def VideoCapture(path: str):
            del path
            return fake_capture

    class FakeDeepFace:
        @staticmethod
        def analyze(*args, **kwargs):
            del args, kwargs
            return [{"dominant_emotion": "happy"}]

    monkeypatch.setattr(emotion_pipeline, "require_cv2", lambda: FakeCV2)
    monkeypatch.setattr(emotion_pipeline, "require_deepface", lambda: FakeDeepFace)
    monkeypatch.setattr(
        emotion_pipeline,
        "suppress_third_party_output",
        contextlib.nullcontext,
    )

    result = emotion_pipeline.single_frame_emotion(
        video_path=tmp_path / "clip.mp4",
        backend="opencv",
        frame_index=0,
        align=True,
    )
    assert result["dominant_emotion"] == "happy"
