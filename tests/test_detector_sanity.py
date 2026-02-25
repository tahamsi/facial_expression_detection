"""Unit tests for detector sanity pipeline."""

from __future__ import annotations

import contextlib
from pathlib import Path

import facial_expression_detection.detector_sanity as detector_sanity


def test_sample_frame_indices_even_spacing() -> None:
    indices = detector_sanity.sample_frame_indices(total_frames=12, samples=3)
    assert indices == [0, 4, 8]


def test_run_detector_sanity_check_with_mocks(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    video = data_dir / "subject.mp4"
    video.touch()

    fake_frame = object()

    monkeypatch.setattr(
        detector_sanity,
        "is_backend_available",
        lambda backend: backend == "opencv",
    )
    monkeypatch.setattr(
        detector_sanity,
        "sample_video_frames",
        lambda video_path, samples: [(0, fake_frame), (8, fake_frame)],
    )
    monkeypatch.setattr(
        detector_sanity,
        "detect_faces_in_frame",
        lambda frame, backend: 1,
    )

    output_csv = tmp_path / "sanity.csv"
    frame = detector_sanity.run_detector_sanity_check(
        data_dir=data_dir,
        output_csv=output_csv,
        backends=("opencv", "dlib"),
        samples_per_video=3,
        show_progress=False,
    )

    assert output_csv.exists()
    assert frame["backend"].tolist() == ["opencv", "opencv"]
    assert frame["faces_detected"].tolist() == [1, 1]
    assert frame["success"].tolist() == [True, True]


def test_detect_faces_in_frame_uses_mocked_deepface(monkeypatch) -> None:
    class FakeDeepFace:
        @staticmethod
        def extract_faces(*args, **kwargs):
            del args, kwargs
            return [{}, {}]

    monkeypatch.setattr(detector_sanity, "require_deepface", lambda: FakeDeepFace)
    monkeypatch.setattr(
        detector_sanity,
        "suppress_third_party_output",
        contextlib.nullcontext,
    )

    faces = detector_sanity.detect_faces_in_frame(frame=object(), backend="opencv")
    assert faces == 2
