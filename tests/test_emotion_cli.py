"""CLI smoke tests for emotion workflow."""

from __future__ import annotations

from pathlib import Path

import facial_expression_detection.emotion_cli as emotion_cli


def test_emotion_parser_defaults() -> None:
    parser = emotion_cli.build_parser()
    args = parser.parse_args([])
    assert args.single_frame_test is False
    assert args.backend == "retinaface"
    assert args.data_dir.endswith("data")


def test_emotion_main_single_frame_uses_video_arg(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()

    captured: dict[str, object] = {}

    def fake_single_frame_emotion(**kwargs):
        captured.update(kwargs)
        return {"dominant_emotion": "happy"}

    monkeypatch.setattr(emotion_cli, "setup_gpu_logging", lambda: None)
    monkeypatch.setattr(emotion_cli, "single_frame_emotion", fake_single_frame_emotion)

    exit_code = emotion_cli.main(
        ["--single-frame-test", "--video", str(video_path), "--backend", "opencv"]
    )
    assert exit_code == 0
    assert captured["video_path"] == video_path
    assert captured["backend"] == "opencv"


def test_emotion_main_batch_invokes_export(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_csv = tmp_path / "emotion_results.csv"

    captured: dict[str, object] = {}

    def fake_run_emotion_export(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(emotion_cli, "setup_gpu_logging", lambda: None)
    monkeypatch.setattr(emotion_cli, "run_emotion_export", fake_run_emotion_export)

    exit_code = emotion_cli.main(
        ["--data-dir", str(data_dir), "--output-csv", str(output_csv)]
    )
    assert exit_code == 0
    assert captured["data_dir"] == data_dir
    assert captured["output_csv"] == output_csv
