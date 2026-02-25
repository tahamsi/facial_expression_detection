"""CLI smoke tests for detector sanity workflow."""

from __future__ import annotations

from pathlib import Path

import facial_expression_detection.detector_cli as detector_cli


def test_detector_parser_defaults() -> None:
    parser = detector_cli.build_parser()
    args = parser.parse_args([])
    assert args.samples_per_video == 3
    assert args.data_dir.endswith("data")


def test_detector_main_invokes_pipeline(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_csv = tmp_path / "sanity.csv"

    captured: dict[str, object] = {}

    def fake_run_detector_sanity_check(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        detector_cli,
        "run_detector_sanity_check",
        fake_run_detector_sanity_check,
    )

    exit_code = detector_cli.main(
        [
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(output_csv),
            "--samples-per-video",
            "5",
        ]
    )

    assert exit_code == 0
    assert captured["data_dir"] == data_dir
    assert captured["output_csv"] == output_csv
    assert captured["samples_per_video"] == 5
