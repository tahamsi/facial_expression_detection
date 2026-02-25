# Facial Expression Detection

Professionalized research utility for two reproducible workflows over local video data:

1. Emotion CSV export using DeepFace emotion analysis.
2. Face-detector backend sanity checks.

## What This Repository Does

- Scans videos recursively from a data directory.
- Runs DeepFace `analyze(..., actions=["emotion"])` to estimate dominant emotion.
- Compares detector backends (for example `opencv`, `mtcnn`, `yunet`) for robustness.
- Exports structured CSV outputs for downstream analysis.

## What This Repository Does NOT Do (Yet)

- No model training or fine-tuning.
- No identity recognition benchmarking.
- No causal or clinical interpretation of emotions.
- No fairness mitigation methods; this repo provides measurement utilities only.

## Repository Layout

```text
facial_expression_detection/
  src/facial_expression_detection/
  scripts/
    emotion_detection.py
    face_detection.py
  data/
  tests/
  .github/workflows/ci.yml
```

## Installation

Requires Python 3.10+.

### Core (CPU)

```bash
python -m pip install -e .
```

### Development tools

```bash
python -m pip install -e .[dev]
```

### Optional GPU runtime (Linux x86_64)

```bash
python -m pip install -e .[gpu]
```

### Optional dlib backend support

```bash
python -m pip install -e .[dlib]
```

A minimal `requirements.txt` is included for users who prefer requirements files.

## Data Setup

Place videos under `data/` (subfolders supported). Supported extensions:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.m4v`

Project hygiene defaults:

- `data/` content and common video extensions are git-ignored.
- Generated CSV outputs are git-ignored.

## Pipeline 1: Emotion CSV Export

Run:

```bash
python scripts/emotion_detection.py --data-dir data --output-csv emotion_results.csv
```

Optional single-frame diagnostic mode:

```bash
python scripts/emotion_detection.py --single-frame-test --video path/to/video.mp4 --backend retinaface --frame-index 0 --align
```

### DeepFace usage clarification

This pipeline uses DeepFace for **facial attribute emotion analysis**. It does **not** compare face-recognition identity models (VGG-Face, Facenet, ArcFace, etc.).

Comparison in this repo is between **detector backends**, not identity-recognition models.

### Emotion CSV columns

- `video_name`: filename only.
- `video_relpath`: path relative to `--data-dir`.
- `backend`: detector backend used by DeepFace.
- `emotion_analyzer`: fixed value `deepface_emotion_attribute_head`.
- `supported_model`: backward-compatibility alias of `emotion_analyzer`.
- `dominant_emotion`: dominant label from sampled frames.
- `sampled_frames`: number of frames sampled (~1 FPS).
- `analyzed_frames`: sampled frames with a valid dominant emotion output.

## Pipeline 2: Detector Backend Sanity Check

Run:

```bash
python scripts/face_detection.py --data-dir data --output-csv face_detection_sanity.csv
```

### Detector sanity CSV columns

- `video_name`: filename only.
- `video_relpath`: path relative to `--data-dir`.
- `backend`: detector backend.
- `frame_index`: sampled frame index.
- `faces_detected`: number of faces returned by DeepFace extraction.
- `success`: `true` when `faces_detected > 0`.

## Determinism and Reproducibility

- Recursive video discovery is sorted.
- Backend execution order is fixed.
- Emotion sampling uses fixed ~1 FPS intervals from video FPS metadata.
- Sanity-check frame indices are deterministic and evenly spaced.
- Emotion ties are resolved deterministically (count descending, label ascending).

## Migration Notes

Previous versions iterated over identity-recognition model names during emotion analysis. That behavior was conceptually misleading and has been removed.

- New behavior: one row per `(video, backend)` for emotion export.
- Compatibility aid: `supported_model` column is retained as a constant alias.

## Data / Ethics Note

You must ensure:

- participant consent for collection and processing,
- lawful basis for data use,
- appropriate licensing/permissions for all media,
- secure handling of personally identifiable recordings.

Do not commit real participant recordings to this repository.

## Limitations & Fairness Caveats

- Emotion recognition systems can be biased across demographics and cultures.
- Expressions and affect labels may not transfer across social contexts.
- Predicted labels are proxies, not ground-truth internal emotional states.
- Detector failure can silently reduce analyzed frames and skew aggregate results.

## Acknowledgement

Exeter Internal Fund - Towards Inclusive AI: Investigating Fairness in Emotion Recognition Algorithms Across Cultures

## Development Checks

```bash
ruff check .
black --check .
pytest
```

## License

MIT (see `LICENSE`).
