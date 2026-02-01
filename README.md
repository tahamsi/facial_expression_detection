# Face Recognition Pipeline (Emotion + Face-Detection Sanity Check)

This project runs two workflows on your local video dataset:

1) **Emotion detection across multiple backends/models**
2) **Face-detection sanity check per backend**

Both scripts are in `scripts/` and read videos from the `data/` folder by default (override with `--data-dir`).

---

## Folder layout

```
face-recognition/
  data/                    # Place your videos here (recursively searched)
  scripts/
    emotion_detection.py    # Dominant emotion per video, backend, model
    face_detection.py       # Face-detection sanity check per backend
  emotion_results.csv       # Output from emotion_detection.py
  face_detection_sanity.csv # Output from face_detection.py
```

---

## Requirements

- Python 3.10+
- A working GPU environment (optional but recommended)
- Core packages:
  - `deepface`
  - `tensorflow[and-cuda]` (for GPU)
  - `opencv-python`
  - `pandas`
  - `tqdm`

Optional backends require extra dependencies (examples):
- `dlib` backend: `pip install dlib` (requires `cmake` + build tools)
- `mediapipe` backend: `pip install mediapipe`
- `yolo*` backends: may require extra packages or weights

> Tip: If a backend fails to detect any faces, it will still be listed but usually won’t produce a `dominant_emotion`.

---

## Data setup

Move or rename your previous `test/` folder to `data/`, then place videos under `data/`. Any of these extensions are accepted:

```
.mp4 .mov .avi .mkv .m4v
```

Subfolders are scanned recursively.

---

## 1) Emotion detection

This script computes the **dominant emotion for each video**, for every available backend/model combination. It samples frames at ~1 FPS and aggregates emotion votes.

Run (default `data/`):

```bash
python scripts/emotion_detection.py
```

Or specify a custom folder:

```bash
python scripts/emotion_detection.py --data-dir /path/to/videos
```

Output: `emotion_results.csv`

Columns:
- `video_name`
- `backend`
- `supported_model`
- `dominant_emotion`

---

## 2) Face-detection sanity check

This script **tests face detection only** (no emotion) for each backend using a few sample frames per video. It helps you identify which backends can detect faces in your data.

Run (default `data/`):

```bash
python scripts/face_detection.py
```

Or specify a custom folder:

```bash
python scripts/face_detection.py --data-dir /path/to/videos
```

Output: `face_detection_sanity.csv`

Columns:
- `video_name`
- `backend`
- `frame_index`
- `faces_detected`
- `success` (True/False)

---

## GPU verification (optional but recommended)

Check whether TensorFlow sees your GPU:

```bash
TF_CPP_MIN_LOG_LEVEL=3 python - <<'PY'
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
PY
```

Expected output example:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Common issues

### 1) Backends return no emotion

Emotion is only computed **after a face is detected**. If a backend can’t detect a face, the dominant emotion will be `None`.

Use the sanity check (`face_detection.py`) to identify which backends are working on your data.

### 2) `dlib` install fails

Install system prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential
pip install dlib
```

### 3) GPU not detected by TensorFlow

Your NVIDIA driver may be installed, but TensorFlow might not have CUDA/CUDNN runtime libraries. Try:

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir "tensorflow[and-cuda]==2.20.*"
```

---

## Tweaks

Open the scripts to adjust:
- `SAMPLES_PER_VIDEO` in `face_detection.py`
- `backends` or `supported_models` lists
- Frame sampling rate in `emotion_detection.py`

---

If you want a resume-safe CSV, per-frame emotion logs, or a smaller backend/model list, I can add those.
