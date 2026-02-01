import os

# Keep logs minimal
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import contextlib
import cv2
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_CSV = "face_detection_sanity.csv"
ALLOWED_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

backends = [
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
]

SAMPLES_PER_VIDEO = 3


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def iter_videos(root_dir, allowed_exts=ALLOWED_EXTS):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.lower().endswith(allowed_exts):
                yield os.path.join(dirpath, fn)


def backend_available(backend):
    if backend == "dlib":
        try:
            import dlib  # noqa: F401
        except Exception:
            return False
    return True


def sample_frames(video_path, samples=SAMPLES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = 1

    # Evenly spaced frame indices
    if samples <= 1:
        indices = [0]
    else:
        step = max(1, total_frames // samples)
        indices = [min(i * step, total_frames - 1) for i in range(samples)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((idx, frame))

    cap.release()
    return frames


def detect_faces_in_frame(frame, backend):
    try:
        with suppress_output():
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=backend,
                enforce_detection=False,
            )
        return len(faces)
    except Exception:
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Face-detection sanity check")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root folder containing videos (default: ./data)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data folder not found: {args.data_dir}")

    videos = list(iter_videos(args.data_dir))
    if not videos:
        print(f"No videos found under: {args.data_dir}")
        return

    available_backends = [b for b in backends if backend_available(b)]
    total = len(videos) * len(available_backends) * SAMPLES_PER_VIDEO

    rows = []
    with tqdm(total=total, desc="Sanity check", unit="frame") as pbar:
        for video_path in videos:
            video_name = os.path.basename(video_path)
            frames = sample_frames(video_path, samples=SAMPLES_PER_VIDEO)
            if not frames:
                # Record no frames found
                for backend in available_backends:
                    rows.append(
                        {
                            "video_name": video_name,
                            "backend": backend,
                            "frame_index": None,
                            "faces_detected": 0,
                            "success": False,
                        }
                    )
                    pbar.update(SAMPLES_PER_VIDEO)
                continue

            for backend in available_backends:
                for frame_idx, frame in frames:
                    faces = detect_faces_in_frame(frame, backend)
                    rows.append(
                        {
                            "video_name": video_name,
                            "backend": backend,
                            "frame_index": frame_idx,
                            "faces_detected": faces,
                            "success": faces > 0,
                        }
                    )
                    pbar.update(1)

                if len(frames) < SAMPLES_PER_VIDEO:
                    pbar.update(SAMPLES_PER_VIDEO - len(frames))

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
