import os

# Silence TensorFlow/DeepFace logs as much as possible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import contextlib
import cv2
import pandas as pd
import tensorflow as tf
from deepface import DeepFace
from tqdm import tqdm

DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_CSV = "emotion_results.csv"
ALLOWED_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    #"fastmtcnn",
    #"retinaface",
    #"mediapipe",
    #"yolov8",
    #"yolov11s",
    #"yolov11n",
    #"yolov11m",
    "yunet",
    "centerface",
]

supported_models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU: not detected — running on CPU")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        name = details.get("device_name", "GPU")
    except Exception:
        name = "GPU"

    print(f"GPU: {name}")


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


def model_available(model_name, cache):
    if model_name in cache:
        return cache[model_name]
    try:
        with suppress_output():
            _ = DeepFace.build_model(model_name)
        cache[model_name] = True
    except Exception:
        cache[model_name] = False
    return cache[model_name]


def compute_dominant_emotion(video_path, backend, model_name, align=True):
    try:
        with suppress_output():
            emotion_model = DeepFace.build_model(model_name)
            DeepFace.custom_models = {model_name: emotion_model}
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps))

    frame_count = 0
    emotion_counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            try:
                with suppress_output():
                    analysis = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend=backend,
                        align=align,
                    )

                if isinstance(analysis, list):
                    analysis = analysis[0]
                curr_emotion = analysis.get("dominant_emotion")
                if curr_emotion:
                    emotion_counts[curr_emotion] = emotion_counts.get(curr_emotion, 0) + 1
            except Exception:
                pass

        frame_count += 1

    cap.release()

    if emotion_counts:
        return max(emotion_counts, key=emotion_counts.get)
    return None


def single_frame_test(video_path, backend, frame_index=0, align=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    try:
        analysis = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=backend,
            align=align,
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        return analysis
    except Exception as e:
        raise RuntimeError(f"Analyze failed: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Emotion detection over videos")
    parser.add_argument("--single-frame-test", action="store_true")
    parser.add_argument("--video", help="Path to a video for single-frame test")
    parser.add_argument("--backend", default="retinaface")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--align", action="store_true")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root folder containing videos (default: ./data)",
    )
    args = parser.parse_args()

    setup_gpu()

    if args.single_frame_test:
        if args.video:
            video_path = args.video
        else:
            videos = list(iter_videos(args.data_dir))
            if not videos:
                raise FileNotFoundError(f"No videos found under: {args.data_dir}")
            video_path = videos[0]

        result = single_frame_test(
            video_path,
            backend=args.backend,
            frame_index=args.frame_index,
            align=args.align,
        )
        print("Single-frame result:")
        print(result)
        return

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data folder not found: {args.data_dir}")

    videos = list(iter_videos(args.data_dir))
    if not videos:
        print(f"No videos found under: {args.data_dir}")
        return

    available_backends = [b for b in backends if backend_available(b)]
    model_cache = {}
    available_models = [m for m in supported_models if model_available(m, model_cache)]

    total = len(videos) * len(available_backends) * len(available_models)
    rows = []

    with tqdm(total=total, desc="Processing", unit="combo") as pbar:
        for video_path in videos:
            video_name = os.path.basename(video_path)
            for backend in available_backends:
                for model_name in available_models:
                    dominant_emotion = compute_dominant_emotion(
                        video_path, backend, model_name, align=True
                    )

                    rows.append(
                        {
                            "video_name": video_name,
                            "backend": backend,
                            "supported_model": model_name,
                            "dominant_emotion": dominant_emotion,
                        }
                    )
                    pbar.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
