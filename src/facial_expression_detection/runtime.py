"""Runtime helpers around optional heavy dependencies."""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
from collections.abc import Iterator
from typing import Any

# Keep TensorFlow/DeepFace noise low unless the caller enables verbose logging.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - import presence depends on host environment.
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None

try:  # pragma: no cover - import presence depends on host environment.
    from deepface import DeepFace as _DeepFace
except Exception:  # pragma: no cover
    _DeepFace = None

try:  # pragma: no cover - import presence depends on host environment.
    import tensorflow as _tf
except Exception:  # pragma: no cover
    _tf = None


def require_cv2() -> Any:
    """Return cv2 module or raise a clear runtime error."""
    if _cv2 is None:
        raise RuntimeError(
            "opencv-python is required. "
            "Install project dependencies with `pip install -e .`."
        )
    return _cv2


def require_deepface() -> Any:
    """Return DeepFace class or raise a clear runtime error."""
    if _DeepFace is None:
        raise RuntimeError(
            "deepface is required. "
            "Install project dependencies with `pip install -e .`."
        )
    return _DeepFace


@contextlib.contextmanager
def suppress_third_party_output() -> Iterator[None]:
    """Suppress stdout/stderr from noisy third-party calls."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def is_backend_available(backend: str) -> bool:
    """Return whether an optional detector backend is importable."""
    if backend != "dlib":
        return True

    try:
        importlib.import_module("dlib")
    except Exception:
        return False
    return True


def setup_gpu_logging() -> None:
    """Log whether TensorFlow GPU devices are visible."""
    if _tf is None:
        LOGGER.debug("TensorFlow is not installed; skipping GPU discovery.")
        return

    gpus = _tf.config.list_physical_devices("GPU")
    if not gpus:
        LOGGER.info("GPU: not detected; running on CPU")
        return

    try:
        for gpu in gpus:
            _tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        LOGGER.debug("Unable to enable TensorFlow memory growth: %s", exc)

    try:
        details = _tf.config.experimental.get_device_details(gpus[0])
        name = details.get("device_name", "GPU")
    except Exception:
        name = "GPU"

    LOGGER.info("GPU: %s", name)
