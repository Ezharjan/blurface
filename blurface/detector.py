"""Face-detection backends for Blurface.

Two pure-PyTorch backends are available, both exposing the same
``detect(frames_rgb)`` API:

``YoloFaceDetector`` — default. Wraps :class:`ultralytics.YOLO` with the
    ``yolov8n-face`` weights from the `akanametov/yolo-face`_ release. YOLOv8
    is a single-stage anchor-free detector: one CNN forward pass produces
    every box, which makes it consistently faster than cascade detectors on
    GPU and noticeably more robust on moving / partially-occluded faces.
    Weights are downloaded once to ``~/.cache/blurface/`` and reused on
    subsequent runs.

``MtcnnDetector`` — optional fallback. Wraps the MTCNN implementation from
    ``facenet-pytorch``. MTCNN is a 3-stage cascade (P-Net / R-Net / O-Net)
    that is lighter on memory but slower per frame and less robust on
    motion. Useful when ``ultralytics`` cannot be installed or you want a
    second opinion on a hard clip.

Both backends return ``(boxes_per_frame, probs_per_frame)`` where each
element of ``boxes_per_frame`` is an ``(N, 4)`` ``np.float32`` array of
``[x1, y1, x2, y2]`` coordinates in original pixel space, and each element
of ``probs_per_frame`` is an ``(N,)`` ``np.float32`` array of detection
confidences.

.. _akanametov/yolo-face: https://github.com/akanametov/yolo-face
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import List, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "blurface")
DEFAULT_YOLO_FACE_NAME = "yolov8n-face.pt"
# We pin to a specific tag for reproducibility. The akanametov/yolo-face
# release ships both ONNX and .pt weights under GitHub releases.
DEFAULT_YOLO_FACE_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/"
    + DEFAULT_YOLO_FACE_NAME
)


def _ensure_yolo_face_weights(path: str = "", url: str = "") -> str:
    """Return the local path to the YOLO-face weights, downloading if needed."""
    if path and os.path.exists(path):
        return path
    os.makedirs(CACHE_DIR, exist_ok=True)
    target = path or os.path.join(CACHE_DIR, DEFAULT_YOLO_FACE_NAME)
    if os.path.exists(target):
        return target
    src = url or DEFAULT_YOLO_FACE_URL
    logger.info("Downloading YOLO-face weights from %s", src)
    try:
        urllib.request.urlretrieve(src, target)  # noqa: S310 (trusted URL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download YOLO-face weights from {src}: {exc}\n"
            f"Workarounds:\n"
            f"  1. Download the .pt file manually and place it at {target}\n"
            f"  2. Pass --model-path /path/to/weights.pt on the CLI\n"
            f"  3. Pass --model-url <mirror> to use a different download URL"
        ) from exc
    return target


class YoloFaceDetector:
    """YOLOv8-face wrapper around :class:`ultralytics.YOLO`."""

    name = "yolov8-face (ultralytics)"

    def __init__(
        self,
        device: torch.device,
        confidence: float = 0.5,
        iou: float = 0.45,
        half: bool = False,
        model_path: str = "",
        model_url: str = "",
        imgsz: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for the YOLO backend. "
                "Install it with `pip install ultralytics` (or "
                "`pip install blurface`, which already depends on it)."
            ) from exc

        weights = _ensure_yolo_face_weights(model_path, model_url)
        logger.info("Loading YOLO-face weights from %s", weights)
        self.model = YOLO(weights)
        self.device = device
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.half = bool(half) and device.type == "cuda"
        self.imgsz = int(imgsz)
        # Move weights to device. Some ultralytics versions defer this to
        # the first predict() call — we ignore the error in that case.
        try:
            self.model.to(device)
        except Exception:  # noqa: BLE001
            pass

    @torch.inference_mode()
    def detect(
        self, frames_rgb: Sequence[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not frames_rgb:
            return [], []
        # ultralytics accepts a list of HxWx3 numpy arrays in RGB.
        results = self.model.predict(
            list(frames_rgb),
            device=self.device,
            half=self.half,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        out_boxes: List[np.ndarray] = []
        out_probs: List[np.ndarray] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out_boxes.append(np.empty((0, 4), dtype=np.float32))
                out_probs.append(np.empty((0,), dtype=np.float32))
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
            out_boxes.append(xyxy)
            out_probs.append(conf)
        return out_boxes, out_probs


class MtcnnDetector:
    """``facenet-pytorch`` MTCNN wrapper — fallback backend."""

    name = "facenet-pytorch MTCNN"

    def __init__(
        self,
        device: torch.device,
        confidence: float = 0.90,
        half: bool = False,
        min_face_size: int = 20,
    ) -> None:
        try:
            from facenet_pytorch import MTCNN  # type: ignore  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is required for the MTCNN backend. "
                "Install with `pip install facenet-pytorch` or "
                "`pip install blurface[mtcnn]`."
            ) from exc
        self.device = device
        self.confidence = float(confidence)
        self.detector = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=int(min_face_size),
            post_process=False,
            select_largest=False,
        )
        if half and device.type == "cuda":
            self.detector.pnet = self.detector.pnet.half()
            self.detector.rnet = self.detector.rnet.half()
            self.detector.onet = self.detector.onet.half()

    @torch.inference_mode()
    def detect(
        self, frames_rgb: Sequence[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not frames_rgb:
            return [], []
        boxes, probs = self.detector.detect(list(frames_rgb))
        out_boxes: List[np.ndarray] = []
        out_probs: List[np.ndarray] = []
        for b, p in zip(boxes, probs):
            if b is None or len(b) == 0:
                out_boxes.append(np.empty((0, 4), dtype=np.float32))
                out_probs.append(np.empty((0,), dtype=np.float32))
                continue
            b = np.asarray(b, dtype=np.float32)
            p = np.asarray(p, dtype=np.float32)
            mask = p >= self.confidence
            out_boxes.append(b[mask])
            out_probs.append(p[mask])
        return out_boxes, out_probs


def build_detector(
    backend: str,
    device: torch.device,
    confidence: float = 0.5,
    half: bool = False,
    min_face_size: int = 20,
    model_path: str = "",
    model_url: str = "",
    imgsz: int = 640,
):
    """Construct a detector by name.

    ``backend`` is ``"yolo"``, ``"mtcnn"`` or ``"auto"`` (try YOLO, fall back
    to MTCNN if it fails to import). Raises :class:`ValueError` for any
    other string.
    """
    backend = (backend or "auto").lower()
    last_err: Exception | None = None

    if backend in ("auto", "yolo"):
        try:
            return YoloFaceDetector(
                device=device,
                confidence=confidence,
                half=half,
                model_path=model_path,
                model_url=model_url,
                imgsz=imgsz,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if backend == "yolo":
                raise
            logger.warning(
                "YOLO backend unavailable (%s); falling back to MTCNN.", exc
            )

    if backend in ("auto", "mtcnn"):
        return MtcnnDetector(
            device=device,
            confidence=max(confidence, 0.5),
            half=half,
            min_face_size=min_face_size,
        )

    raise ValueError(
        f"Unknown backend: {backend!r}. Expected 'yolo', 'mtcnn', or 'auto'. "
        f"Last error: {last_err}"
    )
