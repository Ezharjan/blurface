"""Core face-blurring pipeline (PyTorch, GPU-first).

This module hosts the :class:`FaceMosaicProcessor` — the engine behind the
``blurface`` and ``blurface-eval`` CLIs. Everything that is hot in the loop
is kept on the same :class:`torch.device` so that the *detect → mosaic*
hand-off does not pay for an unnecessary host round-trip:

1.  **Read.** ``cv2.VideoCapture`` decodes the source MP4 frame by frame and
    accumulates a batch of BGR ``np.ndarray`` images on the CPU.
2.  **Detect.** The batch is converted to RGB and handed to the active
    :class:`~blurface.detector` backend (YOLOv8-face by default, MTCNN as a
    fallback). Detection runs on the selected device and returns one
    ``[x1, y1, x2, y2]`` array of face boxes per frame.
3.  **Mosaic.** Each frame is uploaded once to the device as a CHW tensor.
    For every box we downsample the cropped face to a tiny
    ``mosaic_size × mosaic_size`` patch with
    ``torch.nn.functional.interpolate(mode="bilinear")`` and upsample it back
    with ``mode="nearest"`` — that's the classic pixelation effect, computed
    in a single kernel pair. For elliptical mosaics we additionally build a
    GPU-resident boolean mask and blend.
4.  **Encode.** The blurred frame is moved back to the CPU, converted to a
    contiguous ``uint8`` HWC array, and written to a temporary video with
    ``cv2.VideoWriter`` (FOURCC ``mp4v``).
5.  **Mux.** The final step shells out to ``ffmpeg`` (or, if unavailable,
    falls back to the ``ffmpeg-python`` bindings) to re-encode the video as
    H.264 + AAC and re-mux the original audio track back in.

Throughout the run, optional per-frame metrics (detect/mosaic latency, GPU
memory, face counts, mean confidence) are collected into a
:class:`RunReport` dataclass that the evaluation module turns into plots.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from blurface.detector import build_detector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection — actionable errors, graceful fallback
# ---------------------------------------------------------------------------

CUDA_REINSTALL_HINT = (
    "Your installed torch is the CPU build (torch.version.cuda is None or "
    "torch.cuda.is_available() returned False).\n"
    "Reinstall PyTorch with CUDA wheels:\n\n"
    "  blurface-install-gpu --fix --cuda 12.1\n\n"
    "Or manually:\n"
    "  pip uninstall -y torch torchvision\n"
    "  pip install torch torchvision --index-url "
    "https://download.pytorch.org/whl/cu121\n\n"
    "Then re-run with --device cuda."
)


def select_device(
    preferred: str = "auto", allow_cpu_fallback: bool = True
) -> torch.device:
    """Pick the best available torch device.

    Parameters
    ----------
    preferred:
        One of ``"auto"`` (cuda > mps > cpu), ``"cuda"``, ``"mps"`` or
        ``"cpu"``. Anything else is treated as ``"auto"``.
    allow_cpu_fallback:
        If ``True`` (default), an unavailable ``cuda``/``mps`` request falls
        back to CPU with a logged warning. If ``False``, it raises
        :class:`RuntimeError` instead.
    """
    preferred = (preferred or "auto").lower()

    if preferred == "cpu":
        return torch.device("cpu")

    cuda_works = False
    if torch.cuda.is_available():
        try:
            # Round-trip a tiny tensor to catch "compiled-for-wrong-sm" errors.
            _ = torch.zeros(1, device="cuda") * 2
            cuda_works = True
        except RuntimeError as exc:
            logger.debug("CUDA self-test failed: %s", exc)

    if preferred == "cuda":
        if cuda_works:
            return torch.device("cuda")
        msg = (
            "CUDA requested but no CUDA device is available (or compatible).\n"
            + CUDA_REINSTALL_HINT
        )
        if allow_cpu_fallback:
            logger.error(msg)
            logger.error("Falling back to CPU. Pass --device cpu to silence this.")
            return torch.device("cpu")
        raise RuntimeError(msg)

    if preferred == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if allow_cpu_fallback:
            logger.warning("MPS requested but not available; falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError("MPS requested but not available on this machine.")

    # auto
    if cuda_works:
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Return a human-readable label for ``device``."""
    if device.type == "cuda":
        idx = device.index or 0
        try:
            name = torch.cuda.get_device_name(idx)
            mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            return f"cuda:{idx} ({name}, {mem:.1f} GB)"
        except Exception:  # noqa: BLE001 — best-effort label only
            return f"cuda:{idx}"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return f"cpu ({platform.processor() or platform.machine()})"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class FrameMetric:
    """Per-frame timing + detection record."""

    frame_idx: int
    num_faces: int
    mean_confidence: float
    detect_ms: float
    mosaic_ms: float
    total_ms: float
    gpu_mem_mb: float


@dataclass
class RunReport:
    """Aggregate report for a single ``process_video`` invocation."""

    device: str
    backend: str
    video_path: str
    output_path: str
    width: int
    height: int
    fps_source: float
    frame_count: int
    batch_size: int
    half_precision: bool
    mosaic_size: int
    blur_shape: str
    confidence_threshold: float

    frames_processed: int = 0
    total_time_s: float = 0.0
    detect_time_s: float = 0.0
    mosaic_time_s: float = 0.0
    io_time_s: float = 0.0
    mux_time_s: float = 0.0
    avg_faces_per_frame: float = 0.0
    total_faces_detected: int = 0
    frames_with_faces: int = 0
    peak_gpu_mem_mb: float = 0.0
    realtime_fps: float = 0.0
    per_frame: List[FrameMetric] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Face mosaic processor
# ---------------------------------------------------------------------------


class FaceMosaicProcessor:
    """Detect-and-blur faces in a video using a fully GPU-resident pipeline.

    Parameters
    ----------
    mosaic_size:
        Edge length, in pixels, of the down-sampled mosaic block. Higher =
        coarser blur. Default ``10``.
    blur_shape:
        ``"ellipse"`` (default, blends an elliptical mask over each box) or
        ``"rectangle"`` (replaces the whole bounding box with the mosaic).
    device:
        ``"auto"`` (default) | ``"cuda"`` | ``"mps"`` | ``"cpu"``. See
        :func:`select_device`.
    batch_size:
        Number of frames handed to the detector at once. Tune to fill your
        GPU.
    half:
        Run the detector and mosaic in FP16 on CUDA (no-op on other
        devices). Reduces memory and is typically faster on modern GPUs.
    confidence:
        Minimum per-detection confidence (in ``[0, 1]``).
    min_face_size:
        Minimum face edge length, in pixels, accepted by the MTCNN backend.
        Ignored by the YOLO backend (which uses ``imgsz`` instead).
    backend:
        ``"auto"`` (YOLO with MTCNN fallback) | ``"yolo"`` | ``"mtcnn"``.
    model_path, model_url:
        Override the YOLO-face weights location / download URL.
    imgsz:
        YOLO inference image size (default ``640``). Larger = more accurate
        on tiny faces, slower.
    allow_cpu_fallback:
        If ``False``, raise on unavailable accelerator instead of falling
        back to CPU.
    """

    def __init__(
        self,
        mosaic_size: int = 10,
        blur_shape: str = "ellipse",
        device: str = "auto",
        batch_size: int = 8,
        half: bool = False,
        confidence: float = 0.5,
        min_face_size: int = 20,
        backend: str = "auto",
        model_path: str = "",
        model_url: str = "",
        imgsz: int = 640,
        allow_cpu_fallback: bool = True,
    ) -> None:
        if blur_shape not in ("rectangle", "ellipse"):
            raise ValueError("blur_shape must be 'rectangle' or 'ellipse'")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in the range [0, 1]")

        self.mosaic_size = max(1, int(mosaic_size))
        self.blur_shape = blur_shape
        self.batch_size = max(1, int(batch_size))
        self.confidence = float(confidence)
        self.min_face_size = max(1, int(min_face_size))
        self.device = select_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self.use_half = bool(half) and self.device.type == "cuda"

        logger.info("Using device: %s", describe_device(self.device))
        if self.use_half:
            logger.info("Half precision (FP16) enabled for GPU inference.")

        self.detector = build_detector(
            backend=backend,
            device=self.device,
            confidence=self.confidence,
            half=self.use_half,
            min_face_size=self.min_face_size,
            model_path=model_path,
            model_url=model_url,
            imgsz=imgsz,
        )
        logger.info("Detector backend: %s", self.detector.name)

    # ------------------------------------------------------------------ mosaic

    @torch.inference_mode()
    def apply_mosaic_gpu(
        self, frame_bgr: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        """Apply a GPU mosaic to every box on ``frame_bgr`` (HxWx3 ``uint8``)."""
        if boxes.shape[0] == 0:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        dtype = torch.float16 if self.use_half else torch.float32
        frame_t = (
            torch.from_numpy(frame_bgr)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .to(dtype)
        )

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(max(0, np.floor(x1)))
            y1 = int(max(0, np.floor(y1)))
            x2 = int(min(w, np.ceil(x2)))
            y2 = int(min(h, np.ceil(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            face = frame_t[:, y1:y2, x1:x2].unsqueeze(0)
            fh, fw = face.shape[-2], face.shape[-1]
            small = F.interpolate(
                face,
                size=(self.mosaic_size, self.mosaic_size),
                mode="bilinear",
                align_corners=False,
            )
            mosaic = F.interpolate(small, size=(fh, fw), mode="nearest")

            if self.blur_shape == "rectangle":
                frame_t[:, y1:y2, x1:x2] = mosaic.squeeze(0)
            else:
                ys = torch.arange(fh, device=self.device, dtype=dtype).view(-1, 1)
                xs = torch.arange(fw, device=self.device, dtype=dtype).view(1, -1)
                cy = (fh - 1) / 2.0
                cx = (fw - 1) / 2.0
                ry = max(fh / 2.0, 1.0)
                rx = max(fw / 2.0, 1.0)
                mask = ((xs - cx) ** 2) / (rx**2) + ((ys - cy) ** 2) / (ry**2)
                mask = (mask <= 1.0).to(dtype).unsqueeze(0)
                blended = face.squeeze(0) * (1 - mask) + mosaic.squeeze(0) * mask
                frame_t[:, y1:y2, x1:x2] = blended

        frame_t = frame_t.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
        return frame_t.cpu().numpy()

    # ------------------------------------------------------------------ video

    def process_video(
        self,
        video_path: str,
        output_path: str,
        report_path: Optional[str] = None,
        collect_metrics: bool = True,
    ) -> RunReport:
        """Run the full pipeline on ``video_path`` and write to ``output_path``.

        Parameters
        ----------
        video_path:
            Input MP4 path.
        output_path:
            Output MP4 path. Parent directories are created if needed.
        report_path:
            If given, the :class:`RunReport` is also serialised to this JSON
            file.
        collect_metrics:
            When ``True`` (default), populates ``RunReport.per_frame`` with
            one entry per frame — required by the evaluation plots. When
            ``False``, only aggregate counters are tracked, which is a small
            speed-up on very long videos.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Input video: %dx%d @ %.2f fps, %d frames",
            width,
            height,
            fps,
            total_frames,
        )

        report = RunReport(
            device=describe_device(self.device),
            backend=self.detector.name,
            video_path=os.path.abspath(video_path),
            output_path=os.path.abspath(output_path),
            width=width,
            height=height,
            fps_source=fps,
            frame_count=total_frames,
            batch_size=self.batch_size,
            half_precision=self.use_half,
            mosaic_size=self.mosaic_size,
            blur_shape=self.blur_shape,
            confidence_threshold=self.confidence,
        )

        temp_dir = tempfile.mkdtemp(prefix="blurface_")
        temp_video = os.path.join(temp_dir, "blurred_noaudio.mp4")
        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                "Failed to open temporary VideoWriter. "
                "Check that OpenCV was built with FFmpeg/mp4v support."
            )

        # Aggregate counters — kept correct even when collect_metrics is False.
        total_faces = 0
        frames_with_faces = 0
        detect_time_s = 0.0
        mosaic_time_s = 0.0
        peak_gpu_mem_mb = 0.0
        frames_processed = 0
        t_start = time.perf_counter()

        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            batch_bgr: List[np.ndarray] = []
            batch_indices: List[int] = []
            frame_idx = 0

            with tqdm(total=total_frames, desc="Blurring", unit="frame") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_bgr.append(frame)
                    batch_indices.append(frame_idx)
                    frame_idx += 1

                    if len(batch_bgr) >= self.batch_size:
                        n = len(batch_bgr)
                        stats = self._process_batch(
                            batch_bgr,
                            batch_indices,
                            writer,
                            report,
                            collect_metrics,
                        )
                        total_faces += stats["faces"]
                        frames_with_faces += stats["frames_with_faces"]
                        detect_time_s += stats["detect_s"]
                        mosaic_time_s += stats["mosaic_s"]
                        frames_processed += n
                        pbar.update(n)
                        batch_bgr.clear()
                        batch_indices.clear()

                if batch_bgr:
                    n = len(batch_bgr)
                    stats = self._process_batch(
                        batch_bgr,
                        batch_indices,
                        writer,
                        report,
                        collect_metrics,
                    )
                    total_faces += stats["faces"]
                    frames_with_faces += stats["frames_with_faces"]
                    detect_time_s += stats["detect_s"]
                    mosaic_time_s += stats["mosaic_s"]
                    frames_processed += n
                    pbar.update(n)
        finally:
            cap.release()
            writer.release()

        if self.device.type == "cuda":
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated(self.device) / (
                1024**2
            )

        t_end = time.perf_counter()
        report.frames_processed = frames_processed
        report.total_time_s = t_end - t_start
        report.detect_time_s = detect_time_s
        report.mosaic_time_s = mosaic_time_s
        report.total_faces_detected = total_faces
        report.avg_faces_per_frame = total_faces / max(1, frames_processed)
        report.frames_with_faces = frames_with_faces
        report.peak_gpu_mem_mb = peak_gpu_mem_mb
        report.realtime_fps = frames_processed / max(1e-9, report.total_time_s)

        mux_t0 = time.perf_counter()
        try:
            self._mux_audio(video_path, temp_video, output_path)
        except Exception as exc:  # noqa: BLE001 — log and copy as a fallback
            logger.warning("Audio mux failed, copying video-only output: %s", exc)
            shutil.copy(temp_video, output_path)
        report.mux_time_s = time.perf_counter() - mux_t0

        shutil.rmtree(temp_dir, ignore_errors=True)

        if report_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True
            )
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2)
            logger.info("Wrote report to %s", report_path)

        logger.info(
            "Done in %.2fs (%.2f fps). Output: %s",
            report.total_time_s,
            report.realtime_fps,
            output_path,
        )
        return report

    # ------------------------------------------------------------------ helpers

    def _process_batch(
        self,
        frames_bgr: List[np.ndarray],
        frame_indices: List[int],
        writer: "cv2.VideoWriter",
        report: RunReport,
        collect_metrics: bool,
    ) -> dict:
        """Detect + mosaic + write a single batch. Returns aggregate counters."""
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

        t0 = time.perf_counter()
        boxes_list, probs_list = self.detector.detect(frames_rgb)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        batch_detect_s = t1 - t0
        per_frame_detect_ms = (batch_detect_s / max(1, len(frames_bgr))) * 1000.0

        faces = 0
        frames_with_faces = 0
        mosaic_total_s = 0.0

        for idx, frame_bgr, boxes, probs in zip(
            frame_indices, frames_bgr, boxes_list, probs_list
        ):
            m0 = time.perf_counter()
            out = self.apply_mosaic_gpu(frame_bgr, boxes)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            m1 = time.perf_counter()
            writer.write(out)
            mosaic_ms = (m1 - m0) * 1000.0
            mosaic_total_s += (m1 - m0)

            n_faces = int(boxes.shape[0])
            faces += n_faces
            if n_faces > 0:
                frames_with_faces += 1

            if collect_metrics:
                gpu_mem = 0.0
                if self.device.type == "cuda":
                    gpu_mem = torch.cuda.memory_allocated(self.device) / (1024**2)
                report.per_frame.append(
                    FrameMetric(
                        frame_idx=idx,
                        num_faces=n_faces,
                        mean_confidence=float(probs.mean()) if probs.size else 0.0,
                        detect_ms=per_frame_detect_ms,
                        mosaic_ms=mosaic_ms,
                        total_ms=mosaic_ms + per_frame_detect_ms,
                        gpu_mem_mb=gpu_mem,
                    )
                )

        return {
            "faces": faces,
            "frames_with_faces": frames_with_faces,
            "detect_s": batch_detect_s,
            "mosaic_s": mosaic_total_s,
        }

    # ------------------------------------------------------------------ audio

    @staticmethod
    def _have_ffmpeg() -> bool:
        return shutil.which("ffmpeg") is not None

    def _mux_audio(
        self, src_with_audio: str, video_only: str, output: str
    ) -> None:
        """Re-encode ``video_only`` as H.264/AAC and remux audio from src."""
        if self._have_ffmpeg():
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_only,
                "-i", src_with_audio,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output,
            ]
            logger.info("Muxing audio: ffmpeg -> %s", output)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (exit {proc.returncode}): "
                    f"{proc.stderr[-400:]}"
                )
            return

        try:
            import ffmpeg  # ffmpeg-python  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "ffmpeg CLI not found and ffmpeg-python is not installed; "
                "install ffmpeg (https://ffmpeg.org/download.html) to enable "
                "audio passthrough."
            ) from exc

        video_in = ffmpeg.input(video_only)
        audio_in = ffmpeg.input(src_with_audio)
        try:
            (
                ffmpeg.output(
                    video_in.video,
                    audio_in.audio,
                    output,
                    vcodec="libx264",
                    preset="medium",
                    crf=20,
                    acodec="aac",
                    audio_bitrate="192k",
                    shortest=None,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            (
                ffmpeg.input(video_only)
                .output(output, vcodec="libx264", preset="medium", crf=20)
                .overwrite_output()
                .run(quiet=True)
            )
