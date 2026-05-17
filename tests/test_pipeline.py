"""Sanity tests for the blurface pipeline.

These tests are intentionally lightweight — they verify that the pipeline
can be constructed on the current machine and can process a tiny synthetic
clip end-to-end. They do not assert visual correctness of detection.

Run with::

    pytest -q
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from blurface.utils import FaceMosaicProcessor, select_device


def _make_synthetic_clip(path: str, n_frames: int = 30, size: int = 240) -> None:
    """Write a tiny solid-color MP4 so we don't depend on examples/video.mp4."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (size, size))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_select_device_returns_a_device():
    dev = select_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")


def test_processor_construct_cpu():
    proc = FaceMosaicProcessor(device="cpu", batch_size=2)
    assert proc.device.type == "cpu"


def test_processor_runs_on_synthetic_clip(tmp_path):
    src = tmp_path / "input.mp4"
    dst = tmp_path / "output.mp4"
    _make_synthetic_clip(str(src), n_frames=15, size=160)

    proc = FaceMosaicProcessor(device="cpu", batch_size=4, confidence=0.99)
    report = proc.process_video(str(src), str(dst), collect_metrics=True)

    assert os.path.exists(dst)
    assert report.frame_count >= 1
    # We don't expect to detect faces in random noise, just verify it ran.
    assert len(report.per_frame) == report.frame_count


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH; audio mux path not exercised.",
)
def test_audio_mux_path(tmp_path):
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"
    _make_synthetic_clip(str(src), n_frames=10, size=160)

    proc = FaceMosaicProcessor(device="cpu", batch_size=4)
    proc.process_video(str(src), str(dst))
    assert os.path.exists(dst)
