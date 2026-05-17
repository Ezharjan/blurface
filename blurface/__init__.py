"""Blurface — GPU-accelerated face blurring for MP4 videos.

Quick start (Python API):

    from blurface import FaceMosaicProcessor

    proc = FaceMosaicProcessor(device="auto", backend="yolo",
                               batch_size=16, half=True)
    report = proc.process_video("input.mp4", "output.mp4")
    print(f"{report.realtime_fps:.1f} fps on {report.device}")

Quick start (CLI):

    blurface input.mp4 --device cuda --half --batch-size 16

See :mod:`blurface.utils` for the pipeline, :mod:`blurface.detector` for
the available detection backends and :mod:`blurface.evaluate` for the
metrics and plots.
"""

__version__ = "0.2.0"

from blurface.detector import (  # noqa: F401
    MtcnnDetector,
    YoloFaceDetector,
    build_detector,
)
from blurface.utils import (  # noqa: F401
    FaceMosaicProcessor,
    FrameMetric,
    RunReport,
    describe_device,
    select_device,
)

__all__ = [
    "__version__",
    "FaceMosaicProcessor",
    "FrameMetric",
    "RunReport",
    "YoloFaceDetector",
    "MtcnnDetector",
    "build_detector",
    "select_device",
    "describe_device",
]
