#!/usr/bin/env python3
"""Command-line entry point for the Blurface tool.

Run ``blurface --help`` for the full option reference and a few worked
examples. The same module is invokable as ``python -m blurface``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from blurface import __version__
from blurface.utils import FaceMosaicProcessor


_CLI_EPILOG = """\
Examples
--------
  # 1. Defaults: ellipse mosaic, auto device, YOLOv8-face detector.
  blurface input.mp4

  # 2. Force CUDA, FP16, larger batch, custom output path.
  blurface input.mp4 -d cuda -b 32 --half -o out/blurred.mp4

  # 3. Coarser rectangular mosaic (block size 20).
  blurface input.mp4 -m 20 -s rectangle

  # 4. Use the MTCNN fallback backend (needs the [mtcnn] extra).
  blurface input.mp4 --backend mtcnn

  # 5. Emit a full JSON metrics report and a directory of PNG plots.
  blurface input.mp4 --report out/report.json --plots-dir out/plots

  # 6. Provide your own YOLO-face weights (skips the download).
  blurface input.mp4 --model-path /path/to/yolov8n-face.pt

  # 7. Cap inference image size for tiny GPUs / lots of small faces.
  blurface input.mp4 --imgsz 1280 --batch-size 4

See also
--------
  blurface-eval         Detailed evaluation with CSV/JSON/PNG outputs.
  blurface-install-gpu  Diagnose and (re)install PyTorch with CUDA wheels.
"""


def parse_cli_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="blurface",
        description=(
            "Blur human faces in an MP4 video using a GPU-accelerated PyTorch "
            "pipeline (YOLOv8-face via ultralytics by default, MTCNN as a "
            "fallback). Audio is preserved via ffmpeg re-mux."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    parser.add_argument(
        "input",
        help="Path to the input MP4 video file (required).",
    )
    parser.add_argument(
        "--output", "-o",
        help=(
            "Output MP4 path. Defaults to '<input-stem><YYMMDDHHMM>.mp4' in "
            "the current directory."
        ),
    )
    parser.add_argument(
        "--mosaic-size", "-m", type=int, default=10,
        help="Mosaic block size in pixels; higher = coarser blur (default: 10).",
    )
    parser.add_argument(
        "--blur-shape", "-s",
        choices=["rectangle", "ellipse"], default="ellipse",
        help="Mosaic mask shape (default: ellipse).",
    )
    parser.add_argument(
        "--device", "-d",
        choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help=(
            "Compute device. 'auto' picks cuda > mps > cpu, falling back if "
            "needed (default: auto)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "yolo", "mtcnn"], default="auto",
        help=(
            "Detection backend. 'auto' tries YOLO first and falls back to "
            "MTCNN on import failure (default: auto)."
        ),
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=8,
        help="Frames per detection batch (default: 8). Tune to fill your GPU.",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Run detection + mosaic in FP16 on CUDA. Faster, less memory.",
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help=(
            "Minimum detection confidence in [0, 1] (default: 0.5). "
            "Raise for fewer false positives, lower for more aggressive blurring."
        ),
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help=(
            "YOLO inference image size (default: 640). Raise (e.g. 960, 1280) "
            "for tiny faces, lower for speed."
        ),
    )
    parser.add_argument(
        "--min-face-size", type=int, default=20,
        help="Minimum face edge length in px (MTCNN backend only; default: 20).",
    )
    parser.add_argument(
        "--model-path", default="",
        help=(
            "Path to a local YOLO-face .pt file. Overrides the download and "
            "any --model-url."
        ),
    )
    parser.add_argument(
        "--model-url", default="",
        help="Custom URL to download YOLO-face weights from.",
    )
    parser.add_argument(
        "--no-cpu-fallback", action="store_true",
        help="Hard-fail when --device cuda/mps is requested but unavailable.",
    )
    parser.add_argument(
        "--report",
        help=(
            "Optional path for a JSON metrics report (RunReport with all "
            "per-frame metrics included)."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        help=(
            "If set, evaluation plots (PNG) and per_frame_metrics.csv are "
            "written here. Implies metrics collection."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only log warnings and errors (suppresses progress info).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging (overrides --quiet).",
    )
    parser.add_argument(
        "--version", action="version", version=f"blurface {__version__}",
    )

    args = parser.parse_args(argv)

    if not args.input.lower().endswith(".mp4"):
        parser.error("Input file must be an MP4 video (got: %s)." % args.input)
    if not os.path.exists(args.input):
        parser.error(f"Input video not found: {args.input}")

    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        args.output = f"{base_name}{timestamp}.mp4"
    elif not args.output.lower().endswith(".mp4"):
        args.output += ".mp4"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    return args


def _configure_logging(args: argparse.Namespace) -> None:
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv=None) -> int:
    args = parse_cli_args(argv)
    _configure_logging(args)

    try:
        processor = FaceMosaicProcessor(
            mosaic_size=args.mosaic_size,
            blur_shape=args.blur_shape,
            device=args.device,
            batch_size=args.batch_size,
            half=args.half,
            confidence=args.confidence,
            min_face_size=args.min_face_size,
            backend=args.backend,
            model_path=args.model_path,
            model_url=args.model_url,
            imgsz=args.imgsz,
            allow_cpu_fallback=not args.no_cpu_fallback,
        )
        report = processor.process_video(
            args.input,
            args.output,
            report_path=args.report,
            collect_metrics=bool(args.report or args.plots_dir),
        )
        if args.plots_dir:
            from blurface.evaluate import render_plots  # local import: matplotlib

            render_plots(report, args.plots_dir)
            logging.info("Plots written to %s", args.plots_dir)
        logging.info("Processed video saved to %s", args.output)
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        return 130
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to process %s: %s", args.input, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
