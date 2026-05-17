"""Evaluation metrics & visualizations for the Blurface pipeline.

This module turns the per-frame metrics produced by
:class:`blurface.utils.FaceMosaicProcessor` into:

* a tidy ``per_frame_metrics.csv`` (one row per processed frame),
* a ``summary.json`` with aggregate counters,
* a set of PNG plots (faces-per-frame, latency breakdown, rolling FPS,
  GPU memory, confidence histogram, scorecard),
* an optional CPU-vs-GPU benchmark with its own bar chart.

The same logic is reachable from the CLI via ``blurface-eval`` — see
:func:`main`.

Quick Python example:

    from blurface import FaceMosaicProcessor
    from blurface.evaluate import render_plots

    proc = FaceMosaicProcessor(device="auto")
    report = proc.process_video("in.mp4", "out.mp4")
    paths = render_plots(report, "report/plots")
    print(paths["summary_png"])
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe on headless boxes
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from blurface.utils import FaceMosaicProcessor, RunReport  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def report_to_dataframe(report: RunReport) -> pd.DataFrame:
    """Convert a :class:`RunReport` to a per-frame :class:`pandas.DataFrame`."""
    rows = [asdict(m) for m in report.per_frame]
    if not rows:
        return pd.DataFrame(
            columns=[
                "frame_idx",
                "num_faces",
                "mean_confidence",
                "detect_ms",
                "mosaic_ms",
                "total_ms",
                "gpu_mem_mb",
            ]
        )
    return pd.DataFrame(rows)


def render_plots(report: RunReport, out_dir: str) -> dict:
    """Render evaluation plots for ``report`` into ``out_dir``.

    Returns a dict ``{plot_name: path}`` for every file that was created.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = report_to_dataframe(report)
    paths: dict = {}

    # CSV dump — useful for downstream analysis.
    csv_path = os.path.join(out_dir, "per_frame_metrics.csv")
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    summary = {
        "device": report.device,
        "backend": report.backend,
        "video_path": report.video_path,
        "output_path": report.output_path,
        "resolution": f"{report.width}x{report.height}",
        "frame_count": report.frame_count,
        "frames_processed": report.frames_processed,
        "fps_source": report.fps_source,
        "fps_processing": report.realtime_fps,
        "speedup_vs_realtime": (
            report.realtime_fps / report.fps_source if report.fps_source else None
        ),
        "total_time_s": report.total_time_s,
        "detect_time_s": report.detect_time_s,
        "mosaic_time_s": report.mosaic_time_s,
        "mux_time_s": report.mux_time_s,
        "avg_faces_per_frame": report.avg_faces_per_frame,
        "total_faces_detected": report.total_faces_detected,
        "frames_with_faces": report.frames_with_faces,
        "peak_gpu_mem_mb": report.peak_gpu_mem_mb,
        "batch_size": report.batch_size,
        "half_precision": report.half_precision,
        "mosaic_size": report.mosaic_size,
        "blur_shape": report.blur_shape,
        "confidence_threshold": report.confidence_threshold,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    paths["summary"] = summary_path

    if df.empty:
        logger.warning("No per-frame metrics in report; skipping plots.")
        return paths

    # 1. Faces per frame ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["frame_idx"], df["num_faces"], lw=0.9, color="#1f77b4")
    ax.fill_between(df["frame_idx"], 0, df["num_faces"], alpha=0.25, color="#1f77b4")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Faces detected")
    ax.set_title(f"Detections per frame  (avg {df['num_faces'].mean():.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    paths["faces_per_frame"] = os.path.join(out_dir, "faces_per_frame.png")
    fig.savefig(paths["faces_per_frame"], dpi=130)
    plt.close(fig)

    # 2. Latency breakdown (detect vs mosaic) ------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df["frame_idx"], df["detect_ms"], label="Detect (ms)",
        color="#d62728", lw=0.9,
    )
    ax.plot(
        df["frame_idx"], df["mosaic_ms"], label="Mosaic (ms)",
        color="#2ca02c", lw=0.9,
    )
    ax.plot(
        df["frame_idx"], df["total_ms"], label="Total (ms)",
        color="#9467bd", lw=1.1, alpha=0.7,
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-frame latency breakdown")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    paths["latency"] = os.path.join(out_dir, "latency_per_frame.png")
    fig.savefig(paths["latency"], dpi=130)
    plt.close(fig)

    # 3. Rolling FPS (1-second window using source fps) --------------------
    window = max(1, int(report.fps_source) or 25)
    rolling_ms = df["total_ms"].rolling(window=window, min_periods=1).mean()
    fps_series = 1000.0 / rolling_ms.clip(lower=1e-6)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["frame_idx"], fps_series, color="#ff7f0e", lw=1.1)
    ax.axhline(
        report.fps_source, ls="--", color="black", alpha=0.5,
        label=f"Source fps = {report.fps_source:.1f}",
    )
    ax.axhline(
        report.realtime_fps, ls=":", color="#1f77b4", alpha=0.7,
        label=f"Avg processing fps = {report.realtime_fps:.1f}",
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Processing FPS (rolling)")
    ax.set_title("Throughput")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    paths["fps"] = os.path.join(out_dir, "fps_rolling.png")
    fig.savefig(paths["fps"], dpi=130)
    plt.close(fig)

    # 4. GPU memory (if applicable) ----------------------------------------
    if df["gpu_mem_mb"].max() > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["frame_idx"], df["gpu_mem_mb"], color="#17becf", lw=0.9)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("GPU memory allocated (MB)")
        ax.set_title(
            f"GPU memory   peak={report.peak_gpu_mem_mb:.1f} MB   "
            f"device={report.device}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        paths["gpu_mem"] = os.path.join(out_dir, "gpu_memory.png")
        fig.savefig(paths["gpu_mem"], dpi=130)
        plt.close(fig)

    # 5. Confidence histogram ----------------------------------------------
    confs = df.loc[df["num_faces"] > 0, "mean_confidence"]
    if len(confs):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(confs, bins=20, color="#8c564b", edgecolor="white")
        ax.set_xlabel("Mean detection confidence (per frame)")
        ax.set_ylabel("Count")
        ax.set_title("Confidence distribution on frames with faces")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        paths["confidence_hist"] = os.path.join(
            out_dir, "confidence_histogram.png"
        )
        fig.savefig(paths["confidence_hist"], dpi=130)
        plt.close(fig)

    # 6. Summary "scorecard" ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.axis("off")
    rt_factor = report.realtime_fps / max(1e-6, report.fps_source)
    lines = [
        f"Device           : {report.device}",
        f"Backend          : {report.backend}",
        f"Resolution       : {report.width} x {report.height}",
        f"Frames processed : {report.frames_processed} / {report.frame_count}",
        f"Source FPS       : {report.fps_source:.2f}",
        f"Processing FPS   : {report.realtime_fps:.2f}",
        f"Realtime factor  : {rt_factor:.2f}x",
        f"Total time       : {report.total_time_s:.2f} s",
        f"Detect total     : {report.detect_time_s:.2f} s",
        f"Mosaic total     : {report.mosaic_time_s:.2f} s",
        f"Audio mux        : {report.mux_time_s:.2f} s",
        f"Avg faces/frame  : {report.avg_faces_per_frame:.2f}",
        f"Total faces      : {report.total_faces_detected}",
        f"Frames w/ faces  : {report.frames_with_faces}",
        f"Peak GPU memory  : {report.peak_gpu_mem_mb:.1f} MB",
        f"Batch size       : {report.batch_size}",
        f"FP16             : {report.half_precision}",
        f"Mosaic size      : {report.mosaic_size}",
        f"Blur shape       : {report.blur_shape}",
        f"Conf threshold   : {report.confidence_threshold}",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines),
        va="top", ha="left", family="monospace", fontsize=10,
    )
    ax.set_title("Blurface evaluation summary")
    fig.tight_layout()
    paths["summary_png"] = os.path.join(out_dir, "summary.png")
    fig.savefig(paths["summary_png"], dpi=130)
    plt.close(fig)

    return paths


def benchmark_cpu_vs_gpu(
    video_path: str,
    output_dir: str,
    mosaic_size: int = 10,
    blur_shape: str = "ellipse",
    batch_size: int = 8,
    confidence: float = 0.5,
    backend: str = "auto",
    half: bool = False,
    max_frames: Optional[int] = None,
) -> dict:
    """Run the pipeline once on CPU and once on the best non-CPU device.

    Writes ``cpu_vs_gpu.png`` and ``cpu_vs_gpu.json`` to ``output_dir`` and
    returns a dict of per-device FPS / timing entries (plus ``"chart"``).
    """
    import torch  # noqa: WPS433 — keep optional at module-load time

    os.makedirs(output_dir, exist_ok=True)
    results: dict = {}

    devices_to_run = ["cpu"]
    if torch.cuda.is_available():
        devices_to_run.append("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices_to_run.append("mps")

    for device in devices_to_run:
        out_video = os.path.join(output_dir, f"benchmark_{device}.mp4")
        proc = FaceMosaicProcessor(
            mosaic_size=mosaic_size,
            blur_shape=blur_shape,
            device=device,
            batch_size=batch_size,
            confidence=confidence,
            backend=backend,
            half=half,
        )
        t0 = time.perf_counter()
        report = _process_with_optional_cap(proc, video_path, out_video, max_frames)
        wall = time.perf_counter() - t0
        results[device] = {
            "wall_s": wall,
            "fps": report.realtime_fps,
            "detect_s": report.detect_time_s,
            "mosaic_s": report.mosaic_time_s,
            "avg_faces": report.avg_faces_per_frame,
        }
        logger.info("[%s] %.2f fps  (wall %.2fs)", device, report.realtime_fps, wall)

    # Plot ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [d for d in results if isinstance(results[d], dict)]
    fps_vals = [results[d]["fps"] for d in labels]
    bars = ax.bar(labels, fps_vals, color=["#7f7f7f", "#2ca02c"][: len(labels)])
    for bar, v in zip(bars, fps_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
            ha="center", va="bottom",
        )
    ax.set_ylabel("Processing FPS")
    ax.set_title("Blurface CPU vs GPU throughput")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = os.path.join(output_dir, "cpu_vs_gpu.png")
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)
    results["chart"] = chart_path

    with open(
        os.path.join(output_dir, "cpu_vs_gpu.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(results, fh, indent=2)
    return results


def _process_with_optional_cap(
    proc: FaceMosaicProcessor,
    video_path: str,
    output_path: str,
    max_frames: Optional[int],
) -> RunReport:
    """Run ``proc.process_video`` on the first ``max_frames`` if requested."""
    if max_frames is None:
        return proc.process_video(video_path, output_path)

    # Build a short subclip via ffmpeg, then process it.
    import shutil  # noqa: WPS433
    import subprocess  # noqa: WPS433
    import tempfile  # noqa: WPS433

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not on PATH; ignoring --benchmark-frames cap.")
        return proc.process_video(video_path, output_path)

    tmp_dir = tempfile.mkdtemp(prefix="blurface_clip_")
    clip = os.path.join(tmp_dir, "clip.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vframes", str(int(max_frames)),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        clip,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    try:
        return proc.process_video(clip, output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EVAL_EPILOG = """\
Examples
--------
  # Vanilla evaluation run — produces report.json, plots and CSV
  blurface-eval input.mp4 -o out/blurred.mp4 -r out/report

  # CUDA + FP16 + bigger batch, with a CPU-vs-GPU benchmark
  blurface-eval input.mp4 -o out/blurred.mp4 -r out/report \\
      --device cuda --half --batch-size 16 \\
      --benchmark --benchmark-frames 240

  # Use the MTCNN backend, raise the confidence threshold
  blurface-eval input.mp4 -o out/blurred.mp4 -r out/report \\
      --backend mtcnn --confidence 0.9
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="blurface-eval",
        description=(
            "Run Blurface on a video and emit a metrics report "
            "(CSV + JSON) and PNG plots. Optionally benchmark CPU vs GPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EVAL_EPILOG,
    )
    parser.add_argument("input", help="Input MP4 video.")
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output (blurred) MP4 path.",
    )
    parser.add_argument(
        "--report-dir", "-r", required=True,
        help="Directory to write CSV/JSON/PNG metrics into.",
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "yolo", "mtcnn"],
        help="Detection backend (default: auto -> yolo).",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=8,
        help="Frames per detection batch (default: 8).",
    )
    parser.add_argument(
        "--mosaic-size", "-m", type=int, default=10,
        help="Mosaic block size in pixels; higher = coarser blur (default: 10).",
    )
    parser.add_argument(
        "--blur-shape", "-s", choices=["rectangle", "ellipse"], default="ellipse",
        help="Mosaic mask shape (default: ellipse).",
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help="Minimum detection confidence (default: 0.5).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="YOLO inference image size (default: 640).",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="FP16 inference on CUDA.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Also run a CPU-vs-GPU benchmark on a clipped subset.",
    )
    parser.add_argument(
        "--benchmark-frames", type=int, default=240,
        help="Frames to use for the benchmark subclip (default: 240).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only log warnings and errors.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    os.makedirs(args.report_dir, exist_ok=True)

    proc = FaceMosaicProcessor(
        mosaic_size=args.mosaic_size,
        blur_shape=args.blur_shape,
        device=args.device,
        backend=args.backend,
        batch_size=args.batch_size,
        confidence=args.confidence,
        half=args.half,
        imgsz=args.imgsz,
    )
    report = proc.process_video(
        args.input,
        args.output,
        report_path=os.path.join(args.report_dir, "report.json"),
        collect_metrics=True,
    )
    paths = render_plots(report, args.report_dir)

    print("\nWrote:")
    for k, v in paths.items():
        print(f"  {k:>18}: {v}")

    if args.benchmark:
        bench = benchmark_cpu_vs_gpu(
            args.input,
            os.path.join(args.report_dir, "benchmark"),
            mosaic_size=args.mosaic_size,
            blur_shape=args.blur_shape,
            batch_size=args.batch_size,
            confidence=args.confidence,
            backend=args.backend,
            half=args.half,
            max_frames=args.benchmark_frames,
        )
        print("\nBenchmark:")
        print(json.dumps(bench, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
