# Blurface

[![PyPI version](https://img.shields.io/pypi/v/blurface.svg)](https://pypi.org/project/blurface/)
[![Python versions](https://img.shields.io/pypi/pyversions/blurface.svg)](https://pypi.org/project/blurface/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Blurface** is a cross-platform command-line tool — and a tiny Python library — that **blurs every human face in an MP4 video** with a fully GPU-accelerated PyTorch pipeline. The default detector is **YOLOv8-face** via [`ultralytics`](https://github.com/ultralytics/ultralytics) (a state-of-the-art single-stage detector, robust on moving and partially-occluded faces); a lighter **`facenet-pytorch` MTCNN** backend is available as a fallback. The pixel mosaic is computed on the GPU with `torch.nn.functional.interpolate`, and the original audio track is re-muxed back into the output via `ffmpeg`. A built-in evaluation module emits a CSV, a JSON metrics report, and six PNG plots so you can quantify every run.

## Highlights

- **Pure PyTorch, end-to-end.** No TensorFlow anywhere on the hot path. Detection and mosaic both live on the same `torch.device`.
- **State-of-the-art detector for motion.** Default backend is YOLOv8-face — single forward pass per frame, low jitter on moving faces, no `transformers` import noise.
- **Cross-platform GPU acceleration.** Auto-selects CUDA on Windows / Linux, MPS on Apple Silicon, CPU otherwise — with graceful fallback.
- **Batched inference + FP16.** Set `--batch-size` to whatever your GPU can hold; add `--half` for FP16 on CUDA.
- **Rectangular or elliptical mosaic** with a configurable block size.
- **Audio passthrough** via the `ffmpeg` CLI (preferred) or `ffmpeg-python` (fallback).
- **Built-in evaluation.** Per-frame metrics CSV + JSON summary + six PNG plots and an optional CPU-vs-GPU benchmark.
- **Three console scripts.** `blurface`, `blurface-eval`, and `blurface-install-gpu` are registered on install.

## Table of contents

- [Blurface](#blurface)
  - [Highlights](#highlights)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [1. Create / activate a Python environment](#1-create--activate-a-python-environment)
    - [2. Install PyTorch — with CUDA wheels if you have an NVIDIA GPU](#2-install-pytorch--with-cuda-wheels-if-you-have-an-nvidia-gpu)
    - [3. Install Blurface](#3-install-blurface)
    - [4. Install FFmpeg](#4-install-ffmpeg)
  - [Verify your GPU](#verify-your-gpu)
  - [Usage: `blurface` CLI](#usage-blurface-cli)
    - [Worked examples](#worked-examples)
  - [Python API](#python-api)
  - [Pipeline internals](#pipeline-internals)
    - [Performance knobs](#performance-knobs)
  - [Detection methods explained](#detection-methods-explained)
    - [YOLOv8-face (default, `--backend yolo`)](#yolov8-face-default---backend-yolo)
    - [facenet-pytorch MTCNN (fallback, `--backend mtcnn`)](#facenet-pytorch-mtcnn-fallback---backend-mtcnn)
    - [`--backend auto`](#--backend-auto)
  - [Evaluation: `blurface-eval`](#evaluation-blurface-eval)
  - [Metrics reference](#metrics-reference)
  - [GPU diagnostic: `blurface-install-gpu`](#gpu-diagnostic-blurface-install-gpu)
  - [Testing](#testing)
  - [Troubleshooting](#troubleshooting)
  - [Changelog](#changelog)
    - [0.2.0 — 2026](#020--2026)
    - [0.1.0](#010)
  - [License](#license)
  - [Contact](#contact)

## Installation

Blurface targets **Python ≥ 3.9** and is verified on Windows, Linux, and macOS.

### 1. Create / activate a Python environment

```bash
# Recommended: a clean conda env
conda create -n blurface python=3.11 -y
conda activate blurface
```

### 2. Install PyTorch — with CUDA wheels if you have an NVIDIA GPU

This is the single most common failure point. The default `pip install torch` on Windows installs the **CPU build**, which is why `--device cuda` would otherwise refuse to run.

**NVIDIA GPU (recommended) — CUDA 12.1 wheels:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Newer GPUs (e.g. RTX 50-series / Blackwell, `sm_120` compute):**
> Standard CUDA 12.1/12.4 builds will lack your GPU's kernel architecture and crash with `CUDA error: no kernel image is available`. Install the PyTorch nightly bundled with CUDA 13.0 (or newer):
> ```bash
> pip install --pre torch torchvision \
>     --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade
> ```

If your NVIDIA driver is older, you may need `cu118` instead. Check with `nvidia-smi` and the official [PyTorch install matrix](https://pytorch.org/get-started/locally/).

**Apple Silicon (MPS):**

```bash
pip install torch torchvision
```

**CPU only:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Blurface

From PyPI:

```bash
pip install blurface
```

Or from a git clone (editable):

```bash
git clone https://github.com/Ezharjan/blurface.git
cd blurface
pip install -e .
```

This pulls `ultralytics`, `opencv-python`, `ffmpeg-python`, `matplotlib`, `pandas`, `tqdm`, `Pillow`, … and registers three console scripts: `blurface`, `blurface-eval`, `blurface-install-gpu`.

**Optional MTCNN fallback backend:**

```bash
pip install "blurface[mtcnn]"
```

### 4. Install FFmpeg

The audio re-mux step needs the `ffmpeg` binary on `PATH`:

| Platform | Command |
|---|---|
| Windows | `choco install ffmpeg` (or download from <https://ffmpeg.org/download.html> and add `ffmpeg.exe` to `PATH`) |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` |

If `ffmpeg` isn't available the pipeline still produces a video-only MP4 — it just skips the audio.

## Verify your GPU

After installation, run the diagnostic:

```bash
blurface-install-gpu
```

You should see something like:

```
========================================================================
PyTorch
========================================================================
  torch       : 2.4.1+cu121
  CUDA build  : 12.1
  cuda avail. : True
  device[0]   : NVIDIA GeForce RTX 4090  (sm_89, 24.0 GB)
```

If `cuda avail.` is `False` but `nvidia-smi` works, you're on the CPU build of torch — repair it with:

```bash
blurface-install-gpu --fix --cuda 12.1
```

The same script also accepts `--cpu` (force CPU wheels) and `--nightly` (use the PyTorch nightly index for very new architectures).

## Usage: `blurface` CLI

```bash
blurface <input.mp4> [options]
```

The most common flags:

| Flag | Default | Description |
|---|---|---|
| `input` | — | Path to the input MP4 video (required). |
| `--output`, `-o` | `<stem><YYMMDDHHMM>.mp4` | Output file path. |
| `--mosaic-size`, `-m` | `10` | Mosaic block size in pixels; higher = coarser blur. |
| `--blur-shape`, `-s` | `ellipse` | `ellipse` or `rectangle`. |
| `--device`, `-d` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |
| `--backend` | `auto` | `auto` (→ yolo), `yolo`, or `mtcnn`. |
| `--batch-size`, `-b` | `8` | Frames per detection batch. |
| `--half` | off | FP16 inference on CUDA. |
| `--confidence`, `-c` | `0.5` | Minimum face confidence in `[0, 1]`. |
| `--imgsz` | `640` | YOLO inference image size. Raise for tiny faces, lower for speed. |
| `--min-face-size` | `20` | MTCNN minimum face edge in px. |
| `--model-path` | — | Local YOLO-face `.pt` file (skips the download). |
| `--model-url` | — | Custom URL for YOLO-face weights. |
| `--no-cpu-fallback` | off | Hard-fail when CUDA/MPS is requested but unavailable. |
| `--report` | — | Path for a JSON metrics report. |
| `--plots-dir` | — | If set, evaluation PNGs and CSV are written here. |
| `--quiet` / `--verbose` | off | Lower / raise the log level. |
| `--version` | — | Print the installed version and exit. |

Run `blurface --help` for the full reference and worked examples.

### Worked examples

```bash
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

# 7. Raise the inference image size for lots of tiny faces.
blurface input.mp4 --imgsz 1280 --batch-size 4

# 8. Full evaluation: report + plots + CPU-vs-GPU benchmark
blurface-eval video.mp4 --output D:\blurface\out\blurred.mp4 --report-dir D:\blurface\out\report --device auto --batch-size 8 --benchmark --benchmark-frames 120
```

## Python API

```python
from blurface import FaceMosaicProcessor
from blurface.evaluate import render_plots

proc = FaceMosaicProcessor(
    device="auto",        # cuda > mps > cpu, with fallback
    backend="yolo",       # or "mtcnn", or "auto"
    batch_size=16,
    half=True,            # FP16 on CUDA (no-op elsewhere)
    imgsz=640,
    confidence=0.5,
)

report = proc.process_video(
    "input.mp4", "output.mp4",
    report_path="out/report.json",
    collect_metrics=True,
)

render_plots(report, "out/plots")
print(f"{report.realtime_fps:.1f} fps on {report.device} ({report.backend})")
```

Public objects re-exported from the top-level package:

- `FaceMosaicProcessor` — the pipeline.
- `RunReport`, `FrameMetric` — dataclasses returned by `process_video`.
- `select_device(preferred, allow_cpu_fallback)` — the device picker.
- `describe_device(device)` — human-readable device label.
- `build_detector(...)`, `YoloFaceDetector`, `MtcnnDetector` — detection backends.

## Pipeline internals

The video is processed in five clearly-separated stages, kept on the same `torch.device` to avoid host round-trips:

1. **Decode (CPU).** `cv2.VideoCapture` reads MP4 frames as BGR `uint8` numpy arrays. Frames are accumulated into a list of length `--batch-size`.
2. **Detect (device).** The batch is converted to RGB and handed to the active detector backend. The detector returns, per frame, an `(N, 4)` array of `[x1, y1, x2, y2]` boxes in original pixel space and an `(N,)` array of confidences.
3. **Mosaic (device).** Each frame is uploaded once to the device as a CHW float tensor (FP16 if `--half`). For every box:
   - the cropped face region is **down-sampled** to `mosaic_size × mosaic_size` with `F.interpolate(mode="bilinear", align_corners=False)`;
   - it is then **up-sampled** back to the box size with `F.interpolate(mode="nearest")` — that's the classic pixelation effect, computed in a single bilinear + nearest kernel pair;
   - for `blur_shape="ellipse"` an inscribed elliptical mask is built on-device (`(x − cx)² / rx² + (y − cy)² / ry² ≤ 1`) and the mosaic is alpha-blended over the original — only the elliptical region is replaced, the corners of the bounding box are preserved.
4. **Encode (CPU).** The blurred frame is clamped, cast back to `uint8`, transposed to HWC, copied to the CPU, and written to a temporary `mp4v`-encoded MP4 with `cv2.VideoWriter`.
5. **Mux (FFmpeg).** Finally `ffmpeg` re-encodes the temporary video as **H.264 (libx264, CRF 20, `medium` preset)** and **stream-copies the original audio track** with `-c:a copy -map 0:v:0 -map 1:a:0?`. The audio is preserved **bit-for-bit** — no re-encoding, no quality loss, same codec / bitrate / sample rate as the source. If stream-copy is rejected (rare; happens when the source audio codec isn't allowed in the MP4 container, e.g. PCM) Blurface falls back to a 192 kbit/s AAC re-encode. `ffprobe` then verifies the output actually contains audio when the source did — mismatches raise rather than silently producing a muted file. If `ffmpeg` is missing and the source has audio, Blurface fails loudly with install instructions instead of dropping the audio.

Throughout the run, optional per-frame metrics (detect / mosaic latency, GPU memory, face counts, mean confidence) are collected into a `RunReport`, which `render_plots` turns into PNG charts and a CSV.

```
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐
│  decode  │ → │   detect     │ → │   mosaic     │ → │  encode  │ → │   mux    │
│ (cv2)    │   │ (YOLO/MTCNN) │   │ (torch.F)    │   │ (cv2)    │   │ (ffmpeg) │
│  CPU     │   │   device     │   │   device     │   │   CPU    │   │   CPU    │
└──────────┘   └──────────────┘   └──────────────┘   └──────────┘   └──────────┘
                       │                  │
                       ▼                  ▼
                  per-frame metrics ──→ RunReport ──→ CSV / JSON / PNG plots
```

### Performance knobs

- **`--batch-size`** is the single biggest lever once CUDA is enabled. Raise it until you hit your GPU's memory limit.
- **`--half`** roughly halves the detector's memory footprint on CUDA and is faster on Ampere/Ada/Hopper. It has no effect on CPU or MPS.
- **`--imgsz`** trades detector accuracy for speed. Default 640 is a good compromise; 1280 helps on tiny faces in 4K footage; 480 is markedly faster on tight latency budgets.
- **`--mosaic-size`** is *not* a speed knob — the down-sample target is tiny either way — but it changes the visual effect. 4–8 = strongly recognisable as pixelation; 12–20 = blocky, friendlier on small faces; 30+ = single coloured patch.

## Detection methods explained

Blurface ships two interchangeable backends with the same `detect(frames_rgb)` API.

### YOLOv8-face (default, `--backend yolo`)

A single-stage anchor-free detector built on Ultralytics' YOLOv8 backbone, fine-tuned on a face-detection dataset. Why it is the default:

- **Single forward pass per frame.** Detection is a single conv-net evaluation, so latency stays flat as the number of faces grows. Cascade detectors (MTCNN, Haar, etc.) keep proposing and refining candidates, which inflates per-frame cost on busy scenes.
- **Robust to motion blur, profile angles and partial occlusion.** The anchor-free head and the deep backbone learn richer face priors than the small classification networks inside MTCNN's P/R/O stages.
- **Lower jitter across frames.** Because the model is deeper and operates at a single scale per call, box positions are noticeably more stable from frame to frame than MTCNN's, giving smoother mosaics in the output.
- **GPU-friendly.** Batched inference on CUDA is the design point; FP16 is a one-flag switch.

Weights (`yolov8n-face.pt`, ~6 MB) are downloaded once from the [`akanametov/yolo-face`](https://github.com/akanametov/yolo-face) release into `~/.cache/blurface/` and reused on subsequent runs. Override with `--model-path` or `--model-url`.

### facenet-pytorch MTCNN (fallback, `--backend mtcnn`)

A three-stage cascade detector (P-Net → R-Net → O-Net) from [facenet-pytorch](https://github.com/timesler/facenet-pytorch). Useful when:

- you cannot install `ultralytics` (e.g. very old Python, restricted environments),
- you want a second opinion on a hard clip,
- you specifically need MTCNN's facial landmark output (landmarks are computed internally but not exposed by Blurface today),
- you're CPU-only and prefer MTCNN's lighter memory footprint.

Trade-offs: MTCNN is **slower** per frame on GPU than YOLOv8-face, **less robust on motion-blurred or sideways faces**, and produces **more frame-to-frame jitter**. The `--min-face-size` flag is honoured only by this backend.

Install with `pip install "blurface[mtcnn]"`.

### `--backend auto`

Tries YOLOv8-face first; if its `ultralytics` import or weight download fails, falls back to MTCNN. This is the default.

## Evaluation: `blurface-eval`

`blurface-eval` runs the full pipeline and writes a complete report directory:

```bash
blurface-eval input.mp4 \
    --output out/blurred.mp4 \
    --report-dir out/report \
    --device cuda --half --batch-size 16 \
    --benchmark --benchmark-frames 240
```

It accepts the same backend / device / mosaic options as `blurface`, plus `--benchmark` and `--benchmark-frames N`, which produce a CPU-vs-GPU bar chart on a short subclip. Run `blurface-eval --help` for the full reference.

The output directory ends up looking like:

```
out/report/
├── report.json                   # full RunReport (incl. per-frame metrics)
├── summary.json                  # aggregate scorecard
├── per_frame_metrics.csv         # one row per processed frame
├── summary.png                   # text scorecard, ready to share
├── faces_per_frame.png           # detections across the timeline
├── latency_per_frame.png         # detect vs mosaic vs total latency
├── fps_rolling.png               # rolling throughput vs source FPS
├── gpu_memory.png                # allocated GPU memory (CUDA only)
├── confidence_histogram.png      # distribution of per-frame mean confidence
└── benchmark/                    # only with --benchmark
    ├── cpu_vs_gpu.png
    ├── cpu_vs_gpu.json
    ├── benchmark_cpu.mp4
    └── benchmark_cuda.mp4
```

## Metrics reference

Every run produces, conceptually, three artefacts:

- **`report.json`** — the full `RunReport` dataclass: device, backend, source resolution / FPS, frames processed, processing FPS, total wall time, detect / mosaic / mux time breakdowns, total faces detected, average faces per frame, frames with faces, peak GPU memory, batch size, FP16 flag, mosaic configuration, confidence threshold, and the full per-frame metrics list.
- **`per_frame_metrics.csv`** — one row per processed frame with columns:
  `frame_idx, num_faces, mean_confidence, detect_ms, mosaic_ms, total_ms, gpu_mem_mb`.
- **PNG plots**, each focused on a single question:
  - *faces_per_frame.png* — how many faces were detected across the timeline.
  - *latency_per_frame.png* — detect vs mosaic vs total latency per frame.
  - *fps_rolling.png* — rolling throughput, overlaid with the source FPS line and the run's average processing FPS.
  - *gpu_memory.png* — allocated GPU memory over time (CUDA only).
  - *confidence_histogram.png* — distribution of per-frame mean detection confidences (on frames that had faces).
  - *summary.png* — a monospaced text scorecard you can drop into a slide.

## GPU diagnostic: `blurface-install-gpu`

A standalone helper to inspect and repair your PyTorch install:

```bash
# 1. Diagnose only (the default)
blurface-install-gpu

# 2. Reinstall with the right wheels for your CUDA driver
blurface-install-gpu --fix --cuda 12.1

# 3. Very new architectures (RTX 50-series / Blackwell, sm_120)
blurface-install-gpu --fix --nightly --cuda 13.0

# 4. Force the CPU build
blurface-install-gpu --fix --cpu
```

It reports Python, conda env, platform, PyTorch version + CUDA build, every visible CUDA device (with its compute capability and memory), MPS availability on Apple Silicon, the NVIDIA driver via `nvidia-smi`, and whether `ffmpeg` is on `PATH`. With `--fix`, it `pip uninstall`s torch + torchvision and reinstalls them from the appropriate wheel index.

Run as a module too: `python -m blurface.install_gpu`.

## Testing

A minimal pytest suite ships with the repo. It builds a tiny synthetic clip and runs the pipeline end-to-end on CPU — no GPU or face dataset required.

```bash
pip install pytest
pytest -q
```

Tests live in `tests/test_pipeline.py`.

## Troubleshooting

**`RuntimeError: CUDA requested but no CUDA device is available.`**
Your installed `torch` is the CPU build. Repair with the bundled diagnostic:

```bash
blurface-install-gpu --fix --cuda 12.1
```

…or manually:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**`CUDA error: no kernel image is available for execution on the device`**
Your GPU's compute capability is newer than the CUDA version your PyTorch was built against (typical on RTX 50-series / Blackwell). Use the nightly + CUDA 13 wheels:

```bash
blurface-install-gpu --fix --nightly --cuda 13.0
```

**`Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2`**
That's a warning emitted by the `transformers` library when *something else* in your environment imports it. Blurface's default `--backend yolo` does not pull `transformers` in, so the warning is harmless. If you need `--backend mtcnn` with an old torch, upgrade torch (see above) or pin `pip install "transformers<4.40"`.

**`ImportError: ultralytics is required for the YOLO backend.`**
`pip install ultralytics` — or simply `pip install blurface`, which already depends on it.

**`CUDA out of memory`.** Lower `--batch-size`, enable `--half`, or lower `--imgsz`.

**No audio in the output.** This should never happen silently in v0.2.0 — if the source has audio and `ffmpeg` can't preserve it, Blurface raises with install instructions. If you do see a muted output, first check: did the *source* have an audio track? (Run `ffprobe -i your_input.mp4` and look for a `Stream #0:1: Audio:` line.) If the source genuinely has no audio, the muted output is correct. If the source does have audio and you got a muted output anyway, please file a bug at <https://github.com/Ezharjan/blurface/issues>.

**macOS MPS warnings about unimplemented ops.** Harmless — those ops automatically fall back to CPU.

**The downloaded YOLO weights file is corrupted / partial.** Delete `~/.cache/blurface/yolov8n-face.pt` and let the next run re-download, or pass `--model-path` to use a known-good copy.

## Changelog

### 0.2.0 — 2026

- **Audio preservation (bug fix).** Previously, three silent-failure paths in the mux step could quietly produce a **muted** output: the outer wrapper caught any ffmpeg error and copied the audio-less temp file, the ffmpeg-python fallback re-encoded video alone on failure, and even on the happy path the audio was *re-encoded* to AAC 192k (a quality loss). The mux now:
  - **Stream-copies the original audio** (`-c:a copy`) — preserved bit-for-bit, same codec / bitrate / sample rate as the source. No re-encoding.
  - Probes the source with `ffprobe` to decide whether to expect audio at all.
  - Falls back to AAC 192k *only* if stream-copy is rejected by the MP4 container.
  - Verifies the output actually contains audio when the source did; raises if not.
  - Raises a clear, actionable error (with install instructions) when ffmpeg is missing and the source has audio, instead of silently dropping the track.
- **Packaging:** `blurface-install-gpu` now ships inside the installed package, so the console script works after `pip install` (it was broken before). PyPI metadata (`project_urls`, `keywords`, full `classifiers`, MANIFEST, `pyproject.toml`) brought up to standard.
- **Pipeline:** fixed an aggregation bug where `RunReport.total_faces_detected`, `frames_with_faces`, `detect_time_s`, and `mosaic_time_s` were `0` when `process_video(..., collect_metrics=False)`. They are now tracked independently of the per-frame list.
- **Report:** new `frames_processed` and `total_faces_detected` fields on `RunReport`; `summary.json` and the PNG scorecard updated to match.
- **CLI:** richer `--help` output (epilog with worked examples), new `--verbose` flag, more actionable error messages, validated `--confidence` range, cleaner exit codes (`0`/`1`/`2`/`130`).
- **`blurface-install-gpu`:** lists every visible CUDA device (with compute capability + memory), reports `ffmpeg` presence, gains `--nightly` for new architectures, gains a module form (`python -m blurface.install_gpu`).
- **`blurface-eval`:** aligned defaults with `blurface` (confidence 0.5, benchmark-frames 240), exposes `--backend`, `--imgsz`, `--half`, `--quiet`.
- **Public API:** top-level package re-exports `select_device`, `describe_device`, `build_detector`, `YoloFaceDetector`, `MtcnnDetector` alongside the existing `FaceMosaicProcessor`, `RunReport`, `FrameMetric`.
- **Docs:** README rewritten with explicit pipeline-internals and detection-methods sections.

### 0.1.0

- Initial public release: GPU PyTorch pipeline, YOLOv8-face + MTCNN backends, FFmpeg audio re-mux, evaluation plots, `blurface` and `blurface-eval` CLIs.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Issues and PRs welcome at <https://github.com/Ezharjan/blurface>.
