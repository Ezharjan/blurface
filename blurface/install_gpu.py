"""Diagnose — and optionally repair — your PyTorch GPU install.

This helper inspects the active Python environment, reports the installed
``torch`` build and the visible accelerators, and (with ``--fix``) reinstalls
``torch`` + ``torchvision`` from the correct PyTorch wheel index.

Run it after installing Blurface:

    # diagnose only
    blurface-install-gpu

    # reinstall the CUDA 12.1 wheels (most common case)
    blurface-install-gpu --fix --cuda 12.1

    # newer NVIDIA architectures (e.g. RTX 50-series / Blackwell, sm_120)
    blurface-install-gpu --fix --nightly --cuda 13.0

    # roll back to the CPU build
    blurface-install-gpu --fix --cpu

Equivalent module form (handy when the console script isn't on ``PATH``):

    python -m blurface.install_gpu --fix --cuda 12.1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def diagnose() -> dict:
    """Print a compact diagnostic report and return the collected facts."""
    info: dict = {}

    _print_header("Python & environment")
    print(f"  python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  conda env   : {os.environ.get('CONDA_DEFAULT_ENV', '(none)')}")
    print(f"  platform    : {sys.platform}")

    _print_header("PyTorch")
    try:
        import torch  # noqa: WPS433 — intentional lazy import
    except ImportError:
        info["torch_version"] = None
        print("  torch       : not installed")
        print("  (Install with `pip install blurface` or `pip install torch`.)")
    else:
        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        print(f"  torch       : {torch.__version__}")
        print(f"  CUDA build  : {torch.version.cuda}")
        print(f"  cuda avail. : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  device[{i}]   : {name}  (sm_{cap[0]}{cap[1]}, {mem:.1f} GB)")
        mps_ok = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        info["mps_available"] = mps_ok
        if mps_ok:
            print("  MPS         : available (Apple Metal)")

    _print_header("NVIDIA driver")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi"], stderr=subprocess.STDOUT, text=True
            )
            for line in out.splitlines()[:14]:
                print(f"  {line}")
            info["nvidia_smi"] = True
        except Exception as exc:
            print(f"  nvidia-smi failed: {exc}")
            info["nvidia_smi"] = False
    else:
        print("  nvidia-smi  : not found on PATH (no NVIDIA driver visible)")
        info["nvidia_smi"] = False

    _print_header("ffmpeg")
    if shutil.which("ffmpeg"):
        print(f"  ffmpeg      : {shutil.which('ffmpeg')}")
    else:
        print("  ffmpeg      : not found on PATH (audio passthrough will be skipped)")

    _print_header("Recommendation")
    if info.get("cuda_available"):
        print("  PyTorch sees CUDA. You're ready to run with `--device cuda`.")
    elif info.get("nvidia_smi"):
        print("  An NVIDIA driver is present, but PyTorch is the CPU build.")
        print("  Reinstall PyTorch with CUDA wheels:")
        print("    blurface-install-gpu --fix --cuda 12.1")
        print("  (Or pass --nightly --cuda 13.0 for very recent GPUs.)")
    elif info.get("mps_available"):
        print("  Use `--device mps` (Apple Silicon).")
    else:
        print("  No GPU detected. You can still run with `--device cpu`, just slower.")
    return info


def reinstall(
    cuda: str = "12.1",
    cpu_only: bool = False,
    nightly: bool = False,
) -> int:
    """Reinstall ``torch`` + ``torchvision`` from the right wheel index."""
    pip = [sys.executable, "-m", "pip"]
    print("\n>> Uninstalling existing torch / torchvision ...")
    subprocess.run(pip + ["uninstall", "-y", "torch", "torchvision"], check=False)

    if cpu_only:
        index = "https://download.pytorch.org/whl/cpu"
        extra_args: list = []
    else:
        cuda_tag = cuda.replace(".", "")
        base = (
            "https://download.pytorch.org/whl/nightly"
            if nightly
            else "https://download.pytorch.org/whl"
        )
        index = f"{base}/cu{cuda_tag}"
        extra_args = ["--pre", "--upgrade"] if nightly else []

    print(f"\n>> Installing torch + torchvision from {index} ...")
    cmd = pip + ["install", *extra_args, "torch", "torchvision", "--index-url", index]
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(
            "\nInstall failed. Pick a different CUDA version from "
            "https://pytorch.org/get-started/locally/ and re-run with --cuda VERSION."
        )
    return rc


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="blurface-install-gpu",
        description=(
            "Diagnose the active PyTorch install and (optionally) reinstall "
            "torch + torchvision from the correct wheel index for your GPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  blurface-install-gpu                     # diagnose only\n"
            "  blurface-install-gpu --fix --cuda 12.1   # reinstall with CUDA 12.1\n"
            "  blurface-install-gpu --fix --nightly --cuda 13.0\n"
            "  blurface-install-gpu --fix --cpu         # roll back to CPU\n"
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Reinstall torch with the requested wheel after diagnosing.",
    )
    parser.add_argument(
        "--cuda",
        default="12.1",
        help="CUDA version tag, e.g. 11.8, 12.1, 12.4, 13.0 (default: 12.1).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU wheels (ignores --cuda).",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Use PyTorch nightly builds (needed for very new GPU architectures).",
    )
    args = parser.parse_args(argv)

    diagnose()
    if not args.fix:
        return 0
    return reinstall(cuda=args.cuda, cpu_only=args.cpu, nightly=args.nightly)


if __name__ == "__main__":
    raise SystemExit(main())
