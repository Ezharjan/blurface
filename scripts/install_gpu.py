"""Repository-level shim that calls the packaged installer (v0.2.0).

Historically the GPU diagnostic lived at ``scripts/install_gpu.py``. It now
lives inside the installed package at :mod:`blurface.install_gpu` so the
``blurface-install-gpu`` console script keeps working after ``pip install``.

This shim is preserved so that users who only clone the repository (and have
not yet run ``pip install -e .``) can still launch the diagnostic with::

    python scripts/install_gpu.py [--fix] [--cuda 12.1] [--cpu] [--nightly]

It forwards all arguments to :func:`blurface.install_gpu.main`.
"""

from __future__ import annotations

import os
import sys


def _ensure_importable() -> None:
    """Allow `import blurface.install_gpu` when running from a fresh clone."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> int:
    _ensure_importable()
    from blurface.install_gpu import main as _main  # noqa: WPS433

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
