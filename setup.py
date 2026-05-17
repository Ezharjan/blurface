"""Packaging for Blurface (v0.2.0)."""

from __future__ import annotations

import os

from setuptools import find_packages, setup

HERE = os.path.dirname(os.path.abspath(__file__))


def read_file(filename: str) -> str:
    with open(os.path.join(HERE, filename), encoding="utf-8") as fh:
        return fh.read()


def read_version() -> str:
    """Read ``__version__`` without importing torch at setup time."""
    ns: dict = {}
    init_path = os.path.join(HERE, "blurface", "__init__.py")
    with open(init_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                exec(line, ns)  # noqa: S102 — trusted local source
                break
    return ns.get("__version__", "0.0.0")


setup(
    name="blurface",
    version=read_version(),
    description=(
        "GPU-accelerated, cross-platform CLI to blur human faces in MP4 "
        "videos using PyTorch (YOLOv8-face / MTCNN backends), with audio "
        "passthrough and built-in evaluation metrics."
    ),
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="Alexander Ezharjan",
    author_email="machinelearningscholar@gmail.com",
    url="https://github.com/Ezharjan/blurface",
    project_urls={
        "Homepage": "https://github.com/Ezharjan/blurface",
        "Source": "https://github.com/Ezharjan/blurface",
        "Bug Tracker": "https://github.com/Ezharjan/blurface/issues",
        "Documentation": "https://github.com/Ezharjan/blurface#readme",
        "Changelog": "https://github.com/Ezharjan/blurface/blob/main/README.md#changelog",
    },
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*", "scripts", "scripts.*")),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        # Core deep-learning stack. For an explicit CUDA build, install
        # torch / torchvision separately (see the README and the bundled
        # `blurface-install-gpu` helper).
        "torch>=2.0",
        "torchvision>=0.15",
        # Default detector backend: SOTA YOLOv8-face via ultralytics.
        "ultralytics>=8.1",
        # Video / image I/O
        "opencv-python>=4.8.0",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.23",
        "Pillow>=9.0",
        # UX & reporting
        "tqdm>=4.60.0",
        "matplotlib>=3.6",
        "pandas>=1.5",
    ],
    extras_require={
        # Optional MTCNN fallback backend.
        "mtcnn": ["facenet-pytorch>=2.5.3"],
        # Developer tooling.
        "dev": [
            "pytest>=7.0",
            "build>=1.0",
            "twine>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "blurface = blurface.__main__:main",
            "blurface-eval = blurface.evaluate:main",
            "blurface-install-gpu = blurface.install_gpu:main",
        ],
    },
    keywords=[
        "face-detection",
        "face-blur",
        "video-processing",
        "yolo",
        "yolov8",
        "mtcnn",
        "pytorch",
        "gpu",
        "cuda",
        "mps",
        "ffmpeg",
        "anonymization",
        "privacy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
