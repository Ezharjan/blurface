from setuptools import setup, find_packages
import os
import blurface


def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
        return f.read()


setup(
    name="blurface",
    version=blurface.__version__,
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "retina-face==0.0.17",
        "tensorflow>=2.10.0",
        "tf-keras>=2.19.0",
        "tqdm>=4.60.0",
        "torch>=1.9.0,<2.5.0",
        "ffmpeg-python>=0.2.0",
    ],
    description="A command-line tool to blur human faces in MP4 videos using face detection.",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="Alexander Ezharjan",
    author_email="mysoft@111.com",
    url="https://github.com/Ezharjan/blurface",
    license="MIT",
    entry_points={
        "console_scripts": [
            "blurface = blurface.__main__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)