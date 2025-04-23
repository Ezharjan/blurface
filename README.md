# Blurface

Blurface is a command-line tool to blur human faces in MP4 videos using face detection powered by the `retina-face` library. It supports customizable mosaic block sizes and allows users to choose between rectangular or elliptical blur shapes, with dynamic output naming for convenience.

## Features
- Detects and blurs faces in MP4 videos using `retina-face`.
- Supports rectangular or elliptical blur shapes (default: ellipse).
- Adjustable mosaic block size for blur intensity.
- Preserves audio in the output video.
- Dynamic output naming (e.g., `input_name + timestamp.mp4`) or user-specified paths.
- GPU acceleration with PyTorch if available.

## Installation

Blurface requires Python 3.9 or higher. We recommend using a Conda environment for installation.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ezharjan/blurface.git
   cd blurface
   ```

2. **Create and activate a Conda environment**:
   ```bash
   conda create -n blurface python=3.11
   conda activate blurface
   ```

3. **Install FFmpeg**:
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via Chocolatey (`choco install ffmpeg`). Add `ffmpeg.exe` to your PATH.
   - **Linux/macOS**: Install via package manager (e.g., `sudo apt install ffmpeg` or `brew install ffmpeg`).
   - Verify: `ffmpeg -version`

4. **Install Blurface**:
   ```bash
   pip install -e .
   ```

## Dependencies
- `retina-face==0.0.17`: Face detection library.
- `tensorflow>=2.10.0`: Required by `retina-face`.
- `tf-keras>=2.19.0`: Required by `retina-face` for TensorFlow 2.19.0+.
- `tqdm>=4.60.0`: Progress bar for processing.
- `torch>=1.9.0,<2.5.0`: GPU-accelerated mosaic effect.
- `ffmpeg-python>=0.2.0`: Audio handling in videos.

## Usage

Run `blurface` from the command line with the following arguments:

```bash
blurface <input> [--mosaic-size SIZE] [--output OUTPUT] [--blur-shape SHAPE]
```

### Arguments
- `input`: Path to the input MP4 video file (required).
- `--mosaic-size SIZE`, `-m`: Size of mosaic blocks for blurring (default: 10).
- `--output OUTPUT`, `-o`: Path to save the output MP4 video (default: `input_name + timestamp.mp4`, e.g., `movie2504231815.mp4`).
- `--blur-shape SHAPE`, `-s`: Shape of the blur area, either `rectangle` or `ellipse` (default: `ellipse`).

### Examples
1. **Blur faces with default settings (elliptical blur)**:
   ```bash
   blurface example_vids/test.mp4
   ```
   - Output: `test2504231815.mp4` (timestamp-based, e.g., 2025-04-23 18:15) with elliptical blur.

2. **Blur faces with rectangular blur and custom mosaic size**:
   ```bash
   blurface example_vids/test.mp4 --mosaic-size 15 --blur-shape rectangle
   ```
   - Output: `test2504231815.mp4` with rectangular blur.

3. **Specify output path and elliptical blur**:
   ```bash
   blurface example_vids/test.mp4 --mosaic-size 10 --blur-shape ellipse --output output/blurred.mp4
   ```
   - Output: `output/blurred.mp4` with elliptical blur; `output` directory is created if it doesn't exist.

### Error Handling
- Input must be an MP4 file, or an error is raised: `ERROR: Input file must be an MP4 video.`
- If the input file doesn't exist: `ERROR: Input video not found: <path>.`

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For issues or contributions, visit the [GitHub repository](https://github.com/Ezharjan/blurface) or contact Alexander Ezharjan at mysoft@111.com.