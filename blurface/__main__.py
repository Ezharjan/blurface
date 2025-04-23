#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from datetime import datetime
from blurface.utils import FaceMosaicProcessor


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Blur human faces in an MP4 video using face detection."
    )
    parser.add_argument("input", help="Path to the input MP4 video file.")
    parser.add_argument(
        "--mosaic-size",
        "-m",
        type=int,
        default=10,
        help="Size of mosaic blocks for face blurring (default: 10).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the output MP4 video (default: input_name + timestamp, e.g., movie2503231815.mp4).",
    )
    parser.add_argument(
        "--blur-shape",
        "-s",
        choices=["rectangle", "ellipse"],
        default="ellipse",
        help="Shape of the blur area: rectangle or ellipse (default: ellipse).",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.lower().endswith(".mp4"):
        parser.error("Input file must be an MP4 video.")

    # Set default output name if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        args.output = f"{base_name}{timestamp}.mp4"
    # Ensure output ends with .mp4
    elif not args.output.lower().endswith(".mp4"):
        args.output += ".mp4"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_cli_args()

    try:
        processor = FaceMosaicProcessor(
            mosaic_size=args.mosaic_size, blur_shape=args.blur_shape
        )
        processor.process_video(args.input, args.output)
        logging.info(f"Processed video saved as {args.output}")
    except Exception as e:
        logging.error(f"Failed to process {args.input}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
