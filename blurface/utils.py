import cv2
import numpy as np
import torch
from retinaface import RetinaFace
from tqdm import tqdm
import ffmpeg
import os
import tempfile
import shutil
import logging


class FaceMosaicProcessor:
    def __init__(self, mosaic_size=10, blur_shape="ellipse"):
        """Initialize the Face Mosaic Processor.

        Args:
            mosaic_size (int): Size of mosaic blocks for face blurring.
            blur_shape (str): Shape of the blur area ('rectangle' or 'ellipse').
        """
        self.mosaic_size = max(1, mosaic_size)
        self.blur_shape = blur_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def apply_mosaic(self, image, x1, y1, x2, y2):
        """Apply mosaic effect to a specified region of the image.

        Args:
            image: Input image in BGR format.
            x1, y1, x2, y2: Coordinates of the region to apply mosaic.

        Returns:
            Image with mosaic applied to specified region.
        """
        if x2 <= x1 or y2 <= y1:
            return image

        # Extract face region
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return image

        # Convert to torch tensor and move to device
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float().to(self.device)

        # Downscale and upscale for mosaic effect
        small = torch.nn.functional.interpolate(
            face_tensor.unsqueeze(0),
            size=(self.mosaic_size, self.mosaic_size),
            mode="bilinear",
            align_corners=False,
        )
        mosaic = torch.nn.functional.interpolate(
            small, size=(y2 - y1, x2 - x1), mode="nearest"
        )

        # Convert back to numpy
        mosaic_np = mosaic.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        if self.blur_shape == "rectangle":
            # Apply rectangular mosaic
            image[y1:y2, x1:x2] = mosaic_np
        else:
            # Apply elliptical mosaic
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            center = ((x2 - x1) // 2, (y2 - y1) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) // 255
            image[y1:y2, x1:x2] = image[y1:y2, x1:x2] * (1 - mask) + mosaic_np * mask

        return image

    def process_frame(self, frame_bgr):
        """Process a single frame to detect and blur faces.

        Args:
            frame_bgr: Input frame in BGR format.

        Returns:
            Processed frame with faces blurred.
        """
        # Convert frame to temporary JPG in memory
        _, buffer = cv2.imencode(".jpg", frame_bgr)
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_file.write(buffer)
        temp_file.close()

        # Detect faces
        try:
            resp = RetinaFace.detect_faces(temp_file.name)
        except Exception as e:
            logging.warning(f"Face detection failed: {e}")
            os.unlink(temp_file.name)
            return frame_bgr

        # Apply mosaic blur to detected faces
        result_bgr = frame_bgr.copy()
        if isinstance(resp, dict):
            for key in resp.keys():
                if not isinstance(resp[key], dict) or "facial_area" not in resp[key]:
                    continue
                facial_area = resp[key]["facial_area"]
                x1, y1, x2, y2 = facial_area
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    result_bgr = self.apply_mosaic(result_bgr, x1, y1, x2, y2)

        os.unlink(temp_file.name)
        return result_bgr

    def process_video(self, video_path, output_path):
        """Process an MP4 video to blur faces and save the output.

        Args:
            video_path: Path to the input MP4 video file.
            output_path: Path to save the output MP4 video.
        """
        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        if not video_path.lower().endswith(".mp4"):
            raise ValueError("Input file must be an MP4 video.")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_video = os.path.join(temp_dir, "temp_video.mp4")

        try:
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            temp_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

            # Process frames
            with tqdm(
                total=total_frames, desc="Processing frames", unit="frame"
            ) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame = self.process_frame(frame)
                    temp_writer.write(processed_frame)
                    pbar.update(1)

            cap.release()
            temp_writer.release()

            # Handle audio using ffmpeg-python
            input_stream = ffmpeg.input(video_path)
            video_stream = ffmpeg.input(temp_video)
            output_args = {
                "vcodec": "libx264",
                "preset": "slow",
                "crf": "18",
                "acodec": "copy" if input_stream["a"] else None,
            }
            output = ffmpeg.output(
                video_stream,
                input_stream["a"] if input_stream["a"] else None,
                output_path,
                **{k: v for k, v in output_args.items() if v is not None},
            )
            ffmpeg.run(output, overwrite_output=True)
            logging.info(f"Video processing complete. Output saved to {output_path}")

        except Exception as e:
            logging.error(f"Error processing video: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
