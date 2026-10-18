"""
Frames to Video Converter
-------------------------
Converts an image sequence folder (e.g., UCSD Ped2, Avenue, ShanghaiTech)
into a standard MP4 video file.

Usage example:
    python scripts/frames_to_video.py -i data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004 -o data/samples/ped2_sample.mp4 --fps 25
"""

import os
import sys
import glob
import argparse
import cv2
from natsort import natsorted
from tqdm import tqdm


def frames_to_video(
    input_dir: str,
    output_path: str,
    fps: float = 25.0,
    extensions: tuple = (".tif", ".png", ".jpg", ".jpeg", ".bmp"),
) -> str:
    """
    Combines sorted image frames from `input_dir` into an MP4 video file.

    Args:
        input_dir: Directory containing image files.
        output_path: Destination path for the output video (.mp4).
        fps: Playback frame rate (default: 25.0).
        extensions: Tuple of supported image extensions.

    Returns:
        output_path: Path to the created video.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Gather all image files
    all_files = os.listdir(input_dir)
    image_files = [
        os.path.join(input_dir, f)
        for f in all_files
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if not image_files:
        raise ValueError(
            f"No image files found in '{input_dir}' with extensions {extensions}"
        )

    # Sort naturally (e.g., 1, 2, ..., 10, 11 instead of 1, 10, 2)
    image_files = natsorted(image_files)

    # Read the first frame to determine video dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        raise ValueError(f"Failed to read first image: {image_files[0]}")

    h, w, c = first_frame.shape

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Initialize VideoWriter using standard mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for destination: {output_path}")

    print(f"[*] Found {len(image_files)} frames in '{input_dir}'")
    print(f"[*] Resolution: {w}x{h} | FPS: {fps} | Destination: {output_path}")

    for img_path in tqdm(image_files, desc="Encoding Video", unit="frame"):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[!] Warning: Could not read {img_path}, skipping.")
            continue
        # Resize if dimensions differ from first frame
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        out.write(frame)

    out.release()

    # Re-encode to standard H.264 (yuv420p) for universal browser and web-player support
    try:
        import subprocess
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_h264 = output_path + ".h264.mp4"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "veryfast",
            temp_h264,
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and os.path.exists(temp_h264):
            os.replace(temp_h264, output_path)
    except Exception:
        pass

    print(f"[OK] Successfully generated video: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an image folder to an MP4 video")
    parser.add_argument(
        "-i", "--input_dir", required=True, type=str, help="Path to folder containing image frames"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output MP4 file path"
    )
    parser.add_argument(
        "--fps", default=25.0, type=float, help="Playback frame rate (default: 25.0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frames_to_video(
        input_dir=args.input_dir,
        output_path=args.output,
        fps=args.fps,
    )
