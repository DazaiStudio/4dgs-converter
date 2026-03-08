"""Video to Images extraction using ffmpeg.

Extracts frames from a video file at uniform intervals.
"""

import math
import os
import subprocess
import shutil
from typing import Callable, Optional


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def get_video_frame_count(video_path: str) -> int:
    """Get the total number of frames in a video using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return int(result.stdout.strip())
    except Exception:
        # Fallback: try with nb_frames
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=nb_frames",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return int(result.stdout.strip())
        except Exception:
            return -1


def extract_frames(
    video_path: str,
    output_folder: str,
    frame_count: int = 100,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Extract frames from video using ffmpeg.

    Uses uniform sampling: selects every Nth frame to get approximately
    the desired number of output frames.

    Args:
        video_path: Path to input video.
        output_folder: Folder to save extracted frames.
        frame_count: Desired number of frames to extract.
        progress_callback: Callback for log messages.

    Returns:
        List of output file paths.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.\n"
            "Download from: https://ffmpeg.org/download.html"
        )

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    # Get total frame count to calculate step
    total_frames = get_video_frame_count(video_path)
    if total_frames > 0:
        step = max(1, total_frames // frame_count)
        log(f"Video has {total_frames} frames, extracting every {step}th frame")
    else:
        step = 1
        log(f"Could not determine frame count, extracting {frame_count} frames")

    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select=not(mod(n\\,{step}))",
        "-fps_mode", "vfr",
        "-frames:v", str(frame_count),
        "-q:v", "2",
        output_pattern,
        "-y",
    ]

    log(f"Running: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr}")

    # List output files
    output_files = sorted([
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.startswith("frame_") and f.endswith(".jpg")
    ])

    log(f"Extracted {len(output_files)} frames to {output_folder}")
    return output_files
