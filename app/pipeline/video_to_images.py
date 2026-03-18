"""Video to Images extraction using ffmpeg.

Extracts frames from a video file at uniform intervals.
"""

import math
import os
import subprocess
import shutil
import sys
from typing import Callable, Optional

_CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _get_startupinfo():
    """Get startupinfo to hide console window on Windows."""
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        return si
    return None


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def get_video_frame_count(video_path: str) -> int:
    """Get the total number of frames in a video using ffprobe."""
    # Method 1: nb_frames (fast, not always available)
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, timeout=30,
            startupinfo=_get_startupinfo(), creationflags=_CREATE_FLAGS,
        )
        val = result.stdout.strip().rstrip(",")
        if val and val != "N/A":
            return int(val)
    except Exception:
        pass

    # Method 2: count_packets (slow but reliable)
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
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, timeout=120,
            startupinfo=_get_startupinfo(), creationflags=_CREATE_FLAGS,
        )
        val = result.stdout.strip().rstrip(",")
        if val and val != "N/A":
            return int(val)
    except Exception:
        pass

    # Method 3: duration * fps
    try:
        fps = get_video_fps(video_path)
        dur = _get_video_duration(video_path)
        if fps > 0 and dur > 0:
            return int(dur * fps)
    except Exception:
        pass

    return -1


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, timeout=10,
            startupinfo=_get_startupinfo(), creationflags=_CREATE_FLAGS,
        )
        val = result.stdout.strip().rstrip(",")
        if val and val != "N/A":
            return float(val)
    except Exception:
        pass
    return -1.0


def get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe. Returns -1.0 on error."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, timeout=10,
            startupinfo=_get_startupinfo(), creationflags=_CREATE_FLAGS,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().rstrip(",").split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
            return float(parts[0])
    except Exception:
        pass
    return -1.0


def extract_frames(
    video_path: str,
    output_folder: str,
    frame_count: int = 100,
    fps: float = 0,
    start_frame: int = 0,
    end_frame: int = -1,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Extract frames from video using ffmpeg.

    Args:
        video_path: Path to input video.
        output_folder: Folder to save extracted frames.
        frame_count: Total frames in video (for step calculation).
        fps: Target FPS (0 = use video's native FPS).
        start_frame: Start frame (0-based).
        end_frame: End frame (0-based, -1 = last).
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

    # Detect video FPS for time calculations
    video_fps = get_video_fps(video_path)
    if video_fps <= 0:
        video_fps = 30.0

    # Calculate time range
    ss_time = start_frame / video_fps if start_frame > 0 else 0
    if end_frame >= 0:
        duration = (end_frame - start_frame + 1) / video_fps
        extract_count = end_frame - start_frame + 1
    else:
        duration = 0
        extract_count = frame_count - start_frame

    # Build ffmpeg filter
    vf_parts = []
    if fps > 0:
        vf_parts.append(f"fps={fps}")

    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")

    cmd = ["ffmpeg"]
    if ss_time > 0:
        cmd += ["-ss", f"{ss_time:.3f}"]
    cmd += ["-i", video_path]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-q:v", "2", output_pattern, "-y"]

    log(f"Extracting frames {start_frame}-{end_frame if end_frame >= 0 else 'end'} ({extract_count} frames)")
    log(f"Running: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        startupinfo=_get_startupinfo(),
        creationflags=_CREATE_FLAGS,
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
