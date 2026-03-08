"""Images to PLY conversion using ml-sharp.

Wraps the ml-sharp CLI tool (Apple's SHARP model) to generate 3DGS PLY files
from input images.
"""

import os
import re
import shutil
import subprocess
from typing import Callable, Optional


def check_sharp_installed() -> bool:
    """Check if the sharp CLI is available."""
    return shutil.which("sharp") is not None


def generate_ply(
    images_folder: str,
    output_folder: str,
    device: str = "cuda",
    progress_callback: Optional[Callable[[str], None]] = None,
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[str]:
    """Generate PLY files from images using ml-sharp.

    Args:
        images_folder: Folder containing input images.
        output_folder: Folder for output PLY files.
        device: Compute device ("cuda" or "cpu").
        progress_callback: Callback for log messages.
        frame_progress_callback: Callback (current, total) for progress tracking.

    Returns:
        List of generated PLY file paths.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    if not check_sharp_installed():
        raise RuntimeError(
            "sharp CLI not found.\n"
            "Install ml-sharp with: pip install -e ml-sharp"
        )

    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        "sharp", "predict",
        "-i", images_folder,
        "-o", output_folder,
        "--no-render",
    ]

    if device == "cpu":
        cmd.extend(["--device", "cpu"])

    log(f"Running: {' '.join(cmd)}")
    log("This may take a while...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    total_images = 0
    processed = 0

    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        log(f"  [ml-sharp] {line}")

        # Parse "Processing N valid image files."
        m = re.search(r"Processing (\d+) valid image files", line)
        if m:
            total_images = int(m.group(1))

        # Parse "Processing <path>" (each image)
        if "Processing " in line and ("jpg" in line.lower() or "png" in line.lower() or "jpeg" in line.lower() or "heic" in line.lower()):
            processed += 1
            if frame_progress_callback and total_images > 0:
                frame_progress_callback(processed, total_images)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"ml-sharp failed with exit code {process.returncode}")

    # List output PLY files
    output_files = sorted([
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.lower().endswith(".ply")
    ])

    log(f"Generated {len(output_files)} PLY files")
    return output_files
