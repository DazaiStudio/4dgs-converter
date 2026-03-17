"""Environment checker for Video-to-GSD converter dependencies."""

import os
import shutil
import subprocess
import sys

# Hide subprocess console windows on Windows
_CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def check_sharp() -> bool:
    """Check if ml-sharp CLI is available."""
    try:
        result = subprocess.run(
            ["sharp", "--help"],
            capture_output=True, text=True, timeout=10,
            creationflags=_CREATE_FLAGS,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_lz4() -> bool:
    """Check if lz4 Python package is available."""
    try:
        import lz4.block  # noqa: F401
        return True
    except ImportError:
        return False


def check_all() -> dict[str, bool]:
    """Check all dependencies. Returns dict of name -> available."""
    return {
        "ffmpeg": check_ffmpeg(),
        "sharp": check_sharp(),
        "lz4": check_lz4(),
    }
