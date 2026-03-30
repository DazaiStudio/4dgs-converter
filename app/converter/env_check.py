"""Environment checker for Video-to-GSD converter dependencies."""

import os
import shutil
import subprocess
import sys

# Hide subprocess console windows on Windows
_CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def find_ffmpeg() -> str | None:
    """Find the ffmpeg executable."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Search winget install location on Windows
    if sys.platform == "win32":
        winget_dir = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft", "WinGet", "Packages",
        )
        if os.path.isdir(winget_dir):
            for pkg in os.listdir(winget_dir):
                if "ffmpeg" not in pkg.lower():
                    continue
                # Search for ffmpeg.exe inside the package
                pkg_path = os.path.join(winget_dir, pkg)
                for root, dirs, files in os.walk(pkg_path):
                    if "ffmpeg.exe" in files:
                        return os.path.join(root, "ffmpeg.exe")
    return None


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    # ffprobe should be next to ffmpeg
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    return os.path.isfile(ffprobe)


def find_sharp() -> str | None:
    """Find the sharp CLI executable."""
    exe = "sharp.exe" if sys.platform == "win32" else "sharp"
    # Check PATH first
    found = shutil.which("sharp")
    if found:
        return found
    # Also check current Python's Scripts directory (may not be in PATH)
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    candidate = os.path.join(scripts_dir, exe)
    if os.path.isfile(candidate):
        return candidate
    # Search other Python installations on Windows (e.g. sharp installed in a
    # different Python version than the one running the converter)
    if sys.platform == "win32":
        search_roots = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python"),
            "C:\\Python",
        ]
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for entry in os.listdir(root):
                candidate = os.path.join(root, entry, "Scripts", exe)
                if os.path.isfile(candidate):
                    return candidate
    return None


def check_sharp() -> bool:
    """Check if ml-sharp CLI is available."""
    sharp = find_sharp()
    if not sharp:
        return False
    try:
        result = subprocess.run(
            [sharp, "--help"],
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
