# video2ply — Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a standalone open source CLI tool that converts video to 3DGS PLY sequences in one command.

**Architecture:** Two-step pipeline — ffmpeg extracts frames, then ml-sharp/SHARP generates per-frame PLY files. Packaged as a pip-installable CLI tool with click.

**Tech Stack:** Python, click (CLI), ffmpeg (system dep), ml-sharp (pip git dep)

---

## Overview

Open source CLI tool that converts a video file into a 3DGS PLY sequence in one command.
Wraps ffmpeg (frame extraction) and Apple's ml-sharp/SHARP model (single-image 3DGS prediction).

## Pipeline

```
Video --[ffmpeg]--> Images --[ml-sharp/SHARP]--> PLY sequence
```

## Usage

```bash
# Install
pip install git+https://github.com/<user>/video2ply

# Run
video2ply input.mp4
video2ply input.mp4 -o output/ -n 100 --device cuda
```

## CLI Interface

| Argument | Default | Description |
|----------|---------|-------------|
| `input` (positional) | required | Video file path |
| `-o, --output` | `./output/` | Output directory |
| `-n, --frames` | all frames | Number of frames to extract |
| `--device` | auto-detect | `cuda` / `cpu` / `mps` |

Auto-detect logic: cuda if available, else mps if on Apple Silicon, else cpu.

## Project Structure

```
video2ply/
├── README.md
├── pyproject.toml
├── LICENSE
├── video2ply/
│   ├── __init__.py
│   ├── cli.py          # click-based CLI entry point
│   ├── extract.py      # video -> images (ffmpeg subprocess)
│   └── predict.py      # images -> ply (calls sharp CLI)
```

## Dependencies

### Python dependencies (in pyproject.toml)
- `click` — CLI framework
- `sharp @ git+https://github.com/apple/ml-sharp` — auto-installed from Apple's repo

### System dependencies
- `ffmpeg` — must be installed separately; CLI checks on startup and shows install instructions

## License Considerations

- ml-sharp code license (Apple Software License): allows use, modify, redistribute with notice retained.
- ml-sharp model license: **Research Purposes only** (non-commercial).
- Our tool does NOT bundle ml-sharp source or model weights — pip pulls from Apple's repo at install time.
- README must include attribution and note the model's research-only license.

## Cross-Platform Support

| Platform | Device options | Notes |
|----------|---------------|-------|
| Windows | cuda, cpu | Requires ffmpeg in PATH |
| macOS | mps, cpu | SHARP originally developed by Apple, runs well on MPS |
| Linux | cuda, cpu | Requires ffmpeg in PATH |

## Output

Given `video2ply input.mp4 -o output/`:

```
output/
├── frames/             # extracted JPEG frames (intermediate)
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── ply/                # final PLY sequence
    ├── frame_0001.ply
    ├── frame_0002.ply
    └── ...
```

## Non-Goals (for v1)

- No GUI
- No PLY -> RAW conversion (exists in parent project 4dgs-plugin)
- No GSD compression (exists in parent project 4dgs-plugin)
- No video rendering from PLY (SHARP's `--render` requires CUDA)

---

# Implementation Plan

## Task 1: Initialize project scaffold

**Files:**
- Create: `video2ply/pyproject.toml`
- Create: `video2ply/LICENSE`
- Create: `video2ply/video2ply/__init__.py`

**Step 1: Create project directory**

```bash
mkdir -p video2ply/video2ply
```

**Step 2: Create pyproject.toml**

Create `video2ply/pyproject.toml`:

```toml
[project]
name = "video2ply"
version = "0.1.0"
description = "Convert video to 3DGS PLY sequences using SHARP."
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "click",
    "sharp @ git+https://github.com/apple/ml-sharp",
]

[project.scripts]
video2ply = "video2ply.cli:main"

[build-system]
requires = ["setuptools", "setuptools-scm"]
build-backend = "setuptools.build_meta"
```

**Step 3: Create LICENSE (MIT)**

Create `video2ply/LICENSE` with MIT license text.

**Step 4: Create __init__.py**

Create `video2ply/video2ply/__init__.py`:

```python
"""video2ply - Convert video to 3DGS PLY sequences."""
```

**Step 5: Commit**

```bash
cd video2ply && git init && git add -A
git commit -m "chore: initialize video2ply project scaffold"
```

---

## Task 2: Implement frame extraction module

**Files:**
- Create: `video2ply/video2ply/extract.py`
  (Adapted from `4dgs-plugin/app/pipeline/video_to_images.py`)

**Step 1: Create extract.py**

Port `video_to_images.py` into `extract.py`. Key changes from original:
- Remove GUI callback parameters, use `print()` for progress
- When `frame_count` is None, extract all frames (no subsampling)
- Keep `check_ffmpeg()`, `get_video_frame_count()`, `extract_frames()`

```python
"""Extract frames from video using ffmpeg."""

import os
import shutil
import subprocess


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def get_video_frame_count(video_path: str) -> int:
    for entries in ["nb_read_packets", "nb_frames"]:
        try:
            args = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
            if entries == "nb_read_packets":
                args.append("-count_packets")
                args += ["-show_entries", "stream=nb_read_packets"]
            else:
                args += ["-show_entries", "stream=nb_frames"]
            args += ["-of", "csv=p=0", video_path]
            result = subprocess.run(args, capture_output=True, text=True, timeout=60)
            return int(result.stdout.strip())
        except Exception:
            continue
    return -1


def extract_frames(video_path, output_folder, frame_count=None):
    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg not found. Install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html\n"
            "  Linux:   sudo apt install ffmpeg"
        )
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_folder, exist_ok=True)
    pattern = os.path.join(output_folder, "frame_%04d.jpg")

    if frame_count is None:
        cmd = ["ffmpeg", "-i", video_path, "-q:v", "2", pattern, "-y"]
    else:
        total = get_video_frame_count(video_path)
        step = max(1, total // frame_count) if total > 0 else 1
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select=not(mod(n\\,{step}))",
            "-fps_mode", "vfr", "-frames:v", str(frame_count),
            "-q:v", "2", pattern, "-y",
        ]

    print(f"Extracting frames to {output_folder}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr}")

    files = sorted(os.path.join(output_folder, f)
                   for f in os.listdir(output_folder)
                   if f.startswith("frame_") and f.endswith(".jpg"))
    print(f"Extracted {len(files)} frames")
    return files
```

**Step 2: Commit**

```bash
git add video2ply/extract.py
git commit -m "feat: add frame extraction module"
```

---

## Task 3: Implement PLY prediction module

**Files:**
- Create: `video2ply/video2ply/predict.py`
  (Adapted from `4dgs-plugin/app/pipeline/images_to_ply.py`)

**Step 1: Create predict.py**

```python
"""Generate 3DGS PLY files from images using ml-sharp/SHARP."""

import os
import re
import shutil
import subprocess


def check_sharp():
    return shutil.which("sharp") is not None


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def generate_ply(images_folder, output_folder, device=None):
    if not check_sharp():
        raise RuntimeError(
            "sharp CLI not found.\n"
            "Install: pip install git+https://github.com/apple/ml-sharp"
        )
    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    if device is None:
        device = detect_device()

    os.makedirs(output_folder, exist_ok=True)
    cmd = ["sharp", "predict", "-i", images_folder, "-o", output_folder,
           "--no-render", "--device", device]

    print(f"Running SHARP prediction on {device}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    total, done = 0, 0
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        m = re.search(r"Processing (\d+) valid image files", line)
        if m:
            total = int(m.group(1))
        if "Processing " in line and any(e in line.lower() for e in (".jpg", ".png", ".jpeg", ".heic")):
            done += 1
            if total > 0:
                print(f"  [{done}/{total}]")

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ml-sharp failed (exit {proc.returncode})")

    files = sorted(os.path.join(output_folder, f)
                   for f in os.listdir(output_folder) if f.lower().endswith(".ply"))
    print(f"Generated {len(files)} PLY files")
    return files
```

**Step 2: Commit**

```bash
git add video2ply/predict.py
git commit -m "feat: add PLY prediction module"
```

---

## Task 4: Implement CLI entry point

**Files:**
- Create: `video2ply/video2ply/cli.py`

**Step 1: Create cli.py**

```python
"""CLI entry point for video2ply."""

import time
import click
from .extract import extract_frames
from .predict import generate_ply, detect_device


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", default="./output", help="Output directory.")
@click.option("-n", "--frames", type=int, default=None, help="Number of frames (default: all).")
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps", "auto"]), default="auto",
              help="Compute device.")
def main(input, output, frames, device):
    """Convert a video to a 3DGS PLY sequence."""
    if device == "auto":
        device = detect_device()

    t0 = time.time()
    print("\n=== Step 1/2: Extracting frames ===")
    extract_frames(input, f"{output}/frames", frames)
    print(f"\n=== Step 2/2: Generating PLY ({device}) ===")
    ply_files = generate_ply(f"{output}/frames", f"{output}/ply", device)

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"\nDone! {len(ply_files)} PLY files in {m}m {s}s")
    print(f"Output: {output}/ply")
```

**Step 2: Commit**

```bash
git add video2ply/cli.py
git commit -m "feat: add CLI entry point"
```

---

## Task 5: Write README

**Files:**
- Create: `video2ply/README.md`

Content: one-line description, install, usage, output structure, attribution (ml-sharp by Apple, research-only model license), MIT license for this tool.

**Commit:**

```bash
git add README.md && git commit -m "docs: add README"
```

---

## Task 6: Smoke test

```bash
pip install -e .
video2ply --help
video2ply test_video.mp4 -o test_output/ -n 3
```

Verify: `test_output/frames/` has 3 JPEGs, `test_output/ply/` has 3 PLYs.

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Project scaffold | pyproject.toml, LICENSE, __init__.py |
| 2 | Frame extraction | extract.py |
| 3 | PLY prediction | predict.py |
| 4 | CLI entry point | cli.py |
| 5 | README | README.md |
| 6 | Smoke test | — |
