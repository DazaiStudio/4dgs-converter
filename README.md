<p align="center">
  <img src="docs/logo.png" alt="4DGS Converter" width="200">
  <h1 align="center">4DGS Converter</h1>
  <p align="center">Convert videos or 3DGS (.ply) sequences into 4DGS (.gsd) files for real-time Gaussian Splatting playback.</p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
    <img src="https://img.shields.io/badge/Platform-Windows%20|%20macOS-lightgrey" alt="Platform">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
    <a href="https://github.com/DazaiStudio/4dgs-converter/releases"><img src="https://img.shields.io/github/v/release/DazaiStudio/4dgs-converter?label=Release" alt="Release"></a>
  </p>
</p>

![4DGS Converter GUI](docs/screenshot.png)

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Download (Windows)](#option-a-download-windows)
  - [Run from Source](#option-b-run-from-source-windows--macos)
- [How It Works](#how-it-works)
- [How to Use](#how-to-use)
  - [GUI](#gui)
  - [CLI](#cli)
- [Reference](#reference)
- [License](#license)

---

## Features

- **GUI** : One-click conversion with visual controls for FPS, frame range, and mode selection
- **CLI** : Command-line interface for scripting and AI agent integration (Claude, GPT, etc.)
- **Format Support** : Works with any standard 3DGS `.ply` format (SHARP, PostShot, Nerfstudio, etc.)

---

## Getting Started

### Option A: Download (Windows)

Download [**4DGS-Converter.exe**](https://github.com/DazaiStudio/4dgs-converter/releases/latest) and run. No installation required.

### Option B: Run from Source (Windows / macOS)

```bash
git clone https://github.com/DazaiStudio/4dgs-converter.git
cd 4dgs-converter
pip install -r requirements.txt
pip install PySide6 lz4
python -m app.converter
```

---

## How It Works

```
Video ──► Images (ffmpeg) ──► 3DGS .ply (SHARP) ──► 4DGS .gsd
                                                      ▲
              3DGS Sequence (.ply) folder ────────────┘
```

---

## How to Use

### GUI

1. Select mode: **Video to 4DGS** or **3DGS Sequence to 4DGS**
2. Browse for input (video file or 3DGS sequence (.ply) folder)
3. Adjust FPS and frame range if needed
4. Click **Generate**

### CLI

```bash
# Video to GSD (full pipeline)
python -m app.converter --cli -i video.mp4 -o output.gsd

# 3DGS sequence (.ply) folder to GSD
python -m app.converter --cli -i /path/to/ply_folder -o output.gsd --fps 24

# With options
python -m app.converter --cli -i video.mp4 --start 0 --end 100 --keep-ply --keep-images
```

---

## Reference

<details>
<summary><strong>4DGS / GSD Format</strong></summary>

**4D Gaussian Splatting** extends 3DGS with a time dimension, enabling real-time playback of dynamic 3D scenes.

The `.gsd` (Gaussian Stream Data) format packs an entire PLY sequence into a single compressed file, with O(1) random access to any frame.

**Example:** A 6-second video at 24 FPS produces 144 PLY files totaling **8.9 GB**. The resulting `.gsd` file is **4.3 GB** (~48% compression via Byte-Shuffle + LZ4).

</details>

<details>
<summary><strong>CLI Flags</strong></summary>

| Flag | Description |
|------|-------------|
| `--cli` | Run in CLI mode (no GUI) |
| `-i, --input` | Input video file or 3DGS sequence (.ply) folder |
| `-o, --output` | Output .gsd path (auto-derived if omitted) |
| `--mode` | `auto`, `video`, or `ply` (default: auto-detect) |
| `--fps` | Target FPS (default: 30, auto-detected for video) |
| `--start` | Start frame, 0-based (default: 0) |
| `--end` | End frame, 0-based (default: last) |
| `--keep-images` | Keep extracted images (video mode) |
| `--keep-ply` | Keep PLY files (video mode) |
| `--skip-gsd` | Stop after PLY generation (video mode) |

</details>

<details>
<summary><strong>Dependencies</strong></summary>

| Tool | Required For | Install |
|------|-------------|---------|
| **ffmpeg** | Video frame extraction | `winget install ffmpeg` (Win) / `brew install ffmpeg` (Mac) |
| **SHARP** (ml-sharp) | Video → 3DGS | [apple/ml-sharp](https://github.com/apple/ml-sharp) |
| **lz4** | GSD compression | `pip install lz4` |

ffmpeg and SHARP are only needed for **Video → 4DGS** mode. For **3DGS Sequence → 4DGS**, only lz4 is required.

</details>

<details>
<summary><strong>Unreal Engine Plugin</strong></summary>

**Splat Renderer** : UE 5.6 plugin for real-time `.gsd` playback. *Coming soon.*

</details>

---

## License

MIT
