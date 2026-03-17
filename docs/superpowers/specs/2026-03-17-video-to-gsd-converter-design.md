# Video-to-GSD Converter — Design Spec

**Date:** 2026-03-17

## Overview

A standalone PySide6 GUI app that converts video or PLY folders into GSD files for 4DGS streaming. Two input modes, one-click generation, progress tracking, and environment validation.

## Input Modes

### Mode 1: From Video
- User selects a video file
- App auto-detects FPS via ffprobe
- Pipeline: video → images (ffmpeg) → PLY (sharp predict) → GSD (ply_to_gsd)
- Requires: ffmpeg + sharp

### Mode 2: From PLY Folder
- User selects a folder containing .ply files
- User manually enters FPS
- Pipeline: PLY → GSD (ply_to_gsd)
- Requires: nothing extra

## UI Layout

```
┌─────────────────────────────────────────────┐
│  Video to GSD Converter                     │
├─────────────────────────────────────────────┤
│                                             │
│  Mode:  [From Video ▼]                      │
│                                             │
│  Input:  [path/to/video.mp4      ] [Browse] │
│  FPS:    [60] (auto-detected)               │
│  Output: [path/to/output.gsd     ] [Browse] │
│                                             │
│  ☑ Keep image sequence                      │
│  ☑ Keep PLY sequence                        │
│  ☐ Skip GSD (PLY only)                      │
│                                             │
│  [ ● Generate ]                             │
│                                             │
│  Step 2/3: Generating PLY...                │
│  ████████████░░░░░░░  62%  ETA: 3m 24s      │
│                                             │
│  ▶ Log                                      │
│  ┌────────────────────────────────────────┐ │
│  │ [12:03:01] Extracting frames...        │ │
│  │ [12:03:05] 480 frames extracted        │ │
│  │ [12:03:06] Running SHARP predict...    │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  Environment:  ffmpeg ✓   sharp ✓           │
└─────────────────────────────────────────────┘
```

### Mode switching behavior
- "From Video": shows video file picker, FPS auto-detected and read-only, checkboxes for intermediate files visible
- "From PLY Folder": shows folder picker, FPS editable, "Keep image sequence" checkbox hidden, "Keep PLY sequence" checkbox hidden, "Skip GSD" checkbox hidden

### Environment status bar
- Bottom bar shows ffmpeg / sharp availability
- Green checkmark if found, red X if missing
- "From Video" mode disabled if ffmpeg or sharp missing
- Tooltip on red X explains how to install

## Output Path Convention

When user selects input, auto-derive output path:
- Video `D:\data\my_video.mp4` → `D:\data\my_video\my_video.gsd`
- Creates intermediate folders: `D:\data\my_video\images\`, `D:\data\my_video\ply\`
- PLY folder `D:\data\my_ply\` → `D:\data\my_ply.gsd` (sibling of folder)

User can override output path via Browse button.

## Pipeline Execution

### Step 1: Extract frames (From Video only)
- Call `video_to_images.extract_frames()`
- Progress: frame count from ffprobe, track extracted frames

### Step 2: Generate PLY (From Video only)
- Call `images_to_ply.generate_ply()` as subprocess
- Progress: parse SHARP stdout for per-frame progress

### Step 3: Convert to GSD
- Call `ply_to_gsd.convert_ply_to_gsd()`
- Progress: frame_progress_callback from existing API

### Cleanup
- After GSD complete, check "Keep" checkboxes
- Delete unchecked intermediate folders
- If "Skip GSD" checked, stop after PLY step

### Cancellation
- Generate button becomes "Stop" during execution
- Kills running subprocess (ffmpeg/sharp) on stop
- Cleans up partial output

## Settings (hardcoded defaults)

| Setting | Value |
|---------|-------|
| SH degree | 0 |
| Position precision | Full (32-bit) |
| Rotation precision | Half (16-bit) |
| Scale/Opacity precision | Half (16-bit) |
| SH precision | Half (16-bit) |
| Prune | None |

## Progress & ETA

- Per-step progress bar with step label ("Step 1/3: Extracting frames...")
- ETA based on elapsed time per frame × remaining frames
- Overall percentage across all steps (weighted by estimated time)

## Theme

- Follow system theme (dark/light) via `QStyleFactory` or `qt-material`
- No custom theming — native OS look

## Tech Stack

- Python 3.10+
- PySide6 (UI)
- Existing pipeline modules: `video_to_images`, `images_to_ply`, `ply_to_gsd`
- Threading: QThread for pipeline execution, signals for progress updates

## File Structure

```
app/
  converter/
    __init__.py
    main_window.py    # PySide6 main window
    worker.py         # QThread pipeline worker
    env_check.py      # ffmpeg/sharp detection
  pipeline/
    video_to_images.py  # (existing)
    images_to_ply.py    # (existing)
    ply_to_gsd.py       # (existing)
```

## Entry Point

```bash
python -m app.converter
```
