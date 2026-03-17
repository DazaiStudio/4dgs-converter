# Video-to-GSD Converter — Design Spec

**Date:** 2026-03-17

## Overview

A standalone PySide6 GUI app that converts video or PLY folders into GSD files for 4DGS streaming. Two input modes, one-click generation, progress tracking, and environment validation.

## Input Modes

### Mode 1: From Video
- User selects a video file
- App auto-detects FPS via ffprobe (`get_video_fps()` — new helper in `video_to_images.py`)
- Extracts ALL frames at source FPS (pass `frame_count=total_frames` to `extract_frames()`)
- Pipeline: video → images (ffmpeg) → PLY (sharp predict) → GSD (ply_to_gsd)
- Requires: ffmpeg + sharp
- SHARP runs with `device="cuda"` (auto-fallback to `"cpu"` if CUDA unavailable)

### Mode 2: From PLY Folder
- User selects a folder containing .ply files
- User manually enters FPS
- Pipeline: PLY → GSD (ply_to_gsd)
- Requires: lz4 only

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
│  Environment:  ffmpeg ✓   sharp ✓   lz4 ✓   │
└─────────────────────────────────────────────┘
```

### Mode switching behavior
- "From Video": shows video file picker, FPS auto-detected and read-only, checkboxes for intermediate files visible
- "From PLY Folder": shows folder picker, FPS editable, all checkboxes hidden (only PLY→GSD)

### Checkbox logic
- "Skip GSD" checked → forces "Keep PLY" checked and disabled (PLY is final output)
- "Skip GSD" checked → hides progress for Step 3

### Environment status bar
- Bottom bar shows ffmpeg / sharp / lz4 availability
- Green checkmark if found, red X if missing
- "From Video" mode disabled if ffmpeg or sharp missing
- Tooltip on red X explains how to install

## Output Path Convention

When user selects input, auto-derive output path:
- Video `D:\data\my_video.mp4` → `D:\data\my_video\my_video.gsd`
- Creates intermediate folders: `D:\data\my_video\images\`, `D:\data\my_video\ply\`
- PLY folder `D:\data\my_ply\` → `D:\data\my_ply.gsd` (sibling of folder)
- If output file already exists, prompt user to confirm overwrite

`sequence_name` auto-derived from input filename (e.g. `my_video.mp4` → `"my_video"`).

User can override output path via Browse button.

## Pipeline Execution

### Step 1: Extract frames (From Video only)
- New `get_video_fps()` function in `video_to_images.py` using ffprobe
- Call `extract_frames()` with `frame_count` = total video frames (extract all)
- Progress: track extracted frames vs total

### Step 2: Generate PLY (From Video only)
- Call `images_to_ply.generate_ply()` as subprocess
- Device: `cuda` default, fallback to `cpu`
- Progress: parse SHARP stdout for per-frame progress

### Step 3: Convert to GSD
- Call `ply_to_gsd.convert_ply_to_gsd()` with auto-derived `sequence_name` and detected FPS
- Progress: frame_progress_callback from existing API

### Resume logic
- Step 1 skipped if images folder already has the expected frame count
- Step 2 skipped if PLY folder already has matching PLY count
- Step 3 always runs (overwrites existing GSD)

### Cleanup
- After pipeline complete, check "Keep" checkboxes
- Delete unchecked intermediate folders
- If "Skip GSD" checked, stop after PLY step

### Cancellation
- Generate button becomes "Stop" during execution
- Kills running subprocess (ffmpeg/sharp) on stop
- Intermediate folders kept intact (useful for resume)
- Partially-written GSD file deleted

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
- Step weight for overall progress: ffmpeg 1%, SHARP 90%, GSD 9% (SHARP dominates at ~6-8s/frame)

## Theme

- Follow system theme (dark/light) via PySide6 default style
- No custom theming — native OS look

## Tech Stack

- Python 3.10+
- PySide6 (UI)
- Existing pipeline modules: `video_to_images`, `images_to_ply`, `ply_to_gsd`
- Threading: QThread for pipeline execution, Qt signals for progress updates
- Note: `ply_to_gsd` uses ProcessPoolExecutor internally; its `frame_progress_callback` is called from the main process thread (the `as_completed` loop), safe to emit Qt signals from

## File Structure

```
app/
  converter/
    __init__.py
    main_window.py    # PySide6 main window
    worker.py         # QThread pipeline worker
    env_check.py      # ffmpeg/sharp/lz4 detection
  pipeline/
    video_to_images.py  # (existing, add get_video_fps())
    images_to_ply.py    # (existing)
    ply_to_gsd.py       # (existing)
```

## Entry Point

```bash
python -m app.converter
```
