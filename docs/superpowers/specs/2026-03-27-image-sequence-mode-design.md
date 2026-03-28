# Image Sequence Mode + SHARP Install Improvement

**Date:** 2026-03-27
**Branch:** feature/gsd-v2-sharp-vq (work on master-compatible code, GSD v2 not involved)

## Summary

Two changes to the 4DGS Converter:

1. **New "Image Sequence" input source** in the Video to 4DGS tab — skips ffmpeg, goes straight to SHARP → GSD.
2. **One-click SHARP Install button** — replaces the current text-only dialog with an actual `pip install` action.

## 1. Image Sequence Input Source

### 1.1 GUI

Add a radio button row inside the "Video to 4DGS" tab:

```
[ Video to 4DGS ]  [ 3DGS Sequence to 4DGS ]

Input Source:  ○ Video File  ○ Image Sequence

Input: [________________________] [Browse]
```

**Behavior by input source:**

| Aspect | Video File | Image Sequence |
|---|---|---|
| Browse | File picker (mp4/mov/avi/mkv) | Folder picker |
| Dependencies | ffmpeg + sharp | sharp only |
| Pipeline steps | 3 (ffmpeg → SHARP → GSD) | 2 (SHARP → GSD) |
| FPS detection | Auto (ffprobe) | Manual — user sets value |
| FPS spin | Read-only (auto-detected) | Editable |
| Keep images checkbox | Visible | Hidden (user owns the images) |
| Keep PLY checkbox | Visible | Visible |
| Skip GSD checkbox | Visible | Visible |
| Frame range source | Video total frames | Image count in folder |
| Info label | "480 frames, 30.0 fps, 16.0s" | "480 images" |

**Dependency gating:**

- Video File selected + ffmpeg ✗ or sharp ✗ → Generate disabled, tooltip "Requires ffmpeg and sharp"
- Image Sequence selected + sharp ✗ → Generate disabled, tooltip "Requires sharp"
- Image Sequence selected + ffmpeg ✗ + sharp ✓ → Generate **enabled** (ffmpeg not needed)

**Image detection:** Scan folder for files with extensions `.jpg`, `.jpeg`, `.png`, `.heic` (case-insensitive). Sort by filename = frame order. Users are expected to provide correctly ordered/named images.

### 1.2 CLI

New `--mode images` value:

```bash
# New mode
python -m app.converter --cli -i /path/to/images -o output.gsd --mode images

# Existing modes unchanged
python -m app.converter --cli -i video.mp4 -o output.gsd --mode video
python -m app.converter --cli -i /path/to/ply -o output.gsd --mode ply
```

**Auto-detect update** (`--mode auto`, the default):

Current logic:
- Directory → `ply`
- File → `video`

New logic:
- Directory containing `.ply` files → `ply`
- Directory containing image files (`.jpg/.jpeg/.png/.heic`) and no `.ply` files → `images`
- File → `video`

**Output auto-derive:** Same as video mode — `<parent>/<name>/<name>.gsd` based on input folder name.

### 1.3 Worker

New `mode="images"` support in `PipelineWorker`:

**`__init__`:** Accept `mode="images"` in addition to `"video"` and `"ply"`.

**`_derive_paths`:** For images mode:
- `self.images_folder = self.input_path` (user's folder, not a temp folder)
- `self.ply_folder = os.path.join(os.path.dirname(self.output_path), "ply")`

**New `_run_images_pipeline()`:**

```
Step 1/2: Generating PLY (SHARP)...
    → calls self._generate_ply() (reuses existing method)
Step 2/2: Converting to GSD...
    → calls self._convert_to_gsd() (reuses existing method)
```

**Cleanup:** For images mode:
- Never delete `images_folder` (it's the user's input, not a temp folder)
- Delete `ply_folder` if `keep_ply=False`

**`run()` dispatch:**

```python
if self.mode == "video":
    self._run_video_pipeline()
elif self.mode == "images":
    self._run_images_pipeline()
else:
    self._run_ply_pipeline()
```

### 1.4 CLI entry point (`__main__.py`)

New `_run_images_cli()` function:

```
Step 1/2: Generating PLY (SHARP)
Step 2/2: Converting to GSD
```

Same structure as `_run_video_cli()` but skips frame extraction step.

## 2. SHARP Install Button Improvement

### Current behavior

Clicking "Install" next to sharp ✗ shows a `QMessageBox.information` dialog with text instructions:

```
Install ml-sharp from source:
  git clone https://github.com/apple/ml-sharp
  cd ml-sharp
  pip install -e .
After installation, restart 4DGS Converter.
```

User must manually run these commands.

### New behavior

Clicking "Install" opens a confirmation dialog, then runs the installation in a new console window:

```python
# Pseudocode
if os.path.isdir("ml-sharp") and os.path.isfile("ml-sharp/pyproject.toml"):
    # ml-sharp source already present (cloned repo)
    cmd = ["pip", "install", "-e", "ml-sharp/"]
else:
    # Need to clone first
    cmd = ["git clone https://github.com/apple/ml-sharp ml-sharp && pip install -e ml-sharp/"]
```

Runs in a new console window (same as lz4/ffmpeg Install buttons) via `subprocess.Popen` with `CREATE_NEW_CONSOLE`.

**Confirmation dialog text:**

```
Install SHARP (ml-sharp)?

This will run:
  pip install -e ml-sharp/

ml-sharp has large dependencies (PyTorch, etc.)
and may take several minutes to install.

After installation, restart 4DGS Converter.
```

### Sharp ✗ tooltip

When sharp is not found, the env status label tooltip says:

> "sharp not found in current Python environment. Click Install to install here, or activate the environment where sharp is installed."

## 3. Files Changed

| File | Change |
|---|---|
| `app/converter/main_window.py` | Add Input Source radio buttons, update mode/dep logic |
| `app/converter/worker.py` | Add `mode="images"`, `_run_images_pipeline()` |
| `app/converter/__main__.py` | Add `--mode images`, `_run_images_cli()`, update auto-detect |
| `app/converter/main_window.py` | Update `_install_dep("sharp")` to run pip directly |

## 4. Not Changed

- `app/converter/env_check.py` — no changes
- `app/pipeline/ply_to_gsd.py` — no changes
- `app/pipeline/images_to_ply.py` — no changes
- `app/pipeline/video_to_images.py` — no changes
- 3DGS Sequence to 4DGS tab — no changes
- GSD v2 — not involved
- README — deferred to a separate update
