# Image Sequence Mode + SHARP Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Image Sequence" input source to the Video to 4DGS tab (skips ffmpeg, goes SHARP → GSD), and make the SHARP Install button actually install instead of showing text instructions.

**Architecture:** Three files are modified: `worker.py` gets a new `_run_images_pipeline()` that reuses `_generate_ply()` and `_convert_to_gsd()`; `__main__.py` gets `--mode images` and updated auto-detect; `main_window.py` gets Input Source radio buttons and a working SHARP Install button.

**Tech Stack:** Python 3.10+, PySide6, subprocess

**Spec:** `docs/superpowers/specs/2026-03-27-image-sequence-mode-design.md`

---

### Task 1: Worker — Add `mode="images"` pipeline

**Files:**
- Modify: `app/converter/worker.py:82-99` (`_derive_paths` and `run` dispatch)
- Modify: `app/converter/worker.py:255-264` (`_cleanup`)

This task adds the images pipeline to the worker. No tests exist in this project — the worker depends on PySide6 QThread and subprocess calls to SHARP, so we verify manually via CLI in a later task.

- [ ] **Step 1: Update `_derive_paths` to handle images mode**

In `app/converter/worker.py`, replace the `_derive_paths` method:

```python
def _derive_paths(self):
    """Derive intermediate folder paths from input/output."""
    if self.mode == "video":
        parent = os.path.dirname(self.output_path)
        self.images_folder = os.path.join(parent, "images")
        self.ply_folder = os.path.join(parent, "ply")
    elif self.mode == "images":
        self.images_folder = self.input_path
        self.ply_folder = os.path.join(os.path.dirname(self.output_path), "ply")
    else:
        self.ply_folder = self.input_path
        self.images_folder = None
```

- [ ] **Step 2: Add `_run_images_pipeline` method**

Add this new method after `_run_video_pipeline` (after line 141):

```python
def _run_images_pipeline(self):
    total_steps = 1 if self.skip_gsd else 2
    step = 0

    # Step 1: Generate PLY
    step += 1
    self.progress.emit(step, total_steps, "Generating PLY (SHARP)...")
    self._generate_ply()
    self._check_stop()

    # Step 2: Convert to GSD
    if not self.skip_gsd:
        step += 1
        self.progress.emit(step, total_steps, "Converting to GSD...")
        self._convert_to_gsd()

    # Cleanup
    self._cleanup()
```

- [ ] **Step 3: Update `run()` dispatch**

Replace the dispatch in `run()`:

```python
if self.mode == "video":
    self._run_video_pipeline()
elif self.mode == "images":
    self._run_images_pipeline()
else:
    self._run_ply_pipeline()
```

- [ ] **Step 4: Update `_cleanup` to protect user's images**

Replace the `_cleanup` method:

```python
def _cleanup(self):
    """Delete intermediate folders based on keep flags."""
    if self.mode == "video":
        if not self.keep_images and self.images_folder and os.path.isdir(self.images_folder):
            self._log(f"Cleaning up images: {self.images_folder}")
            shutil.rmtree(self.images_folder, ignore_errors=True)
    # images mode: never delete images_folder (user's input)
    if self.mode in ("video", "images"):
        if not self.keep_ply and self.ply_folder and os.path.isdir(self.ply_folder):
            self._log(f"Cleaning up PLY: {self.ply_folder}")
            shutil.rmtree(self.ply_folder, ignore_errors=True)
```

- [ ] **Step 5: Verify syntax**

Run: `python -c "from app.converter.worker import PipelineWorker; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add app/converter/worker.py
git commit -m "feat: add images mode to PipelineWorker

Support mode='images' that skips frame extraction and goes
directly from user's image folder to SHARP → GSD.
Never deletes user's input images during cleanup."
```

---

### Task 2: CLI — Add `--mode images` and update auto-detect

**Files:**
- Modify: `app/converter/__main__.py:88-93` (auto-detect logic)
- Modify: `app/converter/__main__.py:113-116` (dispatch)
- Add function: `_run_images_cli` in `app/converter/__main__.py`

- [ ] **Step 1: Add `_run_images_cli` function**

Add this function after `_run_video_cli` (after line 184):

```python
def _run_images_cli(input_path, output_path, fps, start_frame, end_frame, args):
    import os

    parent = os.path.dirname(output_path)
    ply_folder = os.path.join(parent, "ply")

    # Count images
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")
    image_count = len([
        f for f in os.listdir(input_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    print(f"  Found {image_count} images")

    # Step 1: Generate PLY
    print("Step 1/2: Generating PLY (SHARP)...")
    from app.pipeline.images_to_ply import generate_ply

    os.makedirs(ply_folder, exist_ok=True)
    ply_count = len([f for f in os.listdir(ply_folder) if f.endswith(".ply")])

    if ply_count >= image_count and ply_count > 0:
        print(f"  PLY files already exist ({ply_count} files), skipping.")
    else:
        generate_ply(input_path, ply_folder, progress_callback=lambda msg: print(f"  {msg}"))

    # Step 2: Convert to GSD
    if not args.skip_gsd:
        print("Step 2/2: Converting to GSD...")
        _convert_gsd(ply_folder, output_path, fps, start_frame, end_frame)

    # Cleanup PLY if not keeping
    if not args.keep_ply and os.path.isdir(ply_folder):
        import shutil
        print(f"Cleaning up PLY: {ply_folder}")
        shutil.rmtree(ply_folder, ignore_errors=True)

    print("Done!")
```

- [ ] **Step 2: Update auto-detect logic**

Replace the auto-detect block in `main_cli` (lines 88-93):

```python
    # Auto-detect mode
    if mode == "auto":
        if os.path.isdir(input_path):
            # Check folder contents to distinguish ply vs images
            contents = os.listdir(input_path)
            has_ply = any(f.lower().endswith(".ply") for f in contents)
            IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")
            has_images = any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTS for f in contents
            )
            if has_ply:
                mode = "ply"
            elif has_images:
                mode = "images"
            else:
                mode = "ply"  # fallback
        else:
            mode = "video"
```

- [ ] **Step 3: Add `"images"` to `--mode` choices**

Replace the `--mode` argument (line 228):

```python
        parser.add_argument("--mode", choices=["auto", "video", "images", "ply"], default="auto",
                            help="Conversion mode (default: auto-detect from input)")
```

- [ ] **Step 4: Update dispatch in `main_cli`**

Replace the dispatch at the end of `main_cli` (lines 113-116):

```python
    if mode == "video":
        _run_video_cli(input_path, output_path, fps, start_frame, end_frame, args)
    elif mode == "images":
        _run_images_cli(input_path, output_path, fps, start_frame, end_frame, args)
    else:
        _run_ply_cli(input_path, output_path, fps, start_frame, end_frame)
```

- [ ] **Step 5: Update output auto-derive for images mode**

Replace the output auto-derive block (lines 94-103):

```python
    # Auto-derive output
    if not output_path:
        if mode == "video":
            name = os.path.splitext(os.path.basename(input_path))[0]
            parent = os.path.dirname(input_path)
            output_path = os.path.join(parent, name, f"{name}.gsd")
        elif mode == "images":
            folder_name = os.path.basename(input_path.rstrip("/\\"))
            parent = os.path.dirname(input_path.rstrip("/\\"))
            output_path = os.path.join(parent, folder_name, f"{folder_name}.gsd")
        else:
            folder_name = os.path.basename(input_path.rstrip("/\\"))
            parent = os.path.dirname(input_path.rstrip("/\\"))
            output_path = os.path.join(parent, f"{folder_name}.gsd")
```

- [ ] **Step 6: Verify CLI help**

Run: `python -m app.converter --cli --help`
Expected: `--mode` shows `{auto,video,images,ply}`

- [ ] **Step 7: Commit**

```bash
git add app/converter/__main__.py
git commit -m "feat: add --mode images to CLI with auto-detect

Auto-detect: folder with .ply → ply mode, folder with images → images mode.
Images mode runs SHARP → GSD, skipping ffmpeg frame extraction."
```

---

### Task 3: GUI — Add Input Source radio buttons

**Files:**
- Modify: `app/converter/main_window.py:61-102` (`_build_ui` — add radio buttons after mode tabs)
- Modify: `app/converter/main_window.py:298-342` (`_set_mode`, `_on_mode_changed`)
- Modify: `app/converter/main_window.py:411-439` (`_browse_input`)
- Modify: `app/converter/main_window.py:461-501` (`_update_info`)
- Modify: `app/converter/main_window.py:589-638` (`_on_generate`)

- [ ] **Step 1: Add Input Source radio buttons in `_build_ui`**

After the mode tabs section (after line 90, `layout.addLayout(mode_row)`), add:

```python
        # -- Input Source selector (Video tab only)
        self.source_row = QHBoxLayout()
        self.source_row.addWidget(QLabel("Input Source:"))
        self._source_video = QPushButton("Video File")
        self._source_video.setCheckable(True)
        self._source_video.setChecked(True)
        self._source_images = QPushButton("Image Sequence")
        self._source_images.setCheckable(True)
        self._source_images.setChecked(False)

        source_style_active = (
            "QPushButton { background: #3a3a3a; border: 1px solid #555; "
            "padding: 4px 12px; font-weight: bold; border-radius: 3px; }"
        )
        source_style_inactive = (
            "QPushButton { background: #2a2a2a; border: 1px solid #444; "
            "padding: 4px 12px; color: #888; border-radius: 3px; }"
            "QPushButton:hover { background: #333; color: #ccc; }"
        )
        self._source_style_active = source_style_active
        self._source_style_inactive = source_style_inactive
        self._source_video.setStyleSheet(source_style_active)
        self._source_images.setStyleSheet(source_style_inactive)

        self._source_video.clicked.connect(lambda: self._set_input_source("video"))
        self._source_images.clicked.connect(lambda: self._set_input_source("images"))

        self.source_row.addWidget(self._source_video)
        self.source_row.addWidget(self._source_images)
        self.source_row.addStretch()
        layout.addLayout(self.source_row)
```

- [ ] **Step 2: Add `_input_source` state and `_set_input_source` method**

Add after `_set_mode` method:

```python
    def _set_input_source(self, source: str):
        """Switch input source within Video to 4DGS tab."""
        self._input_source = source
        is_video_src = source == "video"
        self._source_video.setChecked(is_video_src)
        self._source_images.setChecked(not is_video_src)
        self._source_video.setStyleSheet(
            self._source_style_active if is_video_src else self._source_style_inactive
        )
        self._source_images.setStyleSheet(
            self._source_style_inactive if is_video_src else self._source_style_active
        )
        # Clear input when switching
        self.input_edit.clear()
        self.output_edit.clear()
        self.info_label.setText("")
        self._on_mode_changed()
```

Initialize `self._input_source = "video"` in `__init__` (after `self._current_mode = 0`).

- [ ] **Step 3: Update `_on_mode_changed` for input source awareness**

Replace `_on_mode_changed`:

```python
    def _on_mode_changed(self):
        is_video = getattr(self, '_current_mode', 0) == 0
        input_source = getattr(self, '_input_source', 'video')
        is_video_source = is_video and input_source == "video"
        is_images_source = is_video and input_source == "images"

        # Show/hide input source selector (only in Video tab)
        for i in range(self.source_row.count()):
            w = self.source_row.itemAt(i).widget()
            if w:
                w.setVisible(is_video)

        # Input placeholder
        if is_video_source:
            self.input_edit.setPlaceholderText("Select a video file...")
        elif is_images_source:
            self.input_edit.setPlaceholderText("Select image sequence folder...")
        else:
            self.input_edit.setPlaceholderText("Select PLY folder...")

        self.info_label.setText("")

        # Source FPS: only visible in PLY mode
        self.source_fps_label.setVisible(not is_video)
        self.source_fps_spin.setVisible(not is_video)
        self.source_fps_note.setVisible(not is_video)

        # FPS: read-only only for video file source, editable otherwise
        self.fps_spin.setReadOnly(is_video_source)
        if is_video_source:
            self.fps_note.setText("(auto-detected)")
        elif is_images_source:
            self.fps_note.setText("(set manually)")
        else:
            self._update_fps_note()
        self.fps_note.setVisible(True)

        # Checkboxes: Keep images only for video file source
        self.chk_keep_images.setVisible(is_video_source)
        self.chk_keep_ply.setVisible(is_video or is_images_source)
        self.chk_skip_gsd.setVisible(is_video)

        # Dependency gating
        has_ffmpeg = self._env.get("ffmpeg", False)
        has_sharp = self._env.get("sharp", False)
        if is_video_source:
            can_generate = has_ffmpeg and has_sharp
            tooltip = "Requires ffmpeg and sharp" if not can_generate else ""
        elif is_images_source:
            can_generate = has_sharp
            tooltip = "Requires sharp" if not can_generate else ""
        else:
            can_generate = True
            tooltip = ""
        self.generate_btn.setEnabled(can_generate)
        self.generate_btn.setToolTip(tooltip)
```

- [ ] **Step 4: Update `_browse_input` for images source**

Replace `_browse_input`:

```python
    def _browse_input(self):
        is_video = getattr(self, '_current_mode', 0) == 0
        input_source = getattr(self, '_input_source', 'video')

        if is_video and input_source == "video":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video",
                "", "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")

        if not path:
            return

        self.input_edit.setText(path)
        self._auto_derive_output(path)
        self._update_info(path)

        # Auto-detect FPS for video file source only
        if is_video and input_source == "video":
            from app.pipeline.video_to_images import get_video_fps

            fps = get_video_fps(path)
            if fps > 0:
                self.fps_spin.setValue(round(fps))
                self.fps_note.setText(f"(detected: {fps:.2f})")
            else:
                self.fps_note.setText("(detection failed — set manually)")
                self.fps_spin.setReadOnly(False)
```

- [ ] **Step 5: Update `_auto_derive_output` for images source**

Replace `_auto_derive_output`:

```python
    def _auto_derive_output(self, input_path: str):
        is_video = getattr(self, '_current_mode', 0) == 0
        input_source = getattr(self, '_input_source', 'video')

        if is_video and input_source == "video":
            name = os.path.splitext(os.path.basename(input_path))[0]
            parent = os.path.dirname(input_path)
            out_dir = os.path.join(parent, name)
            self.output_edit.setText(os.path.join(out_dir, f"{name}.gsd"))
        elif is_video and input_source == "images":
            folder_name = os.path.basename(input_path.rstrip("/\\"))
            parent = os.path.dirname(input_path.rstrip("/\\"))
            out_dir = os.path.join(parent, folder_name)
            self.output_edit.setText(os.path.join(out_dir, f"{folder_name}.gsd"))
        else:
            folder_name = os.path.basename(input_path.rstrip("/\\"))
            parent = os.path.dirname(input_path.rstrip("/\\"))
            self.output_edit.setText(os.path.join(parent, f"{folder_name}.gsd"))
```

- [ ] **Step 6: Update `_update_info` for images source**

Replace `_update_info`:

```python
    def _update_info(self, input_path: str):
        """Show frame count and duration info after selecting input."""
        is_video = getattr(self, '_current_mode', 0) == 0
        input_source = getattr(self, '_input_source', 'video')

        if is_video and input_source == "video":
            from app.pipeline.video_to_images import get_video_frame_count, get_video_fps

            frames = get_video_frame_count(input_path)
            fps = get_video_fps(input_path)
            self._total_frames = max(frames, 0)
            if frames > 0 and fps > 0:
                duration = frames / fps
                self.info_label.setText(
                    f"{frames} frames, {fps:.1f} fps, {duration:.1f}s"
                )
            elif frames > 0:
                self.info_label.setText(f"{frames} frames")
            else:
                self.info_label.setText("")
        elif is_video and input_source == "images":
            if os.path.isdir(input_path):
                IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")
                img_count = len([
                    f for f in os.listdir(input_path)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                ])
                self._total_frames = img_count
                self.info_label.setText(f"{img_count} images")
            else:
                self._total_frames = 0
                self.info_label.setText("")
        else:
            if os.path.isdir(input_path):
                ply_count = len([
                    f for f in os.listdir(input_path) if f.lower().endswith(".ply")
                ])
                self._total_frames = ply_count
                self.info_label.setText(f"{ply_count} PLY files")
            else:
                self._total_frames = 0
                self.info_label.setText("")

        # Reset range to frames mode
        self.range_unit.setCurrentIndex(0)
        max_frame = max(self._total_frames, 1)
        self.start_spin.setDecimals(0)
        self.start_spin.setSingleStep(1)
        self.start_spin.setRange(1, max_frame)
        self.start_spin.setValue(1)
        self.end_spin.setDecimals(0)
        self.end_spin.setSingleStep(1)
        self.end_spin.setRange(1, max_frame)
        self.end_spin.setValue(max_frame)
        self.range_note.setText(f"(of {self._total_frames})")
```

- [ ] **Step 7: Update `_on_generate` to pass correct mode**

In `_on_generate`, replace the mode determination (line 611-620):

```python
        is_video = getattr(self, '_current_mode', 0) == 0
        input_source = getattr(self, '_input_source', 'video')
        if is_video and input_source == "video":
            worker_mode = "video"
        elif is_video and input_source == "images":
            worker_mode = "images"
        else:
            worker_mode = "ply"

        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.step_label.setText("")
        self.eta_label.setText("")
        self._eta_start_time = time.time()

        self.worker = PipelineWorker(
            mode=worker_mode,
            input_path=input_path,
            output_path=output_path,
            fps=float(self.fps_spin.value()),
            start_frame=self._get_start_frame(),
            end_frame=self._get_end_frame(),
            frame_step=self._get_frame_step(),
            keep_images=self.chk_keep_images.isChecked(),
            keep_ply=self.chk_keep_ply.isChecked(),
            skip_gsd=self.chk_skip_gsd.isChecked(),
        )
```

- [ ] **Step 8: Update `_on_clear` to reset input source**

Add to `_on_clear` (after line 569):

```python
        self._set_input_source("video")
```

- [ ] **Step 9: Verify GUI launches**

Run: `python -m app.converter`
Expected: GUI opens with Input Source buttons visible in Video tab, hidden in PLY tab.

- [ ] **Step 10: Commit**

```bash
git add app/converter/main_window.py
git commit -m "feat: add Image Sequence input source to GUI

Video to 4DGS tab now has Input Source selector: Video File or
Image Sequence. Image Sequence skips ffmpeg, only requires sharp.
Dependency gating updated accordingly."
```

---

### Task 4: SHARP Install button improvement

**Files:**
- Modify: `app/converter/main_window.py:504-537` (`_install_dep`)
- Modify: `app/converter/main_window.py:236-252` (env status tooltip)

- [ ] **Step 1: Update `_install_dep` for sharp**

Replace the `elif name == "sharp":` block in `_install_dep`:

```python
        elif name == "sharp":
            # Determine install command based on whether ml-sharp source exists
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ml_sharp_dir = os.path.join(app_dir, "ml-sharp")
            has_source = (
                os.path.isdir(ml_sharp_dir)
                and os.path.isfile(os.path.join(ml_sharp_dir, "pyproject.toml"))
            )

            if has_source:
                install_cmd = f'pip install -e "{ml_sharp_dir}"'
                detail = f"  pip install -e {ml_sharp_dir}"
            else:
                install_cmd = (
                    f'git clone https://github.com/apple/ml-sharp "{ml_sharp_dir}"'
                    f' && pip install -e "{ml_sharp_dir}"'
                )
                detail = (
                    f"  git clone https://github.com/apple/ml-sharp\n"
                    f"  pip install -e {ml_sharp_dir}"
                )

            reply = QMessageBox.question(
                self, "Install SHARP (ml-sharp)",
                f"Install ml-sharp?\n\n"
                f"This will run:\n{detail}\n\n"
                f"ml-sharp has large dependencies (PyTorch, etc.)\n"
                f"and may take several minutes to install.\n\n"
                f"After installation, restart 4DGS Converter.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                import subprocess
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["cmd", "/c", install_cmd + " & pause"],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
                else:
                    subprocess.Popen(
                        ["bash", "-c", install_cmd + '; echo "Done. Press enter to close."; read'],
                        start_new_session=True,
                    )
```

Note: `import sys` is already used at the top of the file (via PySide6 imports). Add `import sys` to the top of `main_window.py` if not present.

- [ ] **Step 2: Add tooltip to sharp ✗ label**

In `_build_ui`, update the env status bar section (lines 236-252). Replace:

```python
            lbl = QLabel(f'<span style="color:{color}">{icon}</span> {name}')
            env_row.addWidget(lbl)
```

With:

```python
            lbl = QLabel(f'<span style="color:{color}">{icon}</span> {name}')
            if not available and name == "sharp":
                lbl.setToolTip(
                    "sharp not found in current Python environment.\n"
                    "Click Install to install here, or activate\n"
                    "the environment where sharp is installed."
                )
            env_row.addWidget(lbl)
```

- [ ] **Step 3: Add `import sys` if missing**

Check the imports at the top of `main_window.py`. If `import sys` is not present, add it after `import time`:

```python
import os
import sys
import time
```

- [ ] **Step 4: Verify GUI launches and Install button works**

Run: `python -m app.converter`
Expected: If sharp is missing, clicking Install shows confirmation dialog, then opens a console running pip.

- [ ] **Step 5: Commit**

```bash
git add app/converter/main_window.py
git commit -m "feat: one-click SHARP install button

Replace text-only dialog with actual pip install action.
Auto-detects whether ml-sharp source exists locally.
Adds tooltip explaining PATH issues when sharp is not found."
```

---

### Task 5: Manual verification

No files changed — this task verifies everything works together.

- [ ] **Step 1: Verify GUI — Video File source**

Run: `python -m app.converter`

1. Confirm "Video to 4DGS" tab shows Input Source: Video File / Image Sequence
2. With "Video File" selected, Browse opens file picker
3. FPS is read-only, auto-detected
4. "Keep image sequence" checkbox is visible

- [ ] **Step 2: Verify GUI — Image Sequence source**

1. Click "Image Sequence"
2. Browse opens folder picker
3. Select a folder with images → info label shows "N images"
4. FPS is editable, shows "(set manually)"
5. "Keep image sequence" checkbox is hidden
6. Generate button enabled if sharp is available (regardless of ffmpeg status)

- [ ] **Step 3: Verify GUI — PLY tab unchanged**

1. Switch to "3DGS Sequence to 4DGS" tab
2. Input Source buttons are hidden
3. Everything works as before

- [ ] **Step 4: Verify CLI auto-detect**

```bash
# Should auto-detect as images mode if folder has images
python -m app.converter --cli -i /path/to/images/folder -o test.gsd --skip-gsd
# Expected output starts with: Mode: images

# Should auto-detect as ply mode if folder has .ply files
python -m app.converter --cli -i /path/to/ply/folder -o test.gsd
# Expected output starts with: Mode: ply
```

- [ ] **Step 5: Verify CLI --mode images**

```bash
python -m app.converter --cli --help
# Verify --mode shows {auto,video,images,ply}
```

- [ ] **Step 6: Final commit (version bump)**

Update `app/converter/__init__.py`:

```python
__version__ = "1.1.0"
```

```bash
git add app/converter/__init__.py
git commit -m "chore: bump version to 1.1.0"
```
