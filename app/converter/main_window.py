"""PySide6 main window for 4DGS Converter."""

import os
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QFont, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.converter.env_check import check_all
from app.converter.worker import PipelineWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from app.converter import __version__
        self.setWindowTitle(f"4DGS Converter v{__version__}")
        self.setMinimumWidth(620)
        self.resize(620, 520)

        # Also set window-level icon as backup
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.worker = None
        self._eta_start_time = 0.0
        self._total_frames = 0
        self._current_mode = 0
        self._input_source = "video"

        self._env = check_all()
        self._build_ui()

    def closeEvent(self, event: QCloseEvent):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.quit()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
        event.accept()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        # -- Mode selector (tab-style toggle buttons)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        self._mode_buttons = []
        tab_style_active = (
            "QPushButton { background: #3a3a3a; border: 1px solid #555; "
            "border-bottom: 2px solid #4fc3f7; padding: 8px 16px; font-weight: bold; }"
        )
        tab_style_inactive = (
            "QPushButton { background: #2a2a2a; border: 1px solid #444; "
            "border-bottom: 1px solid #444; padding: 8px 16px; color: #888; }"
            "QPushButton:hover { background: #333; color: #ccc; }"
        )
        for i, label in enumerate(["Video to 4DGS", "3DGS Sequence to 4DGS"]):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setStyleSheet(tab_style_active if i == 0 else tab_style_inactive)
            btn.clicked.connect(lambda _, idx=i: self._set_mode(idx))
            mode_row.addWidget(btn, 1)
            self._mode_buttons.append(btn)
        self._tab_style_active = tab_style_active
        self._tab_style_inactive = tab_style_inactive
        layout.addLayout(mode_row)

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

        # -- Input
        input_row = QHBoxLayout()
        self.input_label = QLabel("Input:")
        input_row.addWidget(self.input_label)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select a video file...")
        input_row.addWidget(self.input_edit, 1)
        self.input_btn = QPushButton("Browse")
        self.input_btn.clicked.connect(self._browse_input)
        input_row.addWidget(self.input_btn)
        layout.addLayout(input_row)

        # -- Info label (frame count, duration)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.info_label)

        # -- Source FPS (PLY mode only)
        self.source_fps_row = QHBoxLayout()
        self.source_fps_label = QLabel("Source FPS:")
        self.source_fps_row.addWidget(self.source_fps_label)
        self.source_fps_spin = QSpinBox()
        self.source_fps_spin.setRange(1, 240)
        self.source_fps_spin.setValue(30)
        self.source_fps_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.source_fps_spin.valueChanged.connect(self._update_fps_note)
        self.source_fps_row.addWidget(self.source_fps_spin)
        self.source_fps_note = QLabel("(original video framerate)")
        self.source_fps_note.setStyleSheet("color: gray;")
        self.source_fps_row.addWidget(self.source_fps_note)
        self.source_fps_row.addStretch()
        layout.addLayout(self.source_fps_row)

        # -- Target FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Target FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(30)
        self.fps_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.fps_spin.valueChanged.connect(self._update_fps_note)
        fps_row.addWidget(self.fps_spin)
        self.fps_note = QLabel("(auto-detected)")
        fps_row.addWidget(self.fps_note)
        fps_row.addStretch()
        layout.addLayout(fps_row)

        # -- Range (frames or seconds)
        range_row = QHBoxLayout()
        self.range_unit = QComboBox()
        self.range_unit.addItems(["Frames", "Seconds"])
        self.range_unit.setFixedWidth(90)
        self.range_unit.currentIndexChanged.connect(self._on_range_unit_changed)
        range_row.addWidget(self.range_unit)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(1, 1)
        self.start_spin.setValue(1)
        self.start_spin.setDecimals(0)
        self.start_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        range_row.addWidget(self.start_spin)
        range_row.addWidget(QLabel("to"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(1, 1)
        self.end_spin.setValue(1)
        self.end_spin.setDecimals(0)
        self.end_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        range_row.addWidget(self.end_spin)
        self.range_note = QLabel("")
        self.range_note.setStyleSheet("color: gray;")
        range_row.addWidget(self.range_note)
        range_row.addStretch()
        layout.addLayout(range_row)

        # -- Output
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output .gsd path...")
        out_row.addWidget(self.output_edit, 1)
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_btn)
        layout.addLayout(out_row)

        # -- Checkboxes
        self.chk_keep_images = QCheckBox("Keep image sequence")
        self.chk_keep_images.setChecked(True)
        self.chk_keep_ply = QCheckBox("Keep 3DGS sequence (.ply)")
        self.chk_keep_ply.setChecked(True)
        self.chk_skip_gsd = QCheckBox("Skip 4DGS (.gsd)")
        self.chk_skip_gsd.stateChanged.connect(self._on_skip_gsd_changed)
        layout.addWidget(self.chk_keep_images)
        layout.addWidget(self.chk_keep_ply)
        layout.addWidget(self.chk_skip_gsd)

        # -- Generate / Stop / Clear
        btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumHeight(36)
        font = self.generate_btn.font()
        font.setBold(True)
        self.generate_btn.setFont(font)
        self.generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self.generate_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMinimumHeight(36)
        self.clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self.clear_btn)

        layout.addLayout(btn_row)

        # -- Progress
        self.step_label = QLabel("")
        layout.addWidget(self.step_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        layout.addWidget(self.eta_label)

        # -- Log (collapsible)
        self.log_toggle = QPushButton("\u25b6 Log")
        self.log_toggle.setFlat(True)
        self.log_toggle.setStyleSheet("text-align: left; padding: 2px;")
        self.log_toggle.clicked.connect(self._toggle_log)
        layout.addWidget(self.log_toggle)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)

        # -- Environment status bar
        env_row = QHBoxLayout()
        env_row.addWidget(QLabel("Environment:"))
        for name, available in self._env.items():
            icon = "\u2713" if available else "\u2717"
            color = "green" if available else "red"
            lbl = QLabel(f'<span style="color:{color}">{icon}</span> {name}')
            if not available and name == "sharp":
                lbl.setToolTip(
                    "sharp not found in current Python environment.\n"
                    "Click Install to install here, or activate\n"
                    "the environment where sharp is installed."
                )
            env_row.addWidget(lbl)
            if not available:
                install_btn = QPushButton("Install")
                install_btn.setFixedHeight(22)
                install_btn.setFixedWidth(50)
                install_btn.setStyleSheet("font-size: 11px;")
                install_btn.clicked.connect(lambda _, n=name: self._install_dep(n))
                env_row.addWidget(install_btn)
        env_row.addStretch()
        layout.addLayout(env_row)

        # -- About (collapsible)
        self.about_toggle = QPushButton("\u25b6 About")
        self.about_toggle.setFlat(True)
        self.about_toggle.setStyleSheet("text-align: left; padding: 2px;")
        self.about_toggle.clicked.connect(self._toggle_about)
        layout.addWidget(self.about_toggle)

        self.about_text = QLabel(
            "<b>How to Use</b><br>"
            "1. Select mode: <i>Video to 4DGS</i> or <i>3DGS Sequence to 4DGS</i><br>"
            "2. Browse for input (video file or PLY folder)<br>"
            "3. Adjust Target FPS and frame range if needed<br>"
            "4. Click <b>Generate</b><br><br>"
            "<b>FPS &amp; Frame Step</b><br>"
            "In <i>3DGS Sequence</i> mode, set <b>Source FPS</b> to your original video framerate, "
            "then set <b>Target FPS</b> to your desired output. "
            "If Target &lt; Source, frames are automatically skipped to match.<br>"
            "Example: Source 60 fps, Target 30 fps \u2192 every 2nd frame is used (half the data).<br><br>"
            "<b>Pipeline</b><br>"
            "Video \u2192 Images (ffmpeg) \u2192 3DGS (.ply) (SHARP) \u2192 4DGS (.gsd)<br><br>"
            "<b>4DGS (.gsd)</b> \u2014 Gaussian Stream Data. Compressed format for real-time "
            "4D Gaussian Splatting playback. Byte-Shuffle + LZ4, frame-independent random access.<br>"
            "Typical compression: ~30-70% of raw size.<br><br>"
            "<b>3DGS to 4DGS</b> \u2014 Any 3DGS (.ply) sequence can be packed into "
            "4DGS (.gsd), regardless of model (SHARP, PostShot, Nerfstudio, etc.).<br><br>"
            "<b>CLI</b><br>"
            "<code>python -m app.converter --cli -i input -o output.gsd</code><br><br>"
            "<b>Built by <a href='https://github.com/DazaiStudio'>Dazai Studio</a></b> | "
            "<a href='https://github.com/DazaiStudio/4dgs-converter'>GitHub</a><br>"
            "<b>Version</b> " + __import__('app.converter', fromlist=['__version__']).__version__
        )
        self.about_text.setWordWrap(True)
        self.about_text.setOpenExternalLinks(True)
        self.about_text.setStyleSheet("color: gray; padding: 4px 8px;")

        self.about_scroll = QScrollArea()
        self.about_scroll.setWidget(self.about_text)
        self.about_scroll.setWidgetResizable(True)
        self.about_scroll.setMaximumHeight(300)
        self.about_scroll.setVisible(False)
        layout.addWidget(self.about_scroll)

        self._on_mode_changed()

    # ------------------------------------------------------------ Mode logic
    def _set_mode(self, idx: int):
        """Switch mode via tab buttons."""
        self._current_mode = idx
        for i, btn in enumerate(self._mode_buttons):
            btn.setChecked(i == idx)
            btn.setStyleSheet(
                self._tab_style_active if i == idx else self._tab_style_inactive
            )
        self._on_mode_changed()

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
        self.chk_keep_ply.setVisible(is_video)
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

    def _update_fps_note(self):
        """Update the FPS note to show frame step info in PLY mode."""
        is_video = getattr(self, '_current_mode', 0) == 0
        if is_video:
            return
        source = self.source_fps_spin.value()
        target = self.fps_spin.value()
        step = max(1, round(source / target))
        if step > 1:
            effective_fps = source / step
            out_frames = max(self._total_frames // step, 1) if self._total_frames > 0 else 0
            duration = f", {out_frames / effective_fps:.1f}s" if out_frames > 0 else ""
            self.fps_note.setText(
                f"(every {step} frames → {out_frames} frames{duration})"
            )
        else:
            self.fps_note.setText("(all frames)")

    def _get_frame_step(self) -> int:
        """Calculate frame step based on source/target FPS."""
        is_video = getattr(self, '_current_mode', 0) == 0
        if is_video:
            return 1
        source = self.source_fps_spin.value()
        target = self.fps_spin.value()
        return max(1, round(source / target))

    def _on_range_unit_changed(self):
        """Switch between Frames (1-based int) and Seconds (float) display."""
        is_seconds = self.range_unit.currentIndex() == 1
        fps = max(self.fps_spin.value(), 1)
        max_frame = max(self._total_frames, 1)

        if is_seconds:
            # Convert current frame values to seconds
            old_start = self.start_spin.value()
            old_end = self.end_spin.value()
            self.start_spin.setDecimals(1)
            self.start_spin.setSingleStep(0.1)
            self.start_spin.setRange(0.0, (max_frame - 1) / fps)
            self.start_spin.setValue((old_start - 1) / fps)
            self.end_spin.setDecimals(1)
            self.end_spin.setSingleStep(0.1)
            self.end_spin.setRange(0.0, (max_frame - 1) / fps)
            self.end_spin.setValue((old_end - 1) / fps)
            self.range_note.setText(f"({max_frame} frames @ {fps} fps)")
        else:
            # Convert current seconds values to frames
            old_start = self.start_spin.value()
            old_end = self.end_spin.value()
            self.start_spin.setDecimals(0)
            self.start_spin.setSingleStep(1)
            self.start_spin.setRange(1, max_frame)
            self.start_spin.setValue(round(old_start * fps) + 1)
            self.end_spin.setDecimals(0)
            self.end_spin.setSingleStep(1)
            self.end_spin.setRange(1, max_frame)
            self.end_spin.setValue(round(old_end * fps) + 1)
            self.range_note.setText(f"(of {self._total_frames})")

    def _on_skip_gsd_changed(self):
        if self.chk_skip_gsd.isChecked():
            self.chk_keep_ply.setChecked(True)
            self.chk_keep_ply.setEnabled(False)
        else:
            self.chk_keep_ply.setEnabled(True)

    # ----------------------------------------------------------- File browse
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
                self.fps_note.setText("(detection failed \u2014 set manually)")
                self.fps_spin.setReadOnly(False)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GSD", self.output_edit.text(),
            "GSD Files (*.gsd);;All Files (*)",
        )
        if path:
            self.output_edit.setText(path)

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

    # ------------------------------------------------- Install dependencies
    def _run_install_bat(self, title: str, commands: list[str]):
        """Run install commands in a new console via a temp .bat file.

        Shows success/failure and reminds user to restart.
        The console stays open (pause) so the user can see the result.
        """
        import subprocess
        import tempfile
        bat = os.path.join(tempfile.gettempdir(), "_4dgs_install.bat")
        with open(bat, "w") as f:
            f.write(f"@echo === {title} ===\n@echo.\n")
            for cmd in commands:
                f.write(f"@{cmd}\n")
                f.write(f"@if errorlevel 1 (\n")
                f.write(f"  @echo.\n  @echo [FAILED] {title}\n  @echo.\n  @pause\n  @exit /b 1\n)\n")
            f.write(f"@echo.\n")
            f.write(f"@echo [OK] {title} succeeded.\n")
            f.write(f"@echo Please restart 4DGS Converter.\n")
            f.write(f"@echo.\n@pause\n")
        subprocess.Popen(
            ["cmd", "/c", bat],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    def _install_dep(self, name: str):
        import subprocess
        python = sys.executable
        if name == "lz4":
            reply = QMessageBox.question(
                self, "Install lz4",
                "Run: pip install lz4?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._run_install_bat("Install lz4", [
                    f'"{python}" -m pip install lz4',
                ])
        elif name == "ffmpeg":
            import webbrowser
            reply = QMessageBox.question(
                self, "Install ffmpeg",
                "ffmpeg is required for video frame extraction.\n\n"
                "Open the ffmpeg download page?\n"
                "After installing, add ffmpeg to your system PATH\n"
                "and restart 4DGS Converter.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                webbrowser.open("https://ffmpeg.org/download.html")
        elif name == "sharp":
            # Determine install command based on whether ml-sharp source exists
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ml_sharp_dir = os.path.join(app_dir, "ml-sharp")
            has_source = (
                os.path.isdir(ml_sharp_dir)
                and os.path.isfile(os.path.join(ml_sharp_dir, "pyproject.toml"))
            )

            if has_source:
                detail = f"  pip install -e {ml_sharp_dir}"
            else:
                detail = (
                    f"  git clone https://github.com/apple/ml-sharp\n"
                    f"  pip install -e {ml_sharp_dir}"
                )

            reply = QMessageBox.question(
                self, "Install SHARP (ml-sharp)",
                f"Install ml-sharp?\n\n"
                f"This will run:\n{detail}\n\n"
                f"ml-sharp has large dependencies (PyTorch, etc.)\n"
                f"and may take several minutes to install.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                commands = []
                if not has_source:
                    commands.append(f'git clone https://github.com/apple/ml-sharp "{ml_sharp_dir}"')
                commands.append(f'"{python}" -m pip install -e "{ml_sharp_dir}"')
                self._run_install_bat("Install SHARP (ml-sharp)", commands)

    # ------------------------------------------------- Frame range helpers
    def _get_start_frame(self) -> int:
        """Convert UI value to 0-based frame index."""
        if self.range_unit.currentIndex() == 1:  # Seconds
            fps = max(self.fps_spin.value(), 1)
            return int(self.start_spin.value() * fps)
        return int(self.start_spin.value()) - 1

    def _get_end_frame(self) -> int:
        """Convert UI value to 0-based frame index."""
        if self.range_unit.currentIndex() == 1:  # Seconds
            fps = max(self.fps_spin.value(), 1)
            return int(self.end_spin.value() * fps)
        return int(self.end_spin.value()) - 1

    # --------------------------------------------------------- Toggles
    def _toggle_about(self):
        visible = not self.about_scroll.isVisible()
        self.about_scroll.setVisible(visible)
        self.about_toggle.setText("\u25bc About" if visible else "\u25b6 About")

    def _toggle_log(self):
        visible = not self.log_text.isVisible()
        self.log_text.setVisible(visible)
        self.log_toggle.setText("\u25bc Log" if visible else "\u25b6 Log")

    # ----------------------------------------------------------- Clear
    def _on_clear(self):
        self.input_edit.clear()
        self.output_edit.clear()
        self.source_fps_spin.setValue(30)
        self.fps_spin.setValue(30)
        self.fps_note.setText("(auto-detected)")
        self.info_label.setText("")
        self.range_unit.setCurrentIndex(0)
        self.start_spin.setDecimals(0)
        self.start_spin.setSingleStep(1)
        self.start_spin.setRange(1, 1)
        self.start_spin.setValue(1)
        self.end_spin.setDecimals(0)
        self.end_spin.setSingleStep(1)
        self.end_spin.setRange(1, 1)
        self.end_spin.setValue(1)
        self.range_note.setText("")
        self._total_frames = 0
        self.step_label.setText("")
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.log_text.clear()
        self._set_input_source("video")

    # ------------------------------------------------------- Generate / Stop
    def _on_generate(self):
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()

        if not input_path:
            QMessageBox.warning(self, "Error", "Please select an input.")
            return
        if not output_path:
            QMessageBox.warning(self, "Error", "Please set an output path.")
            return

        # Check overwrite
        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"{output_path} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

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
        self.worker.progress.connect(self._on_progress)
        self.worker.frame_progress.connect(self._on_frame_progress)
        self.worker.log_message.connect(self._on_log)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_error.connect(self._on_finished_error)

        self._set_running(True)
        self.worker.start()

    def _on_stop(self):
        if self.worker:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")

    def _set_running(self, running: bool):
        self.generate_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.stop_btn.setText("Stop")
        self.clear_btn.setEnabled(not running)
        for btn in self._mode_buttons:
            btn.setEnabled(not running)
        self._source_video.setEnabled(not running)
        self._source_images.setEnabled(not running)
        self.input_btn.setEnabled(not running)
        self.output_btn.setEnabled(not running)
        self.fps_spin.setEnabled(not running)
        self.start_spin.setEnabled(not running)
        self.end_spin.setEnabled(not running)

    # -------------------------------------------------------- Signal handlers
    def _on_progress(self, step: int, total: int, label: str):
        self.step_label.setText(f"Step {step}/{total}: {label}")
        self._eta_start_time = time.time()
        self.progress_bar.setValue(0)
        self.eta_label.setText("")

    def _on_frame_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self.progress_bar.setValue(pct)

            elapsed = time.time() - self._eta_start_time
            if current > 0:
                eta = elapsed / current * (total - current)
                if eta >= 60:
                    self.eta_label.setText(f"{pct}%  ETA: {eta / 60:.1f}m")
                else:
                    self.eta_label.setText(f"{pct}%  ETA: {eta:.0f}s")

    def _on_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")

    def _on_finished_ok(self, output_path: str):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.eta_label.setText("")
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1e6
            self.step_label.setText(f"Done! ({size_mb:.1f} MB)")
            QMessageBox.information(
                self, "Complete",
                f"Output: {output_path}\nSize: {size_mb:.1f} MB",
            )
        else:
            self.step_label.setText("Done (file not found?)")
            QMessageBox.warning(
                self, "Warning",
                f"Pipeline finished but output not found:\n{output_path}",
            )

    def _on_finished_error(self, error: str):
        self._set_running(False)
        self.step_label.setText(f"Error: {error}")
        self.eta_label.setText("")
