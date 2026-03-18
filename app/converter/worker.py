"""QThread worker for Video-to-GSD pipeline execution."""

import os
import shutil
import signal
import time

from PySide6.QtCore import QThread, Signal


class StopRequested(Exception):
    pass


class PipelineWorker(QThread):
    """Runs the conversion pipeline in a background thread.

    Signals:
        progress(int, int, str): (current_step, total_steps, step_label)
        frame_progress(int, int): (current_frame, total_frames)
        log_message(str): log line for the UI
        finished_ok(str): emitted on success with output path
        finished_error(str): emitted on failure with error message
    """

    progress = Signal(int, int, str)
    frame_progress = Signal(int, int)
    log_message = Signal(str)
    finished_ok = Signal(str)
    finished_error = Signal(str)

    def __init__(
        self,
        mode: str,              # "video" or "ply"
        input_path: str,        # video file or PLY folder
        output_path: str,       # .gsd output path
        fps: float,
        start_frame: int = 0,
        end_frame: int = -1,
        frame_step: int = 1,
        keep_images: bool = True,
        keep_ply: bool = True,
        skip_gsd: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.input_path = input_path
        self.output_path = output_path
        self.fps = fps
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_step = frame_step
        self.keep_images = keep_images
        self.keep_ply = keep_ply
        self.skip_gsd = skip_gsd
        self._stop_requested = False
        self.images_folder = None
        self.ply_folder = None

    def request_stop(self):
        self._stop_requested = True
        # Kill any tracked subprocess immediately
        if hasattr(self, '_active_process') and self._active_process is not None:
            try:
                self._active_process.kill()
            except OSError:
                pass

    def _check_stop(self):
        if self._stop_requested:
            raise StopRequested("Stopped by user")

    def _log(self, msg: str):
        self._check_stop()
        self.log_message.emit(msg)

    def _frame_progress(self, cur: int, total: int):
        self._check_stop()
        self.frame_progress.emit(cur, total)

    def _derive_paths(self):
        """Derive intermediate folder paths from input/output."""
        if self.mode == "video":
            parent = os.path.dirname(self.output_path)
            self.images_folder = os.path.join(parent, "images")
            self.ply_folder = os.path.join(parent, "ply")
        else:
            self.ply_folder = self.input_path
            self.images_folder = None

    def run(self):
        try:
            self._derive_paths()

            if self.mode == "video":
                self._run_video_pipeline()
            else:
                self._run_ply_pipeline()

            self.finished_ok.emit(self.output_path)

        except StopRequested:
            self.log_message.emit("Stopped by user.")
            if os.path.exists(self.output_path):
                try:
                    os.remove(self.output_path)
                    self.log_message.emit(f"Deleted partial: {self.output_path}")
                except OSError:
                    pass
            self.finished_error.emit("Stopped by user")

        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            self.finished_error.emit(str(e))

    def _run_video_pipeline(self):
        total_steps = 2 if self.skip_gsd else 3
        step = 0

        # Step 1: Extract frames
        step += 1
        self.progress.emit(step, total_steps, "Extracting frames...")
        self._extract_frames()
        self._check_stop()

        # Step 2: Generate PLY
        step += 1
        self.progress.emit(step, total_steps, "Generating PLY (SHARP)...")
        self._generate_ply()
        self._check_stop()

        # Step 3: Convert to GSD
        if not self.skip_gsd:
            step += 1
            self.progress.emit(step, total_steps, "Converting to GSD...")
            self._convert_to_gsd()

        # Cleanup
        self._cleanup()

    def _run_ply_pipeline(self):
        self.progress.emit(1, 1, "Converting to GSD...")
        self._convert_to_gsd()

    def _extract_frames(self):
        from app.pipeline.video_to_images import (
            extract_frames,
            get_video_frame_count,
        )

        os.makedirs(self.images_folder, exist_ok=True)

        total_frames = get_video_frame_count(self.input_path)
        if total_frames <= 0:
            raise RuntimeError(f"Could not determine frame count for {self.input_path}")

        # Calculate expected extracted count
        start = self.start_frame
        end = self.end_frame if self.end_frame >= 0 else total_frames - 1
        expected = end - start + 1

        # Resume: skip if images folder already has enough files
        existing = [f for f in os.listdir(self.images_folder) if f.endswith(".jpg")]
        if len(existing) >= expected:
            self._log(f"Frames already extracted ({len(existing)} files), skipping.")
            return

        self._log(f"Extracting frames {start}-{end} ({expected} of {total_frames})...")
        extract_frames(
            video_path=self.input_path,
            output_folder=self.images_folder,
            frame_count=total_frames,
            fps=self.fps,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            progress_callback=self._log,
        )

    def _generate_ply(self):
        import re
        import subprocess

        os.makedirs(self.ply_folder, exist_ok=True)

        # Resume: count existing PLY vs images
        image_count = len([f for f in os.listdir(self.images_folder) if f.endswith(".jpg")])
        ply_count = len([f for f in os.listdir(self.ply_folder) if f.endswith(".ply")])
        if ply_count >= image_count and ply_count > 0:
            self._log(f"PLY files already exist ({ply_count} files), skipping.")
            return

        self._log(f"Running SHARP predict on {image_count} images...")

        cmd = [
            "sharp", "predict",
            "-i", self.images_folder,
            "-o", self.ply_folder,
            "--no-render",
        ]

        self._active_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        total_images = 0
        processed = 0

        try:
            for line in self._active_process.stdout:
                self._check_stop()
                line = line.rstrip()
                if not line:
                    continue
                self._log(f"  [ml-sharp] {line}")

                m = re.search(r"Processing (\d+) valid image files", line)
                if m:
                    total_images = int(m.group(1))

                if "Processing " in line and any(
                    ext in line.lower() for ext in (".jpg", ".png", ".jpeg", ".heic")
                ):
                    processed += 1
                    if total_images > 0:
                        self._frame_progress(processed, total_images)

            self._active_process.wait()
            if self._active_process.returncode != 0:
                raise RuntimeError(f"ml-sharp failed with exit code {self._active_process.returncode}")
        finally:
            self._active_process = None

    def _convert_to_gsd(self):
        from app.pipeline.ply_to_gsd import convert_ply_to_gsd

        sequence_name = os.path.splitext(os.path.basename(self.output_path))[0]
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        step_str = f", step {self.frame_step}" if self.frame_step > 1 else ""
        self._log(f"Converting PLY → GSD at {self.fps} FPS (frames {self.start_frame}-{self.end_frame}{step_str})...")
        convert_ply_to_gsd(
            ply_folder=self.ply_folder,
            output_path=self.output_path,
            sequence_name=sequence_name,
            target_fps=self.fps,
            start_frame=self.start_frame,
            end_frame=self.end_frame if self.end_frame >= 0 else None,
            frame_step=self.frame_step,
            progress_callback=self._log,
            frame_progress_callback=self._frame_progress,
        )

    def _cleanup(self):
        """Delete intermediate folders based on keep flags."""
        if self.mode == "video":
            if not self.keep_images and self.images_folder and os.path.isdir(self.images_folder):
                self._log(f"Cleaning up images: {self.images_folder}")
                shutil.rmtree(self.images_folder, ignore_errors=True)
            if not self.keep_ply and self.ply_folder and os.path.isdir(self.ply_folder):
                self._log(f"Cleaning up PLY: {self.ply_folder}")
                shutil.rmtree(self.ply_folder, ignore_errors=True)
