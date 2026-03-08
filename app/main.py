"""4DGS Pipeline Tool - Video to UE-Compatible RAW.

Tkinter GUI application for the complete pipeline:
  Step 1: Video -> Images (ffmpeg)
  Step 2: Images -> PLY (ml-sharp)
  Step 3: PLY -> RAW (UE GaussianStreamer format)

Only requires selecting a video file. All intermediate paths are auto-derived.
"""

import os
import signal
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.video_to_images import extract_frames, check_ffmpeg
from app.pipeline.images_to_ply import generate_ply
from app.pipeline.ply_to_raw import (
    convert_ply_sequence,
    PRECISION_FULL,
    PRECISION_HALF,
)
from app.pipeline.raw_to_gsd import convert_raw_to_gsd


class StopRequested(Exception):
    """Raised when user clicks Stop."""
    pass


class PipelineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("4DGS Pipeline Tool")
        self.root.geometry("720x780")
        self.root.minsize(680, 700)

        self._running = False
        self._stop_flag = False
        self._subprocess = None
        self._log_visible = True
        self._video_base = None  # base path for auto-derived folders
        self._build_ui()

        # Update RAW path when prune settings change
        self.prune_enabled.trace_add("write", lambda *_: self._update_raw_path())
        self.prune_keep_pct.trace_add("write", lambda *_: self._update_raw_path())

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Input: Video File ===
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding=8)
        input_frame.pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(input_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Video File:").pack(side=tk.LEFT)
        self.video_path = tk.StringVar()
        ttk.Entry(row, textvariable=self.video_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(row, text="Browse", command=self._browse_video).pack(side=tk.LEFT)

        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Frame Count:").pack(side=tk.LEFT)
        self.frame_count = tk.StringVar(value="100")
        ttk.Entry(row2, textvariable=self.frame_count, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="Device:").pack(side=tk.LEFT, padx=(15, 0))
        self.device = tk.StringVar(value="cuda")
        ttk.Combobox(row2, textvariable=self.device, values=["cuda", "cpu"], width=8, state="readonly").pack(side=tk.LEFT, padx=5)

        # === Auto-derived Paths ===
        paths_frame = ttk.LabelFrame(main_frame, text="Output Paths (auto-derived, click Browse to override)", padding=8)
        paths_frame.pack(fill=tk.X, pady=(0, 5))

        self.images_output = tk.StringVar()
        self.ply_output = tk.StringVar()
        self.raw_output = tk.StringVar()
        self.gsd_output = tk.StringVar()
        self.seq_name = tk.StringVar(value="sequence")

        for label_text, var, browse_cmd in [
            ("Frames:", self.images_output, self._browse_images_output),
            ("PLY:", self.ply_output, self._browse_ply_output),
            ("RAW:", self.raw_output, self._browse_raw_output),
            ("GSD:", self.gsd_output, self._browse_gsd_output),
        ]:
            r = ttk.Frame(paths_frame)
            r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=label_text, width=8).pack(side=tk.LEFT)
            ttk.Entry(r, textvariable=var, width=55).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Button(r, text="Browse", command=browse_cmd).pack(side=tk.LEFT)

        # === RAW Conversion Settings ===
        settings_frame = ttk.LabelFrame(main_frame, text="RAW Conversion Settings", padding=8)
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Sequence Name:").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.seq_name, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(row3, text="FPS:").pack(side=tk.LEFT, padx=(15, 0))
        self.target_fps = tk.StringVar(value="30")
        ttk.Entry(row3, textvariable=self.target_fps, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(row3, text="SH Degree:").pack(side=tk.LEFT, padx=(15, 0))
        self.sh_degree = tk.StringVar(value="0")
        ttk.Combobox(row3, textvariable=self.sh_degree, values=["0", "1", "2", "3"], width=3, state="readonly").pack(side=tk.LEFT, padx=5)

        row4 = ttk.Frame(settings_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="Precision:").pack(side=tk.LEFT)

        prec_values = ["32bit (Full)", "16bit (Half)"]

        ttk.Label(row4, text="Pos").pack(side=tk.LEFT, padx=(10, 0))
        self.pos_prec = tk.StringVar(value=prec_values[0])
        ttk.Combobox(row4, textvariable=self.pos_prec, values=prec_values, width=12, state="readonly").pack(side=tk.LEFT, padx=2)

        ttk.Label(row4, text="Rot").pack(side=tk.LEFT, padx=(8, 0))
        self.rot_prec = tk.StringVar(value=prec_values[1])
        ttk.Combobox(row4, textvariable=self.rot_prec, values=prec_values, width=12, state="readonly").pack(side=tk.LEFT, padx=2)

        ttk.Label(row4, text="Scale").pack(side=tk.LEFT, padx=(8, 0))
        self.scale_prec = tk.StringVar(value=prec_values[1])
        ttk.Combobox(row4, textvariable=self.scale_prec, values=prec_values, width=12, state="readonly").pack(side=tk.LEFT, padx=2)

        ttk.Label(row4, text="SH").pack(side=tk.LEFT, padx=(8, 0))
        self.sh_prec = tk.StringVar(value=prec_values[1])
        ttk.Combobox(row4, textvariable=self.sh_prec, values=prec_values, width=12, state="readonly").pack(side=tk.LEFT, padx=2)

        row5 = ttk.Frame(settings_frame)
        row5.pack(fill=tk.X, pady=(4, 0))
        self.prune_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(row5, text="Prune by contribution", variable=self.prune_enabled).pack(side=tk.LEFT)
        ttk.Label(row5, text="Keep top:").pack(side=tk.LEFT, padx=(15, 0))
        self.prune_keep_pct = tk.StringVar(value="50")
        ttk.Entry(row5, textvariable=self.prune_keep_pct, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row5, text="%").pack(side=tk.LEFT)
        ttk.Label(row5, text="(ranked by opacity \u00d7 volume)", foreground="gray").pack(side=tk.LEFT, padx=(10, 0))

        # === GSD Compression Settings ===
        gsd_frame = ttk.LabelFrame(main_frame, text="GSD Compression (Step 4, optional)", padding=8)
        gsd_frame.pack(fill=tk.X, pady=(0, 5))

        row_gsd = ttk.Frame(gsd_frame)
        row_gsd.pack(fill=tk.X, pady=2)
        self.gsd_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_gsd, text="Enable Byte-Shuffle + LZ4 compression", variable=self.gsd_enabled).pack(side=tk.LEFT)
        ttk.Label(row_gsd, text="(lossless, ~33% smaller, O(1) random access)", foreground="gray").pack(side=tk.LEFT, padx=(10, 0))

        # === Action Buttons ===
        sep = ttk.Separator(main_frame, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=8)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)

        self.full_pipeline_btn = ttk.Button(
            btn_frame,
            text="\u25b6 Run Full Pipeline",
            command=self._run_full_pipeline,
        )
        self.full_pipeline_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(
            btn_frame,
            text="\u25a0 Stop",
            command=self._stop,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        ttk.Button(btn_frame, text="Step 4 Only", command=self._run_step4).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Step 3 Only", command=self._run_step3).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Step 2 Only", command=self._run_step2).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Step 1 Only", command=self._run_step1).pack(side=tk.RIGHT, padx=5)

        # === Progress ===
        sep2 = ttk.Separator(main_frame, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, pady=8)

        self.progress_label = ttk.Label(main_frame, text="Ready")
        self.progress_label.pack(fill=tk.X)

        self.progress = ttk.Progressbar(main_frame, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(2, 5))

        # === Estimates (persistent, not in log) ===
        self.estimates_frame = ttk.LabelFrame(main_frame, text="Estimates", padding=6)
        self.estimates_frame.pack(fill=tk.X, pady=(0, 5))
        self.est_time_label = ttk.Label(self.estimates_frame, text="Time:  --", font=("Consolas", 9))
        self.est_time_label.pack(fill=tk.X)
        self.est_disk_label = ttk.Label(self.estimates_frame, text="Disk:  --", font=("Consolas", 9))
        self.est_disk_label.pack(fill=tk.X)

        # === Log Area (collapsible) ===
        log_header = ttk.Frame(main_frame)
        log_header.pack(fill=tk.X)
        self.log_toggle_btn = ttk.Button(log_header, text="\u25bc Log", width=10, command=self._toggle_log)
        self.log_toggle_btn.pack(side=tk.LEFT)

        self.log_frame = ttk.Frame(main_frame)
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, height=10, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # --- Log toggle ---

    def _toggle_log(self):
        if self._log_visible:
            self.log_frame.pack_forget()
            self.log_toggle_btn.config(text="\u25b6 Log")
            self._log_visible = False
        else:
            self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
            self.log_toggle_btn.config(text="\u25bc Log")
            self._log_visible = True

    # --- Auto-fill all paths from video ---

    def _auto_fill_from_video(self, video_path: str):
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self._video_base = os.path.join(video_dir, video_name)

        self.images_output.set(self._video_base + "_frames")
        self.ply_output.set(self._video_base + "_ply")
        self.seq_name.set(video_name)
        self._update_raw_path()

    def _update_raw_path(self):
        """Update RAW output path with pruning suffix and GSD path."""
        if self._video_base is None:
            return
        suffix = "_raw"
        if self.prune_enabled.get():
            try:
                pct = int(float(self.prune_keep_pct.get()))
                suffix = f"_raw_top{pct}"
            except ValueError:
                suffix = "_raw_pruned"
        self.raw_output.set(self._video_base + suffix)
        self.gsd_output.set(self._video_base + suffix + ".gsd")

    # --- Browse helpers ---

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)
            self._auto_fill_from_video(path)

    def _browse_images_output(self):
        path = filedialog.askdirectory(title="Select Images Output Folder")
        if path:
            self.images_output.set(path)

    def _browse_ply_output(self):
        path = filedialog.askdirectory(title="Select PLY Output Folder")
        if path:
            self.ply_output.set(path)

    def _browse_raw_output(self):
        path = filedialog.askdirectory(title="Select RAW Output Folder")
        if path:
            self.raw_output.set(path)

    def _browse_gsd_output(self):
        path = filedialog.asksaveasfilename(
            title="Save GSD File",
            defaultextension=".gsd",
            filetypes=[("GSD files", "*.gsd"), ("All files", "*.*")],
        )
        if path:
            self.gsd_output.set(path)

    # --- Logging ---

    def _log(self, message: str):
        self.root.after(0, self._append_log, message)

    def _append_log(self, message: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _set_progress(self, value: float, label: str = None):
        self.root.after(0, self._update_progress, value, label)

    def _update_progress(self, value: float, label: str = None):
        self.progress["value"] = value
        if label:
            self.progress_label.config(text=label)

    def _get_precision(self, var: tk.StringVar) -> int:
        return PRECISION_FULL if "32bit" in var.get() else PRECISION_HALF

    def _check_stop(self):
        if self._stop_flag:
            raise StopRequested("Stopped by user.")

    # --- Stop ---

    def _stop(self):
        self._stop_flag = True
        self._log("\nStopping... (will finish current operation)")
        # Kill subprocess if running
        if self._subprocess and self._subprocess.poll() is None:
            self._subprocess.terminate()

    # --- UI state ---

    def _set_running(self, running: bool):
        self._running = running
        self._stop_flag = False
        state = tk.DISABLED if running else tk.NORMAL
        stop_state = tk.NORMAL if running else tk.DISABLED
        self.root.after(0, lambda: self.stop_btn.config(state=stop_state))
        self.root.after(0, lambda: self.full_pipeline_btn.config(state=state))

    # --- Step runners ---

    def _format_elapsed(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"

    def _run_in_thread(self, func):
        if self._running:
            self._log("A task is already running. Please wait.")
            return
        self._set_running(True)
        thread = threading.Thread(target=self._thread_wrapper, args=(func,), daemon=True)
        thread.start()

    def _thread_wrapper(self, func):
        try:
            func()
        except StopRequested:
            self._log("\nStopped.")
            self._set_progress(0, "Stopped")
        except Exception as e:
            self._log(f"\nERROR: {e}")
            self._set_progress(0, "Error")
        finally:
            self._set_running(False)

    def _run_step1(self):
        self._run_in_thread(self._step1)

    def _step1(self):
        self._set_progress(0, "Step 1: Extracting frames...")
        self._log("=== Step 1: Video -> Images ===")
        t0 = time.time()

        video = self.video_path.get()
        output = self.images_output.get()
        count = int(self.frame_count.get())

        if not video:
            raise ValueError("Please select a video file.")
        if not output:
            raise ValueError("Please set an output folder.")

        # Show estimates
        est_space = count * 80 * 1024
        time_text = f"Time:  Step1 ~{self._format_elapsed(count * 0.1)}  ({count} frames)"
        disk_text = f"Disk:  Frames ~{self._format_size(est_space)}"
        self.root.after(0, self.est_time_label.config, {"text": time_text})
        self.root.after(0, self.est_disk_label.config, {"text": disk_text})

        # Resume: skip if frames already exist
        if os.path.isdir(output):
            existing = [f for f in os.listdir(output) if f.startswith("frame_") and f.endswith(".jpg")]
            if len(existing) >= count:
                self._log(f"Frames folder already has {len(existing)} frames, skipping. (delete folder to re-extract)")
                self._set_progress(100, "Step 1 skipped (frames exist)")
                return

            # Clean partial output
            for f in existing:
                os.remove(os.path.join(output, f))

        files = extract_frames(video, output, count, progress_callback=self._log)
        elapsed = time.time() - t0
        self._log(f"\nStep 1 complete: {len(files)} frames extracted. ({self._format_elapsed(elapsed)})")
        self._set_progress(100, f"Step 1 done ({self._format_elapsed(elapsed)})")

    def _run_step2(self):
        self._run_in_thread(self._step2)

    def _make_eta_callback(self, step_name: str):
        """Create a frame progress callback that shows ETA."""
        start_time = [None]

        def callback(current, total):
            self._check_stop()
            if start_time[0] is None:
                start_time[0] = time.time()
            pct = current / total * 100
            elapsed = time.time() - start_time[0]
            if current > 0:
                eta = elapsed / current * (total - current)
                eta_str = self._format_elapsed(eta)
                self._set_progress(pct, f"{step_name}: {current}/{total} (ETA: {eta_str})")
            else:
                self._set_progress(pct, f"{step_name}: {current}/{total}")

        return callback

    def _step2(self):
        self._check_stop()
        self._set_progress(0, "Step 2: Generating PLY (ml-sharp)...")
        self._log("=== Step 2: Images -> PLY ===")
        t0 = time.time()

        images = self.images_output.get()
        output = self.ply_output.get()
        device = self.device.get()

        if not images:
            raise ValueError("Please set the frames output folder (select a video first).")
        if not output:
            raise ValueError("Please set the PLY output folder.")

        # Show estimates
        if os.path.isdir(images):
            n_imgs = len([f for f in os.listdir(images)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))])
            est_time = n_imgs * (8 if device == "cuda" else 40)
            est_space = n_imgs * 64 * 1024 * 1024
            time_text = f"Time:  Step2 ~{self._format_elapsed(est_time)}  ({n_imgs} images, {device})"
            disk_text = f"Disk:  PLY ~{self._format_size(est_space)}"
            self.root.after(0, self.est_time_label.config, {"text": time_text})
            self.root.after(0, self.est_disk_label.config, {"text": disk_text})

        # Resume: skip if PLY files already exist matching frame count
        if os.path.isdir(output):
            existing_ply = [f for f in os.listdir(output) if f.lower().endswith(".ply")]
            expected_images = [f for f in os.listdir(images)
                               if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))]
            if len(existing_ply) >= len(expected_images):
                self._log(f"PLY folder already has {len(existing_ply)} files, skipping. (delete folder to regenerate)")
                self._set_progress(100, "Step 2 skipped (PLY exists)")
                return

        files = generate_ply(
            images, output, device,
            progress_callback=self._log,
            frame_progress_callback=self._make_eta_callback("Step 2"),
        )
        elapsed = time.time() - t0
        self._log(f"\nStep 2 complete: {len(files)} PLY files generated. ({self._format_elapsed(elapsed)})")
        self._set_progress(100, f"Step 2 done ({self._format_elapsed(elapsed)})")

    def _run_step3(self):
        self._run_in_thread(self._step3)

    def _step3(self):
        self._check_stop()
        self._set_progress(0, "Step 3: Converting PLY to RAW...")
        self._log("=== Step 3: PLY -> RAW ===")
        t0 = time.time()

        ply_dir = self.ply_output.get()
        output = self.raw_output.get()
        seq_name = self.seq_name.get() or "sequence"
        fps = float(self.target_fps.get())
        sh_deg = int(self.sh_degree.get())

        if not ply_dir:
            raise ValueError("Please set the PLY folder (select a video first).")
        if not output:
            raise ValueError("Please set the RAW output folder.")

        # Show estimates in persistent area
        if os.path.isdir(ply_dir):
            n_ply = len([f for f in os.listdir(ply_dir) if f.lower().endswith(".ply")])
            est_time = n_ply * 5
            est_space = n_ply * 146 * 1024 * 1024
            time_text = f"Time:  Step3 ~{self._format_elapsed(est_time)}  ({n_ply} PLY files)"
            disk_text = f"Disk:  RAW ~{self._format_size(est_space)}"
            self.root.after(0, self.est_time_label.config, {"text": time_text})
            self.root.after(0, self.est_disk_label.config, {"text": disk_text})

        frame_progress = self._make_eta_callback("Step 3")

        keep_ratio = None
        if self.prune_enabled.get():
            keep_ratio = float(self.prune_keep_pct.get()) / 100.0

        convert_ply_sequence(
            ply_folder=ply_dir,
            output_folder=output,
            sequence_name=seq_name,
            target_fps=fps,
            sh_degree=sh_deg,
            position_precision=self._get_precision(self.pos_prec),
            rotation_precision=self._get_precision(self.rot_prec),
            scale_opacity_precision=self._get_precision(self.scale_prec),
            sh_precision=self._get_precision(self.sh_prec),
            prune_keep_ratio=keep_ratio,
            progress_callback=self._log,
            frame_progress_callback=frame_progress,
        )

        elapsed = time.time() - t0
        self._log(f"\nStep 3 complete. ({self._format_elapsed(elapsed)})")
        self._set_progress(100, f"Step 3 done ({self._format_elapsed(elapsed)})")

    def _run_step4(self):
        self._run_in_thread(self._step4)

    def _step4(self):
        self._check_stop()
        self._set_progress(0, "Step 4: Compressing RAW to GSD...")
        self._log("=== Step 4: RAW -> GSD (Byte-Shuffle + LZ4) ===")
        t0 = time.time()

        raw_dir = self.raw_output.get()
        gsd_path = self.gsd_output.get()

        if not raw_dir:
            raise ValueError("Please set the RAW folder (select a video first).")
        if not gsd_path:
            raise ValueError("Please set the GSD output path.")

        # Show estimates
        if os.path.isdir(raw_dir):
            seq_path = os.path.join(raw_dir, "sequence.json")
            if os.path.isfile(seq_path):
                import json as _json
                with open(seq_path) as f:
                    _seq = _json.load(f)
                n_frames = _seq["frameCount"]
                time_text = f"Time:  Step4 ~{self._format_elapsed(n_frames * 0.5)}  ({n_frames} frames)"
                disk_text = f"Disk:  GSD ~estimated after completion"
                self.root.after(0, self.est_time_label.config, {"text": time_text})
                self.root.after(0, self.est_disk_label.config, {"text": disk_text})

        frame_progress = self._make_eta_callback("Step 4")

        stats = convert_raw_to_gsd(
            raw_folder=raw_dir,
            output_path=gsd_path,
            progress_callback=self._log,
            frame_progress_callback=frame_progress,
        )

        elapsed = time.time() - t0
        ratio = stats["overall_ratio"] * 100
        self._log(f"\nStep 4 complete. Compression ratio: {ratio:.1f}% ({self._format_elapsed(elapsed)})")
        self._set_progress(100, f"Step 4 done ({self._format_elapsed(elapsed)})")

    def _run_full_pipeline(self):
        self._run_in_thread(self._full_pipeline)

    def _format_size(self, bytes_val: float) -> str:
        if bytes_val < 1024:
            return f"{bytes_val:.0f} B"
        elif bytes_val < 1024 ** 2:
            return f"{bytes_val / 1024:.1f} KB"
        elif bytes_val < 1024 ** 3:
            return f"{bytes_val / 1024**2:.1f} MB"
        else:
            return f"{bytes_val / 1024**3:.2f} GB"

    def _show_estimates(self, n_frames: int):
        """Show estimated time and disk space in persistent labels."""
        device = self.device.get()

        # Time estimates (rough benchmarks)
        step1_time = n_frames * 0.1                        # ffmpeg: ~0.1s/frame
        step2_time = n_frames * (8 if device == "cuda" else 40)  # ml-sharp: ~8s GPU, ~40s CPU
        step3_time = n_frames * 5                           # PLY→RAW: ~5s/frame
        total_time = step1_time + step2_time + step3_time

        # Disk space estimates (based on actual output measurements)
        step1_space = n_frames * 80 * 1024                  # ~80KB per JPEG
        step2_space = n_frames * 64 * 1024 * 1024           # ~64MB per PLY (1.18M gaussians)
        # Step 3: pos=19MB + rot=9.1MB + scaleOp=9.1MB + 12*SH=109MB ≈ 146MB/frame
        step3_space = n_frames * 146 * 1024 * 1024
        total_space = step1_space + step2_space + step3_space

        # Step 4 estimates (GSD compression)
        step4_time = 0
        step4_space = 0
        step4_text = ""
        if self.gsd_enabled.get():
            step4_time = n_frames * 0.5                       # ~0.5s/frame for shuffle+LZ4
            step4_space = int(step3_space * 0.67)             # ~67% of RAW (shuffle+LZ4)
            total_time += step4_time
            total_space += step4_space
            step4_text = f"  |  Step4 ~{self._format_elapsed(step4_time)}"

        time_text = (
            f"Time:  Step1 ~{self._format_elapsed(step1_time)}  |  "
            f"Step2 ~{self._format_elapsed(step2_time)} ({device})  |  "
            f"Step3 ~{self._format_elapsed(step3_time)}{step4_text}  |  "
            f"Total ~{self._format_elapsed(total_time)}"
        )

        gsd_disk_text = f"  |  GSD ~{self._format_size(step4_space)}" if self.gsd_enabled.get() else ""
        disk_text = (
            f"Disk:  Frames ~{self._format_size(step1_space)}  |  "
            f"PLY ~{self._format_size(step2_space)}  |  "
            f"RAW ~{self._format_size(step3_space)}{gsd_disk_text}  |  "
            f"Total ~{self._format_size(total_space)}"
        )
        self.root.after(0, self.est_time_label.config, {"text": time_text})
        self.root.after(0, self.est_disk_label.config, {"text": disk_text})

    def _full_pipeline(self):
        self._log("========== FULL PIPELINE ==========\n")

        n_frames = int(self.frame_count.get())
        self._show_estimates(n_frames)

        t0 = time.time()
        self._step1()
        self._step2()
        self._step3()
        if self.gsd_enabled.get():
            self._step4()
        elapsed = time.time() - t0
        self._log(f"\n========== PIPELINE COMPLETE ({self._format_elapsed(elapsed)}) ==========")
        self._set_progress(100, f"Done ({self._format_elapsed(elapsed)})")


def main():
    root = tk.Tk()
    app = PipelineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
