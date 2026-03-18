"""Entry point for python -m app.converter.

Usage:
    GUI:  python -m app.converter
    CLI:  python -m app.converter --cli -i INPUT -o OUTPUT [options]
"""

import argparse
import sys


def _find_ico():
    """Find icon.ico in PyInstaller bundle or source tree."""
    import os
    for d in [
        os.path.join(getattr(sys, '_MEIPASS', ''), "app", "converter"),
        os.path.dirname(os.path.abspath(__file__)),
    ]:
        p = os.path.join(d, "icon.ico")
        if os.path.exists(p):
            return p
    return None


def _set_native_icon(hwnd, ico_path):
    """Set native Win32 icon via WM_SETICON + LoadImageW."""
    import ctypes
    user32 = ctypes.windll.user32
    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x00000010
    LR_DEFAULTSIZE = 0x00000040
    WM_SETICON = 0x0080
    ICON_BIG = 1
    ICON_SMALL = 0

    hicon_big = user32.LoadImageW(
        None, ico_path, IMAGE_ICON, 48, 48,
        LR_LOADFROMFILE,
    )
    hicon_small = user32.LoadImageW(
        None, ico_path, IMAGE_ICON, 16, 16,
        LR_LOADFROMFILE,
    )
    if hicon_big:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
    if hicon_small:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)


def main_gui():
    import os

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from app.converter.main_window import MainWindow

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # Force taskbar icon via native Win32 API (bypasses Qt)
    if sys.platform == "win32":
        hwnd = int(window.winId())
        ico_path = _find_ico()
        if ico_path:
            _set_native_icon(hwnd, ico_path)

    sys.exit(app.exec())


def main_cli(args):
    import os
    import time

    input_path = args.input
    output_path = args.output
    mode = args.mode
    fps = args.fps
    start_frame = args.start
    end_frame = args.end

    if not input_path:
        print("Error: --input is required")
        sys.exit(1)

    # Auto-detect mode
    if mode == "auto":
        if os.path.isdir(input_path):
            mode = "ply"
        else:
            mode = "video"

    # Auto-derive output
    if not output_path:
        if mode == "video":
            name = os.path.splitext(os.path.basename(input_path))[0]
            parent = os.path.dirname(input_path)
            output_path = os.path.join(parent, name, f"{name}.gsd")
        else:
            folder_name = os.path.basename(input_path.rstrip("/\\"))
            parent = os.path.dirname(input_path.rstrip("/\\"))
            output_path = os.path.join(parent, f"{folder_name}.gsd")

    print(f"Mode: {mode}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    if start_frame > 0 or end_frame is not None:
        print(f"Frame range: {start_frame}-{end_frame if end_frame is not None else 'end'}")
    print()

    if mode == "video":
        _run_video_cli(input_path, output_path, fps, start_frame, end_frame, args)
    else:
        _run_ply_cli(input_path, output_path, fps, start_frame, end_frame)


def _run_video_cli(input_path, output_path, fps, start_frame, end_frame, args):
    import os

    parent = os.path.dirname(output_path)
    images_folder = os.path.join(parent, "images")
    ply_folder = os.path.join(parent, "ply")

    # Step 1: Extract frames
    print("Step 1/3: Extracting frames...")
    from app.pipeline.video_to_images import extract_frames, get_video_frame_count, get_video_fps

    detected_fps = get_video_fps(input_path)
    if detected_fps > 0:
        fps = detected_fps
        print(f"  Detected FPS: {fps:.2f}")

    total_frames = get_video_frame_count(input_path)
    os.makedirs(images_folder, exist_ok=True)

    # Calculate expected frame count
    ef = end_frame if end_frame is not None and end_frame >= 0 else total_frames - 1
    expected = ef - start_frame + 1

    existing = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
    if len(existing) >= expected:
        print(f"  Frames already extracted ({len(existing)} files), skipping.")
    else:
        extract_frames(
            video_path=input_path,
            output_folder=images_folder,
            frame_count=total_frames,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame if end_frame is not None else -1,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

    # Step 2: Generate PLY
    print("Step 2/3: Generating PLY (SHARP)...")
    from app.pipeline.images_to_ply import generate_ply

    os.makedirs(ply_folder, exist_ok=True)
    ply_count = len([f for f in os.listdir(ply_folder) if f.endswith(".ply")])
    image_count = len([f for f in os.listdir(images_folder) if f.endswith(".jpg")])

    if ply_count >= image_count and ply_count > 0:
        print(f"  PLY files already exist ({ply_count} files), skipping.")
    else:
        generate_ply(images_folder, ply_folder, progress_callback=lambda msg: print(f"  {msg}"))

    # Step 3: Convert to GSD
    if not args.skip_gsd:
        print("Step 3/3: Converting to GSD...")
        _convert_gsd(ply_folder, output_path, fps, start_frame, end_frame)

    # Cleanup
    if not args.keep_images and os.path.isdir(images_folder):
        import shutil
        print(f"Cleaning up images: {images_folder}")
        shutil.rmtree(images_folder, ignore_errors=True)
    if not args.keep_ply and os.path.isdir(ply_folder):
        import shutil
        print(f"Cleaning up PLY: {ply_folder}")
        shutil.rmtree(ply_folder, ignore_errors=True)

    print("Done!")


def _run_ply_cli(input_path, output_path, fps, start_frame, end_frame):
    print("Converting PLY to GSD...")
    _convert_gsd(input_path, output_path, fps, start_frame, end_frame)
    print("Done!")


def _convert_gsd(ply_folder, output_path, fps, start_frame, end_frame):
    import os
    from app.pipeline.ply_to_gsd import convert_ply_to_gsd

    sequence_name = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stats = convert_ply_to_gsd(
        ply_folder=ply_folder,
        output_path=output_path,
        sequence_name=sequence_name,
        target_fps=fps,
        start_frame=start_frame,
        end_frame=end_frame if end_frame is not None and end_frame >= 0 else None,
        progress_callback=lambda msg: print(f"  {msg}"),
    )
    return stats


def main():
    # Must be before QApplication for Windows taskbar grouping
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "dazaistudio.4dgs-converter"
        )

    if "--cli" in sys.argv:
        parser = argparse.ArgumentParser(
            prog="python -m app.converter",
            description="4DGS Converter - Convert video/PLY to GSD format",
        )
        parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no GUI)")
        parser.add_argument("-i", "--input", required=True, help="Input video file or PLY folder")
        parser.add_argument("-o", "--output", help="Output .gsd path (auto-derived if omitted)")
        parser.add_argument("--mode", choices=["auto", "video", "ply"], default="auto",
                            help="Conversion mode (default: auto-detect from input)")
        parser.add_argument("--fps", type=float, default=30.0, help="Target FPS (default: 30)")
        parser.add_argument("--start", type=int, default=0, help="Start frame (0-based, default: 0)")
        parser.add_argument("--end", type=int, default=None, help="End frame (0-based, default: last)")
        parser.add_argument("--keep-images", action="store_true", help="Keep extracted images (video mode)")
        parser.add_argument("--keep-ply", action="store_true", help="Keep PLY files (video mode)")
        parser.add_argument("--skip-gsd", action="store_true", help="Skip GSD, stop after PLY (video mode)")

        args = parser.parse_args()
        main_cli(args)
    else:
        main_gui()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
