"""PLY to GSD direct converter.

Converts a folder of PLY files directly to a single .gsd file,
skipping the intermediate RAW step. Supports optional pruning.

Pipeline: PLY -> load -> prune -> morton sort -> pack textures -> shuffle -> LZ4 -> GSD
"""

import json
import math
import os
import re
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np

from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order
from app.pipeline.ply_to_raw import (
    PRECISION_FULL,
    PRECISION_HALF,
    SH_DEGREE_TO_TEXTURES,
    _pack_textures,
    _prune_by_contribution,
)

try:
    import lz4.block
except ImportError:
    print("ERROR: lz4 package required. Install with: pip install lz4")
    sys.exit(1)

GSD_MAGIC = b"GSD1"


def _get_bpp(precision: int) -> int:
    return 16 if precision == PRECISION_FULL else 8


def _pixel_shuffle(data: bytes, bytes_per_pixel: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bytes_per_pixel).T.reshape(-1).tobytes()


def _textures_to_shuffled_blob(
    textures: list[np.ndarray],
    precisions: list[int],
) -> bytes:
    """Convert packed textures to a shuffled+concatenated blob."""
    parts = []
    for tex, prec in zip(textures, precisions):
        if prec == PRECISION_FULL:
            raw = tex.astype(np.float32).tobytes()
        else:
            raw = tex.astype(np.float16).tobytes()
        bpp = _get_bpp(prec)
        parts.append(_pixel_shuffle(raw, bpp))
    return b"".join(parts)


def _process_single_frame(args: tuple) -> tuple[int, bytes, dict, int]:
    """Process a single PLY frame into a compressed blob. Designed for multiprocessing."""
    (
        frame_idx, ply_path, precisions, num_sh,
        prune_keep_ratio, uniform_texture_size,
    ) = args

    gaussians = load_gaussian_ply(ply_path)

    if prune_keep_ratio is not None and prune_keep_ratio < 1.0:
        gaussians = _prune_by_contribution(gaussians, prune_keep_ratio)
    gaussian_count = len(gaussians["position"])

    sorted_indices, min_pos, max_pos = sort_3d_morton_order(gaussians["position"])

    texture_size = uniform_texture_size if uniform_texture_size else math.ceil(math.sqrt(gaussian_count))

    all_textures = _pack_textures(gaussians, sorted_indices, texture_size)
    textures = all_textures[:3 + num_sh]

    blob = _textures_to_shuffled_blob(textures, precisions)
    raw_size = len(blob)

    compressed = lz4.block.compress(blob, store_size=False)

    frame_info = {
        "compressedSize": len(compressed),
        "textureWidth": texture_size,
        "textureHeight": texture_size,
        "gaussianCount": gaussian_count,
        "minPosition": {
            "x": float(min_pos[0]),
            "y": float(min_pos[1]),
            "z": float(min_pos[2]),
        },
        "maxPosition": {
            "x": float(max_pos[0]),
            "y": float(max_pos[1]),
            "z": float(max_pos[2]),
        },
    }

    return frame_idx, compressed, frame_info, raw_size


def convert_ply_to_gsd(
    ply_folder: str,
    output_path: str,
    sequence_name: str,
    target_fps: float = 30.0,
    sh_degree: int = 0,
    position_precision: int = PRECISION_FULL,
    rotation_precision: int = PRECISION_HALF,
    scale_opacity_precision: int = PRECISION_HALF,
    sh_precision: int = PRECISION_HALF,
    prune_keep_ratio: Optional[float] = None,
    max_workers: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Convert a folder of PLY files directly to a .gsd file.

    Args:
        ply_folder: Folder containing .ply files.
        output_path: Output .gsd file path.
        sequence_name: Name for the sequence.
        target_fps: Target playback FPS.
        sh_degree: SH degree (0, 1, 2, or 3).
        position_precision: PRECISION_FULL or PRECISION_HALF.
        rotation_precision: PRECISION_FULL or PRECISION_HALF.
        scale_opacity_precision: PRECISION_FULL or PRECISION_HALF.
        sh_precision: PRECISION_FULL or PRECISION_HALF.
        prune_keep_ratio: If set, keep top N% by contribution (e.g. 0.5 for 50%).
        max_workers: Max parallel workers (default: half of CPU cores).
        progress_callback: Callback for log messages.
        frame_progress_callback: Callback (current_frame, total_frames).

    Returns:
        Stats dict with compression info.
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Find PLY files (natural numeric sort)
    def _natural_sort_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    ply_files = sorted([
        f for f in os.listdir(ply_folder)
        if f.lower().endswith(".ply")
    ], key=_natural_sort_key)

    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {ply_folder}")

    # Apply frame range
    total_available = len(ply_files)
    if end_frame is None:
        end_frame = total_available - 1
    end_frame = min(end_frame, total_available - 1)
    start_frame = max(0, min(start_frame, end_frame))
    ply_files = ply_files[start_frame:end_frame + 1]

    frame_count = len(ply_files)
    num_sh = SH_DEGREE_TO_TEXTURES.get(sh_degree, 12)

    prune_str = f", pruning to {prune_keep_ratio*100:.0f}%" if prune_keep_ratio else ""
    range_str = f" (frames {start_frame}-{end_frame} of {total_available})" if start_frame > 0 or end_frame < total_available - 1 else ""
    log(f"PLY -> GSD: {frame_count} frames{range_str}, SH degree {sh_degree} ({num_sh} SH textures){prune_str}")
    log(f"Target FPS: {target_fps} (make sure this matches your source video framerate)")

    # Precision list for all textures
    precisions = [
        position_precision,
        rotation_precision,
        scale_opacity_precision,
    ] + [sh_precision] * num_sh

    # Determine worker count
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, cpu_count // 2)
    log(f"Using {max_workers} workers")

    # Pre-scan: find max gaussian count to compute uniform texture size
    log("Pre-scanning PLY files for max gaussian count...")
    max_gaussian_count = 0
    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(ply_folder, ply_file)
        g = load_gaussian_ply(ply_path)
        n = len(g["position"])
        if prune_keep_ratio is not None and prune_keep_ratio < 1.0:
            n = int(n * prune_keep_ratio)
        if n > max_gaussian_count:
            max_gaussian_count = n
        if frame_progress_callback:
            frame_progress_callback(i + 1, frame_count)
    uniform_texture_size = math.ceil(math.sqrt(max_gaussian_count))
    log(f"Uniform texture size: {uniform_texture_size}x{uniform_texture_size} (max {max_gaussian_count:,} gaussians)")

    # Build task args
    task_args = []
    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(ply_folder, ply_file)
        task_args.append((
            i, ply_path, precisions, num_sh,
            prune_keep_ratio, uniform_texture_size,
        ))

    # Process frames in parallel
    t0 = time.time()
    results = [None] * frame_count  # preserve order
    completed_count = 0
    total_raw_size = 0
    total_compressed_size = 0

    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(_process_single_frame, args): args[0]
            for args in task_args
        }

        for future in as_completed(futures):
            frame_idx, compressed, frame_info, raw_size = future.result()
            results[frame_idx] = (compressed, frame_info, raw_size)

            completed_count += 1
            total_raw_size += raw_size
            total_compressed_size += len(compressed)

            if frame_progress_callback:
                frame_progress_callback(completed_count, frame_count)

            if completed_count % 50 == 0 or completed_count == frame_count:
                ratio = total_compressed_size / total_raw_size * 100
                log(f"  Encoded {completed_count}/{frame_count} frames ({ratio:.1f}%)")
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    encode_time = time.time() - t0

    # Unpack results in order
    frame_blobs = [r[0] for r in results]
    frame_infos = [r[1] for r in results]

    expected_blob_size = results[0][2]  # raw_size of first frame

    log(f"Uncompressed frame size: {expected_blob_size / 1e6:.1f} MB")
    log(f"Texture: {uniform_texture_size}x{uniform_texture_size}, max {max_gaussian_count:,} gaussians")

    # Build header — use uniform texture size and max gaussian count
    # so UE renderer allocates correct buffers and renders all points
    header = {
        "version": 1,
        "compression": "shuffle_lz4",
        "sequenceName": sequence_name,
        "frameCount": frame_count,
        "targetFPS": target_fps,
        "shDegree": sh_degree,
        "textureWidth": uniform_texture_size,
        "textureHeight": uniform_texture_size,
        "gaussianCount": max_gaussian_count,
        "positionPrecision": position_precision,
        "rotationPrecision": rotation_precision,
        "scaleOpacityPrecision": scale_opacity_precision,
        "shPrecision": sh_precision,
        "frames": frame_infos,
    }

    if prune_keep_ratio is not None:
        header["pruneKeepRatio"] = prune_keep_ratio

    # Write .gsd file
    log(f"Writing {output_path}...")
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(GSD_MAGIC)
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        for compressed in frame_blobs:
            f.write(struct.pack("<I", len(compressed)))
            f.write(compressed)

    file_size = os.path.getsize(output_path)

    stats = {
        "frame_count": frame_count,
        "uncompressed_frame_size": expected_blob_size,
        "total_raw_size": total_raw_size,
        "total_compressed_size": total_compressed_size,
        "file_size": file_size,
        "overall_ratio": total_compressed_size / total_raw_size,
        "encode_time": encode_time,
    }

    log(f"\n{'='*60}")
    log(f"PLY -> GSD Stats (Byte-Shuffle + LZ4)")
    log(f"{'='*60}")
    log(f"  Frames:          {frame_count}")
    log(f"  Gaussians:       {frame_infos[0]['gaussianCount']:,}" +
        (f" (pruned to {prune_keep_ratio*100:.0f}%)" if prune_keep_ratio else ""))
    log(f"  Raw frame size:  {expected_blob_size / 1e6:.1f} MB")
    log(f"  Total RAW:       {total_raw_size / 1e9:.2f} GB")
    log(f"  Total GSD:       {file_size / 1e9:.2f} GB")
    log(f"  Overall ratio:   {stats['overall_ratio']*100:.1f}%")
    log(f"  Encode time:     {encode_time:.1f}s ({encode_time / frame_count * 1000:.0f}ms/frame)")
    log(f"  Workers:         {max_workers}")
    log(f"  Output:          {output_path}")

    return stats
