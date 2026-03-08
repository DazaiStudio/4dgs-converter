"""PLY to GSD direct converter.

Converts a folder of PLY files directly to a single .gsd file,
skipping the intermediate RAW step. Supports optional pruning.

Pipeline: PLY -> load -> prune -> morton sort -> pack textures -> shuffle -> LZ4 -> GSD
"""

import json
import math
import os
import struct
import sys
import time
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


def convert_ply_to_gsd(
    ply_folder: str,
    output_path: str,
    sequence_name: str,
    target_fps: float = 24.0,
    sh_degree: int = 0,
    position_precision: int = PRECISION_FULL,
    rotation_precision: int = PRECISION_HALF,
    scale_opacity_precision: int = PRECISION_HALF,
    sh_precision: int = PRECISION_HALF,
    prune_keep_ratio: Optional[float] = None,
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
        progress_callback: Callback for log messages.
        frame_progress_callback: Callback (current_frame, total_frames).

    Returns:
        Stats dict with compression info.
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Find PLY files
    ply_files = sorted([
        f for f in os.listdir(ply_folder)
        if f.lower().endswith(".ply")
    ])

    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {ply_folder}")

    frame_count = len(ply_files)
    num_sh = SH_DEGREE_TO_TEXTURES.get(sh_degree, 12)

    prune_str = f", pruning to {prune_keep_ratio*100:.0f}%" if prune_keep_ratio else ""
    log(f"PLY -> GSD: {frame_count} frames, SH degree {sh_degree} ({num_sh} SH textures){prune_str}")

    # Precision list for all textures
    precisions = [
        position_precision,
        rotation_precision,
        scale_opacity_precision,
    ] + [sh_precision] * num_sh

    # Process frames into compressed blobs
    t0 = time.time()
    frame_blobs = []
    frame_infos = []
    total_raw_size = 0
    total_compressed_size = 0
    expected_blob_size = None

    for i, ply_file in enumerate(ply_files):
        ply_path = os.path.join(ply_folder, ply_file)

        # Load PLY
        gaussians = load_gaussian_ply(ply_path)
        original_count = len(gaussians["position"])

        # Prune
        if prune_keep_ratio is not None and prune_keep_ratio < 1.0:
            gaussians = _prune_by_contribution(gaussians, prune_keep_ratio)
        gaussian_count = len(gaussians["position"])

        # Morton sort
        sorted_indices, min_pos, max_pos = sort_3d_morton_order(gaussians["position"])

        # Texture size
        texture_size = math.ceil(math.sqrt(gaussian_count))

        # Pack textures (returns all 15, we only use the ones we need)
        all_textures = _pack_textures(gaussians, sorted_indices, texture_size)
        textures = all_textures[:3 + num_sh]

        # Convert to shuffled blob
        blob = _textures_to_shuffled_blob(textures, precisions)

        if expected_blob_size is None:
            expected_blob_size = len(blob)
            log(f"Uncompressed frame size: {expected_blob_size / 1e6:.1f} MB")
            log(f"Texture: {texture_size}x{texture_size}, {gaussian_count} gaussians")

        # LZ4 compress
        compressed = lz4.block.compress(blob, store_size=False)

        frame_blobs.append(compressed)
        frame_infos.append({
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
        })

        total_raw_size += len(blob)
        total_compressed_size += len(compressed)

        if frame_progress_callback:
            frame_progress_callback(i + 1, frame_count)

        if (i + 1) % 50 == 0 or i == frame_count - 1:
            ratio = total_compressed_size / total_raw_size * 100
            log(f"  Encoded {i + 1}/{frame_count} frames ({ratio:.1f}%)")

    encode_time = time.time() - t0

    # Build header
    header = {
        "version": 1,
        "compression": "shuffle_lz4",
        "sequenceName": sequence_name,
        "frameCount": frame_count,
        "targetFPS": target_fps,
        "shDegree": sh_degree,
        "textureWidth": frame_infos[0]["textureWidth"],
        "textureHeight": frame_infos[0]["textureHeight"],
        "gaussianCount": frame_infos[0]["gaussianCount"],
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
    log(f"  Output:          {output_path}")

    return stats
