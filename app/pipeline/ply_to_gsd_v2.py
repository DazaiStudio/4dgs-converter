"""PLY to GSD v2 (SHARP-VQ) converter.

SHARP-optimized encoding:
- Position: fp16 shuffle+LZ4 (same as V1 but 3-channel)
- Rotation: VQ K=256, per-frame codebook
- Scale: VQ K=256, per-frame codebook (pre-exp raw values)
- Opacity: uint8 (sigmoid-activated, [0,255])
- SH DC: VQ K=256, per-frame codebook

File structure: GSD1 magic + JSON header + sequential frame blobs.
Each frame blob: codebooks + LZ4-compressed sections.
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
from sklearn.cluster import MiniBatchKMeans

from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order

try:
    import lz4.block
except ImportError:
    print("ERROR: lz4 package required. Install with: pip install lz4")
    sys.exit(1)

GSD_MAGIC = b"GSD1"
VQ_K = 256


def _pixel_shuffle(data: bytes, bytes_per_pixel: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bytes_per_pixel).T.reshape(-1).tobytes()


def _vq_encode(data: np.ndarray, k: int = VQ_K, n_sample: int = 50000) -> tuple:
    """Vector quantize. Returns (codebook float32, indices uint8)."""
    n = len(data)
    sample_idx = np.random.choice(n, min(n_sample, n), replace=False)
    km = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, random_state=42)
    km.fit(data[sample_idx])
    indices = km.predict(data).astype(np.uint8)
    codebook = km.cluster_centers_.astype(np.float32)
    return codebook, indices


def _process_single_frame_v2(args: tuple) -> tuple:
    """Process a single PLY frame into a V2 compressed blob."""
    (frame_idx, ply_path, uniform_texture_size) = args

    gaussians = load_gaussian_ply(ply_path)
    n_gauss = len(gaussians["position"])
    sorted_indices, min_pos, max_pos = sort_3d_morton_order(gaussians["position"])

    texture_size = uniform_texture_size or math.ceil(math.sqrt(n_gauss))
    n_pixels = texture_size * texture_size

    idx = sorted_indices

    # Extract and prepare attributes
    pos = gaussians["position"][idx].astype(np.float32)
    rot = gaussians["rotation"][idx].astype(np.float32)
    scale_raw = gaussians["scale"][idx].astype(np.float32)
    opacity_raw = gaussians["opacity"][idx].astype(np.float32)
    sh_dc = gaussians["sh_dc"][idx].astype(np.float32)

    # Normalize rotation
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot_norm = np.where(rot_norm == 0, 1.0, rot_norm)
    rot = rot / rot_norm

    # Activate opacity
    opacity_act = (1.0 / (1.0 + np.exp(-opacity_raw))).astype(np.float32)

    # --- Pad to texture size ---
    def pad_2d(arr, target_n):
        if len(arr) >= target_n:
            return arr[:target_n]
        pad = np.zeros((target_n - len(arr), arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad])

    def pad_1d(arr, target_n):
        if len(arr) >= target_n:
            return arr[:target_n]
        pad = np.zeros(target_n - len(arr), dtype=arr.dtype)
        return np.concatenate([arr, pad])

    pos = pad_2d(pos, n_pixels)        # (P, 3)
    rot = pad_2d(rot, n_pixels)        # (P, 4)
    scale_raw = pad_2d(scale_raw, n_pixels)  # (P, 3)
    opacity_act = pad_1d(opacity_act, n_pixels)  # (P,)
    sh_dc = pad_2d(sh_dc, n_pixels)    # (P, 3)

    # Pad rotation: unused pixels get identity quat (0,0,0,1) in raw form
    if n_gauss < n_pixels:
        rot[n_gauss:] = [0, 0, 0, 1]

    # --- Encode ---

    # Position: fp16 × 4 (RGBA, A=0) for compatibility with V1 texture format
    # Pack as (Z, X, -Y, 0) matching V1 layout
    pos_rgba = np.zeros((n_pixels, 4), dtype=np.float16)
    pos_rgba[:, 0] = pos[:, 2].astype(np.float16)   # R = Z
    pos_rgba[:, 1] = pos[:, 0].astype(np.float16)   # G = X
    pos_rgba[:, 2] = (-pos[:, 1]).astype(np.float16) # B = -Y
    pos_fp16 = pos_rgba.tobytes()
    pos_shuffled = _pixel_shuffle(pos_fp16, 8)  # 4 × fp16 = 8 bpp
    pos_compressed = lz4.block.compress(pos_shuffled, store_size=False)

    # Rotation: VQ K=256
    rot_codebook, rot_indices = _vq_encode(rot, VQ_K)
    rot_compressed = lz4.block.compress(rot_indices.tobytes(), store_size=False)

    # Scale: VQ K=256
    sc_codebook, sc_indices = _vq_encode(scale_raw, VQ_K)
    sc_compressed = lz4.block.compress(sc_indices.tobytes(), store_size=False)

    # Opacity: uint8
    op_uint8 = (np.clip(opacity_act, 0, 1) * 255).astype(np.uint8)
    op_compressed = lz4.block.compress(op_uint8.tobytes(), store_size=False)

    # SH DC: VQ K=256
    sh_codebook, sh_indices = _vq_encode(sh_dc, VQ_K)
    sh_compressed = lz4.block.compress(sh_indices.tobytes(), store_size=False)

    # --- Pack blob ---
    # Codebooks (raw, not compressed)
    rot_cb_bytes = rot_codebook.tobytes()   # K × 4 × 4 bytes
    sc_cb_bytes = sc_codebook.tobytes()     # K × 3 × 4 bytes
    sh_cb_bytes = sh_codebook.tobytes()     # K × 3 × 4 bytes

    section_header = struct.pack("<IIIIIIII",
        len(rot_cb_bytes),
        len(sc_cb_bytes),
        len(sh_cb_bytes),
        len(pos_compressed),
        len(rot_compressed),
        len(sc_compressed),
        len(op_compressed),
        len(sh_compressed),
    )

    blob = b"".join([
        section_header,
        rot_cb_bytes,
        sc_cb_bytes,
        sh_cb_bytes,
        pos_compressed,
        rot_compressed,
        sc_compressed,
        op_compressed,
        sh_compressed,
    ])

    # Raw size for comparison (V1-equivalent)
    raw_size = n_pixels * (8 + 8 + 8 + 8)  # pos fp16×4 + rot fp16×4 + so fp16×4 + sh fp16×4

    frame_info = {
        "compressedSize": len(blob),
        "textureWidth": texture_size,
        "textureHeight": texture_size,
        "gaussianCount": n_gauss,
        "minPosition": {"x": float(min_pos[0]), "y": float(min_pos[1]), "z": float(min_pos[2])},
        "maxPosition": {"x": float(max_pos[0]), "y": float(max_pos[1]), "z": float(max_pos[2])},
    }

    return frame_idx, blob, frame_info, raw_size


def convert_ply_to_gsd_v2(
    ply_folder: str,
    output_path: str,
    sequence_name: str,
    target_fps: float = 30.0,
    max_workers: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_step: int = 1,
    progress_callback: Optional[Callable[[str], None]] = None,
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Convert PLY folder to GSD v2 (SHARP-VQ format)."""
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # Find PLY files
    def _natural_sort_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    ply_files = sorted([
        f for f in os.listdir(ply_folder) if f.lower().endswith(".ply")
    ], key=_natural_sort_key)

    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {ply_folder}")

    total_available = len(ply_files)
    if end_frame is None:
        end_frame = total_available - 1
    end_frame = min(end_frame, total_available - 1)
    start_frame = max(0, min(start_frame, end_frame))
    frame_step = max(1, frame_step)
    ply_files = ply_files[start_frame:end_frame + 1:frame_step]
    frame_count = len(ply_files)

    log(f"PLY -> GSD v2 (SHARP-VQ): {frame_count} frames, target {target_fps} fps")

    # Workers
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, cpu_count // 2)
    log(f"Using {max_workers} workers")

    # Pre-scan for uniform texture size
    log("Pre-scanning for max gaussian count...")
    max_gaussian_count = 0
    for i, ply_file in enumerate(ply_files):
        g = load_gaussian_ply(os.path.join(ply_folder, ply_file))
        n = len(g["position"])
        max_gaussian_count = max(max_gaussian_count, n)
        if frame_progress_callback and (i + 1) % 20 == 0:
            frame_progress_callback(i + 1, frame_count)

    uniform_texture_size = math.ceil(math.sqrt(max_gaussian_count))
    log(f"Uniform texture: {uniform_texture_size}x{uniform_texture_size} ({max_gaussian_count:,} gaussians)")

    # Build tasks
    task_args = [
        (i, os.path.join(ply_folder, ply_files[i]), uniform_texture_size)
        for i in range(frame_count)
    ]

    # Process frames
    t0 = time.time()
    results = [None] * frame_count
    completed = 0
    total_raw = 0
    total_compressed = 0

    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(_process_single_frame_v2, args): args[0]
            for args in task_args
        }
        for future in as_completed(futures):
            frame_idx, blob, frame_info, raw_size = future.result()
            results[frame_idx] = (blob, frame_info, raw_size)
            completed += 1
            total_raw += raw_size
            total_compressed += len(blob)

            if frame_progress_callback:
                frame_progress_callback(completed, frame_count)
            if completed % 50 == 0 or completed == frame_count:
                ratio = total_compressed / total_raw * 100
                log(f"  Encoded {completed}/{frame_count} ({ratio:.1f}%)")
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    encode_time = time.time() - t0

    frame_blobs = [r[0] for r in results]
    frame_infos = [r[1] for r in results]

    # Build header
    header = {
        "version": 2,
        "compression": "sharp_vq",
        "sequenceName": sequence_name,
        "frameCount": frame_count,
        "targetFPS": target_fps,
        "shDegree": 0,
        "textureWidth": uniform_texture_size,
        "textureHeight": uniform_texture_size,
        "gaussianCount": max_gaussian_count,
        "positionEncoding": "fp16_shuffle_lz4",
        "rotationEncoding": f"vq{VQ_K}",
        "scaleEncoding": f"vq{VQ_K}",
        "opacityEncoding": "uint8",
        "shEncoding": f"vq{VQ_K}",
        "vqK": VQ_K,
        "frames": frame_infos,
    }

    # Write file
    log(f"Writing {output_path}...")
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(GSD_MAGIC)
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        for blob in frame_blobs:
            f.write(struct.pack("<I", len(blob)))
            f.write(blob)

    file_size = os.path.getsize(output_path)

    stats = {
        "frame_count": frame_count,
        "total_raw_size": total_raw,
        "total_compressed_size": total_compressed,
        "file_size": file_size,
        "overall_ratio": total_compressed / total_raw,
        "encode_time": encode_time,
    }

    log(f"\n{'='*60}")
    log(f"PLY -> GSD v2 Stats (SHARP-VQ)")
    log(f"{'='*60}")
    log(f"  Frames:        {frame_count}")
    log(f"  Gaussians:     {max_gaussian_count:,}")
    log(f"  V1-equiv raw:  {total_raw / 1e9:.2f} GB")
    log(f"  V2 GSD:        {file_size / 1e9:.2f} GB")
    log(f"  Ratio:         {stats['overall_ratio']*100:.1f}%")
    log(f"  Per-frame avg: {total_compressed / frame_count / 1e6:.2f} MB")
    log(f"  Encode time:   {encode_time:.1f}s ({encode_time/frame_count*1000:.0f}ms/frame)")
    log(f"  Output:        {output_path}")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PLY to GSD v2 (SHARP-VQ)")
    parser.add_argument("ply_folder", help="Folder with PLY files")
    parser.add_argument("output", help="Output .gsd file path")
    parser.add_argument("--name", default="sequence", help="Sequence name")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--vq-k", type=int, default=None, help="VQ codebook size (default: 4096)")
    args = parser.parse_args()

    if args.vq_k is not None:
        import app.pipeline.ply_to_gsd_v2 as _self
        _self.VQ_K = args.vq_k

    convert_ply_to_gsd_v2(
        args.ply_folder, args.output, args.name,
        target_fps=args.fps, max_workers=args.workers,
        start_frame=args.start, end_frame=args.end, frame_step=args.step,
    )
