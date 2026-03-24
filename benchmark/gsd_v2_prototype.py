"""GSD v2 SHARP-optimized encoding prototype.

Tests the full encode/decode roundtrip:
- Rotation: VQ K=256 (8-bit codebook index)
- Scale: 1-channel base (codebook) + 2 delta (int8)
- Opacity: 1-bit flag + sparse fp16 exceptions
- SH DC: VQ K=256 (8-bit codebook index)
- Position: fp16 (unchanged)

Measures: compression ratio, decode error, encode/decode speed.
"""

import json
import math
import os
import struct
import sys
import time

import lz4.block
import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order
from app.pipeline.ply_to_raw import _pack_textures, PRECISION_HALF, PRECISION_FULL


def pixel_shuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bpp).T.reshape(-1).tobytes()


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def decompress_lz4(data: bytes, size: int) -> bytes:
    return lz4.block.decompress(data, uncompressed_size=size)


# ---------------------------------------------------------------------------
# VQ Encoder/Decoder
# ---------------------------------------------------------------------------

def vq_encode(data: np.ndarray, k: int = 256, n_sample: int = 50000) -> tuple:
    """Vector quantize data. Returns (codebook, indices)."""
    n = len(data)
    # Subsample for fitting
    sample_idx = np.random.choice(n, min(n_sample, n), replace=False)
    km = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, random_state=42)
    km.fit(data[sample_idx])
    indices = km.predict(data).astype(np.uint8)
    codebook = km.cluster_centers_.astype(np.float32)
    return codebook, indices


def vq_decode(codebook: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Reconstruct from codebook + indices."""
    return codebook[indices]


# ---------------------------------------------------------------------------
# Scale Encoder/Decoder (VQ on full 3-axis)
# ---------------------------------------------------------------------------

def scale_encode(scale_raw: np.ndarray, k: int = 256) -> tuple:
    """Encode scale as VQ on all 3 axes together.

    scale_raw: (N, 3) pre-exp scale values.
    Returns: (codebook, indices)
    """
    codebook, indices = vq_encode(scale_raw, k=k)
    return codebook, indices


def scale_decode(codebook, indices):
    """Reconstruct scale from VQ."""
    return vq_decode(codebook, indices)


# ---------------------------------------------------------------------------
# Opacity Encoder/Decoder (uint8 quantization)
# ---------------------------------------------------------------------------

def opacity_encode(opacity_activated: np.ndarray) -> tuple:
    """Encode opacity as uint8 [0, 255].

    opacity_activated: (N,) sigmoid-activated values in [0, 1].
    Returns: (quantized_uint8,)
    """
    q = (np.clip(opacity_activated, 0, 1) * 255).astype(np.uint8)
    return (q,)


def opacity_decode(q_uint8, n_total):
    """Reconstruct opacity from uint8."""
    return q_uint8.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Full frame encode/decode
# ---------------------------------------------------------------------------

def encode_frame_v2(gaussians: dict, sorted_indices: np.ndarray,
                    texture_size: int) -> tuple:
    """Encode a single frame with SHARP-optimized encoding.

    Returns (compressed_blob, metadata, original_for_comparison).
    """
    n_gauss = len(sorted_indices)
    idx = sorted_indices

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
    opacity_act = 1.0 / (1.0 + np.exp(-opacity_raw))

    # --- Encode each attribute ---

    # Position: fp16 (unchanged from v1)
    pos_fp16 = pos.astype(np.float16).tobytes()
    pos_shuffled = pixel_shuffle(pos_fp16, 6)  # 3 × fp16 = 6 bpp
    pos_compressed = compress_lz4(pos_shuffled)

    # Rotation: VQ K=256
    rot_codebook, rot_indices = vq_encode(rot, k=256)
    rot_idx_compressed = compress_lz4(rot_indices.tobytes())

    # Scale: VQ K=256 on full 3-axis
    sc_codebook, sc_indices = scale_encode(scale_raw, k=256)
    sc_idx_compressed = compress_lz4(sc_indices.tobytes())

    # Opacity: uint8
    (op_uint8,) = opacity_encode(opacity_act)
    op_compressed = compress_lz4(pixel_shuffle(op_uint8.tobytes(), 1))

    # SH DC: VQ K=256
    sh_codebook, sh_indices = vq_encode(sh_dc, k=256)
    sh_idx_compressed = compress_lz4(sh_indices.tobytes())

    # Pack everything into a blob with a mini-header
    sections = []

    # Section sizes header (for decoder to know offsets)
    section_header = struct.pack("<IIIII",
        len(pos_compressed),
        len(rot_idx_compressed),
        len(sc_idx_compressed),
        len(op_compressed),
        len(sh_idx_compressed),
    )

    blob = b"".join([
        section_header,
        pos_compressed,
        rot_idx_compressed,
        sc_idx_compressed,
        op_compressed,
        sh_idx_compressed,
    ])

    total_compressed = len(blob)

    # Metadata for header
    meta = {}

    # Codebooks (per-frame; in production could be shared or per-GOP)
    codebooks = {
        "rot_codebook": rot_codebook,
        "sc_codebook": sc_codebook,
        "sh_codebook": sh_codebook,
    }

    # Original data for comparison
    original = {
        "position": pos,
        "rotation": rot,
        "scale_raw": scale_raw,
        "opacity_act": opacity_act,
        "sh_dc": sh_dc,
    }

    # Size breakdown
    sizes = {
        "position": len(pos_compressed),
        "rotation": len(rot_idx_compressed),
        "scale": len(sc_idx_compressed),
        "opacity": len(op_compressed),
        "sh_dc": len(sh_idx_compressed),
        "section_header": len(section_header),
        "total": total_compressed,
    }

    return blob, meta, codebooks, original, sizes


def decode_frame_v2(blob: bytes, codebooks: dict, meta: dict,
                    n_gaussians: int) -> dict:
    """Decode a v2 frame. Returns reconstructed attributes."""
    offset = 0

    # Read section sizes
    sizes = struct.unpack_from("<IIIII", blob, offset)
    offset += 20
    pos_sz, rot_sz, sc_sz, op_sz, sh_sz = sizes

    # Position: unshuffle + fp16 decode
    pos_compressed = blob[offset:offset + pos_sz]; offset += pos_sz
    pos_shuffled = decompress_lz4(pos_compressed, n_gaussians * 6)
    pos_arr = np.frombuffer(pos_shuffled, dtype=np.uint8)
    pos_unshuffled = pos_arr.reshape(6, n_gaussians).T.reshape(-1)
    pos = np.frombuffer(pos_unshuffled.tobytes(), dtype=np.float16).reshape(-1, 3).astype(np.float32)

    # Rotation VQ
    rot_compressed = blob[offset:offset + rot_sz]; offset += rot_sz
    rot_indices = np.frombuffer(decompress_lz4(rot_compressed, n_gaussians), dtype=np.uint8)
    rot = vq_decode(codebooks["rot_codebook"], rot_indices)

    # Scale VQ
    sc_compressed = blob[offset:offset + sc_sz]; offset += sc_sz
    sc_indices = np.frombuffer(decompress_lz4(sc_compressed, n_gaussians), dtype=np.uint8)
    scale = scale_decode(codebooks["sc_codebook"], sc_indices)

    # Opacity uint8
    op_compressed = blob[offset:offset + op_sz]; offset += op_sz
    op_shuffled = decompress_lz4(op_compressed, n_gaussians)
    op_uint8 = np.frombuffer(op_shuffled, dtype=np.uint8)
    opacity = opacity_decode(op_uint8, n_gaussians)

    # SH DC VQ
    sh_compressed = blob[offset:offset + sh_sz]; offset += sh_sz
    sh_indices = np.frombuffer(decompress_lz4(sh_compressed, n_gaussians), dtype=np.uint8)
    sh_dc = vq_decode(codebooks["sh_codebook"], sh_indices)

    return {
        "position": pos,
        "rotation": rot,
        "scale_raw": scale,
        "opacity_act": opacity,
        "sh_dc": sh_dc,
    }


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def compute_errors(original: dict, decoded: dict, n_gauss: int) -> dict:
    errors = {}

    # Position
    pos_diff = np.abs(original["position"][:n_gauss] - decoded["position"][:n_gauss])
    errors["pos_mean_abs"] = float(np.mean(pos_diff))
    errors["pos_max_abs"] = float(np.max(pos_diff))

    # Rotation angular error
    o_rot = original["rotation"][:n_gauss]
    d_rot = decoded["rotation"][:n_gauss]
    o_norm = o_rot / (np.linalg.norm(o_rot, axis=1, keepdims=True) + 1e-10)
    d_norm = d_rot / (np.linalg.norm(d_rot, axis=1, keepdims=True) + 1e-10)
    dot = np.abs(np.sum(o_norm * d_norm, axis=1))
    angles = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
    errors["rot_mean_deg"] = float(np.mean(angles))
    errors["rot_p99_deg"] = float(np.percentile(angles, 99))

    # Scale
    sc_diff = np.abs(original["scale_raw"][:n_gauss] - decoded["scale_raw"][:n_gauss])
    errors["scale_mean_abs"] = float(np.mean(sc_diff))
    errors["scale_max_abs"] = float(np.max(sc_diff))

    # Opacity
    op_diff = np.abs(original["opacity_act"][:n_gauss] - decoded["opacity_act"][:n_gauss])
    errors["opacity_mean_abs"] = float(np.mean(op_diff))
    errors["opacity_p99_abs"] = float(np.percentile(op_diff, 99))

    # SH DC
    sh_diff = np.abs(original["sh_dc"][:n_gauss] - decoded["sh_dc"][:n_gauss])
    errors["sh_mean_abs"] = float(np.mean(sh_diff))
    errors["sh_max_abs"] = float(np.max(sh_diff))

    return errors


# ---------------------------------------------------------------------------
# V1 baseline (current GSD)
# ---------------------------------------------------------------------------

def encode_frame_v1(gaussians: dict, sorted_indices: np.ndarray,
                    texture_size: int) -> int:
    """Encode with current v1 method, return compressed size."""
    from app.pipeline.ply_to_gsd import _textures_to_shuffled_blob

    all_textures = _pack_textures(gaussians, sorted_indices, texture_size)
    textures = all_textures[:4]  # pos + rot + scaleOp + sh0

    precisions = [PRECISION_FULL, PRECISION_HALF, PRECISION_HALF, PRECISION_HALF]
    blob = _textures_to_shuffled_blob(textures, precisions)
    compressed = compress_lz4(blob)
    return len(compressed), len(blob)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(ply_folder: str, num_frames: int = 5):
    ply_files = sorted([
        os.path.join(ply_folder, f) for f in os.listdir(ply_folder)
        if f.endswith(".ply")
    ])
    print(f"Found {len(ply_files)} PLY files")
    print(f"Testing {num_frames} frames\n")

    all_v1_sizes = []
    all_v2_sizes = []
    all_v2_breakdowns = []
    all_errors = []

    for i in range(min(num_frames, len(ply_files))):
        print(f"--- Frame {i} ---")
        t0 = time.perf_counter()

        g = load_gaussian_ply(ply_files[i])
        n_gauss = len(g["position"])
        sorted_indices, min_pos, max_pos = sort_3d_morton_order(g["position"])
        texture_size = math.ceil(math.sqrt(n_gauss))

        # V1 baseline
        v1_compressed_size, v1_raw_size = encode_frame_v1(g, sorted_indices, texture_size)
        all_v1_sizes.append(v1_compressed_size)

        # V2 encode
        blob, meta, codebooks, original, sizes = encode_frame_v2(
            g, sorted_indices, texture_size)
        all_v2_sizes.append(sizes["total"])
        all_v2_breakdowns.append(sizes)

        # V2 decode
        decoded = decode_frame_v2(blob, codebooks, meta, n_gauss)

        # Errors
        errors = compute_errors(original, decoded, n_gauss)
        all_errors.append(errors)

        t1 = time.perf_counter()

        print(f"  {n_gauss:,} gaussians, {t1-t0:.1f}s")
        print(f"  V1 (shuffle+LZ4): {v1_compressed_size/1e6:.2f} MB")
        print(f"  V2 (SHARP-VQ):    {sizes['total']/1e6:.2f} MB "
              f"({sizes['total']/v1_compressed_size*100:.1f}% of V1)")
        print(f"    pos:     {sizes['position']/1e6:.2f} MB")
        print(f"    rot VQ:  {sizes['rotation']/1e6:.2f} MB")
        print(f"    scale VQ:{sizes['scale']/1e6:.2f} MB")
        print(f"    opacity: {sizes['opacity']/1e6:.2f} MB")
        print(f"    sh VQ:   {sizes['sh_dc']/1e6:.2f} MB")
        print(f"  Errors:")
        print(f"    rot: {errors['rot_mean_deg']:.3f}° mean, {errors['rot_p99_deg']:.3f}° p99")
        print(f"    scale: {errors['scale_mean_abs']:.4f} mean, {errors['scale_max_abs']:.4f} max")
        print(f"    opacity: {errors['opacity_mean_abs']:.4f} mean")
        print(f"    sh_dc: {errors['sh_mean_abs']:.4f} mean")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_v1 = np.mean(all_v1_sizes)
    avg_v2 = np.mean(all_v2_sizes)
    total_frames = len(ply_files)

    print(f"\n  Per-frame average:")
    print(f"    V1 (current GSD): {avg_v1/1e6:.2f} MB")
    print(f"    V2 (SHARP-VQ):    {avg_v2/1e6:.2f} MB ({avg_v2/avg_v1*100:.1f}% of V1)")
    print(f"    Savings:          {(1-avg_v2/avg_v1)*100:.1f}%")

    print(f"\n  V2 breakdown (average):")
    for key in ["position", "rotation", "scale", "opacity", "sh_dc"]:
        avg = np.mean([b[key] for b in all_v2_breakdowns])
        print(f"    {key:<20}: {avg/1e6:.2f} MB")

    print(f"\n  Full sequence estimate ({total_frames} frames):")
    print(f"    V1: {avg_v1 * total_frames / 1e9:.2f} GB")
    print(f"    V2: {avg_v2 * total_frames / 1e9:.2f} GB")

    print(f"\n  Quality (average across frames):")
    for key in all_errors[0]:
        avg = np.mean([e[key] for e in all_errors])
        print(f"    {key:<20}: {avg:.4f}")


if __name__ == "__main__":
    ply_folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\4dgs-data\nature\ANIMAL_camel\ply"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run(ply_folder, num_frames)
