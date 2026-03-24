"""Rotation & Scale encoding benchmark.

Tests alternative encodings for rotation and scaleOpacity textures
to reduce entropy before LZ4 compression.

Reads real frames from GSD, re-encodes, measures compression + error.
"""

import json
import struct
import sys
import time

import lz4.block
import numpy as np


# ---------------------------------------------------------------------------
# GSD reader (same as compression benchmark)
# ---------------------------------------------------------------------------

def read_gsd_header(path: str) -> tuple[dict, list[int]]:
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GSD1"
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 4 + 4 + header_len
    offsets = []
    pos = data_start
    for fi in header["frames"]:
        offsets.append(pos)
        pos += 4 + fi["compressedSize"]
    return header, offsets


def read_compressed_frame(path: str, offset: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        comp_size = struct.unpack("<I", f.read(4))[0]
        return f.read(comp_size)


def pixel_unshuffle(data: bytes, bpp: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr) // bpp
    return arr.reshape(bpp, n).T.reshape(-1)


def pixel_shuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bpp).T.reshape(-1).tobytes()


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


# ---------------------------------------------------------------------------
# Extract textures from a decompressed GSD frame
# ---------------------------------------------------------------------------

def extract_textures(raw_data: bytes, header: dict) -> dict:
    """Split raw (shuffled) frame into individual texture byte sections."""
    w = header["textureWidth"]
    h = header["textureHeight"]
    pixels = w * h

    def bpp(prec): return 16 if prec == 0 else 8

    pos_bpp = bpp(header["positionPrecision"])
    rot_bpp = bpp(header["rotationPrecision"])
    so_bpp = bpp(header["scaleOpacityPrecision"])
    sh_bpp = bpp(header.get("shPrecision", header.get("SHPrecision", 1)))

    pos_size = pixels * pos_bpp
    rot_size = pixels * rot_bpp
    so_size = pixels * so_bpp
    sh0_size = pixels * sh_bpp

    offset = 0
    textures = {}

    textures["position_shuffled"] = raw_data[offset:offset + pos_size]
    offset += pos_size

    textures["rotation_shuffled"] = raw_data[offset:offset + rot_size]
    offset += rot_size

    textures["scaleOpacity_shuffled"] = raw_data[offset:offset + so_size]
    offset += so_size

    textures["sh0_shuffled"] = raw_data[offset:offset + sh0_size]
    offset += sh0_size

    # Unshuffle to get actual pixel data
    rot_raw = pixel_unshuffle(textures["rotation_shuffled"], rot_bpp)
    so_raw = pixel_unshuffle(textures["scaleOpacity_shuffled"], so_bpp)

    # Parse as fp16 RGBA
    rot_fp16 = np.frombuffer(rot_raw.tobytes(), dtype=np.float16).reshape(-1, 4)
    so_fp16 = np.frombuffer(so_raw.tobytes(), dtype=np.float16).reshape(-1, 4)

    textures["rotation_fp16"] = rot_fp16  # (pixels, 4) = (qZ, qX, -qY, qW)
    textures["scaleOpacity_fp16"] = so_fp16  # (pixels, 4) = (exp_s2, exp_s0, exp_s1, sigmoid_opacity)

    textures["pixels"] = pixels
    textures["rot_bpp"] = rot_bpp
    textures["so_bpp"] = so_bpp

    return textures


# ---------------------------------------------------------------------------
# Rotation encodings
# ---------------------------------------------------------------------------

def rotation_current_fp16(rot_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Current encoding: fp16 RGBA (baseline)."""
    encoded = rot_fp16.astype(np.float16).tobytes()
    decoded = np.frombuffer(encoded, dtype=np.float16).reshape(-1, 4)
    return encoded, decoded.astype(np.float32)


def rotation_smallest_three_10bit(rot_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Smallest-three: drop largest component, store 3 others as 10-bit + 2-bit index.
    Total: 32 bits = 4 bytes per quaternion (vs 8 bytes fp16).
    """
    rot = rot_fp16.astype(np.float32)
    # rot columns: (qZ, qX, -qY, qW) - convert to standard (qX, qY, qZ, qW)
    qx = rot[:, 1]
    qy = -rot[:, 2]
    qz = rot[:, 0]
    qw = rot[:, 3]
    quat = np.stack([qx, qy, qz, qw], axis=1)  # (N, 4)

    # Ensure w >= 0 (flip if negative to reduce range)
    mask = quat[:, 3] < 0
    quat[mask] *= -1

    # Find largest component
    abs_q = np.abs(quat)
    largest_idx = np.argmax(abs_q, axis=1).astype(np.uint8)  # 2-bit index

    n = len(quat)
    # Extract the 3 non-largest components
    # The 3 remaining values are in [-1/sqrt(2), 1/sqrt(2)] ≈ [-0.7071, 0.7071]
    three = np.zeros((n, 3), dtype=np.float32)
    for i in range(4):
        mask = largest_idx == i
        cols = [c for c in range(4) if c != i]
        three[mask] = quat[mask][:, cols]

    # Quantize to 10-bit: map [-0.7071, 0.7071] -> [0, 1023]
    RANGE = 1.0 / np.sqrt(2.0)
    clamped = np.clip(three, -RANGE, RANGE)
    quantized = ((clamped + RANGE) / (2 * RANGE) * 1023).astype(np.uint16)
    quantized = np.clip(quantized, 0, 1023)

    # Pack: 2-bit index + 3x 10-bit = 32 bits
    packed = (largest_idx.astype(np.uint32) << 30) | \
             (quantized[:, 0].astype(np.uint32) << 20) | \
             (quantized[:, 1].astype(np.uint32) << 10) | \
             (quantized[:, 2].astype(np.uint32))
    encoded = packed.astype(np.uint32).tobytes()

    # Decode for error measurement
    unpacked = np.frombuffer(encoded, dtype=np.uint32)
    dec_idx = (unpacked >> 30) & 0x3
    dec_a = ((unpacked >> 20) & 0x3FF).astype(np.float32) / 1023 * (2 * RANGE) - RANGE
    dec_b = ((unpacked >> 10) & 0x3FF).astype(np.float32) / 1023 * (2 * RANGE) - RANGE
    dec_c = (unpacked & 0x3FF).astype(np.float32) / 1023 * (2 * RANGE) - RANGE

    # Reconstruct full quaternion
    dec_quat = np.zeros((n, 4), dtype=np.float32)
    for i in range(4):
        mask = dec_idx == i
        cols = [c for c in range(4) if c != i]
        dec_quat[mask, cols[0]] = dec_a[mask]
        dec_quat[mask, cols[1]] = dec_b[mask]
        dec_quat[mask, cols[2]] = dec_c[mask]
        sumsq = dec_a[mask] ** 2 + dec_b[mask] ** 2 + dec_c[mask] ** 2
        dec_quat[mask, i] = np.sqrt(np.clip(1.0 - sumsq, 0, 1))

    # Convert back to GSD layout (qZ, qX, -qY, qW)
    decoded = np.stack([dec_quat[:, 2], dec_quat[:, 0], -dec_quat[:, 1], dec_quat[:, 3]], axis=1)

    return encoded, decoded


def rotation_uint8(rot_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Simple uint8 quantization: map [-1, 1] -> [0, 255]. 4 bytes per quat."""
    rot = rot_fp16.astype(np.float32)
    quantized = ((rot + 1.0) / 2.0 * 255).astype(np.uint8)
    quantized = np.clip(quantized, 0, 255)
    encoded = quantized.tobytes()

    # Decode
    decoded = quantized.astype(np.float32) / 255.0 * 2.0 - 1.0
    return encoded, decoded


def rotation_uint16(rot_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """uint16 quantization: map [-1, 1] -> [0, 65535]. 8 bytes per quat (same as fp16)."""
    rot = rot_fp16.astype(np.float32)
    quantized = ((rot + 1.0) / 2.0 * 65535).astype(np.uint16)
    quantized = np.clip(quantized, 0, 65535)
    encoded = quantized.tobytes()

    decoded = quantized.astype(np.float32) / 65535.0 * 2.0 - 1.0
    return encoded, decoded


def rotation_smallest_three_8bit(rot_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Smallest-three with 8-bit per component. 1 byte index + 3 bytes = 4 bytes.
    Actually pack as: [index_byte, a_byte, b_byte, c_byte] = 4 bytes.
    """
    rot = rot_fp16.astype(np.float32)
    qx = rot[:, 1]
    qy = -rot[:, 2]
    qz = rot[:, 0]
    qw = rot[:, 3]
    quat = np.stack([qx, qy, qz, qw], axis=1)

    mask = quat[:, 3] < 0
    quat[mask] *= -1

    abs_q = np.abs(quat)
    largest_idx = np.argmax(abs_q, axis=1).astype(np.uint8)

    n = len(quat)
    three = np.zeros((n, 3), dtype=np.float32)
    for i in range(4):
        m = largest_idx == i
        cols = [c for c in range(4) if c != i]
        three[m] = quat[m][:, cols]

    RANGE = 1.0 / np.sqrt(2.0)
    clamped = np.clip(three, -RANGE, RANGE)
    q8 = ((clamped + RANGE) / (2 * RANGE) * 255).astype(np.uint8)

    # Pack as 4 bytes: [index, a, b, c]
    encoded = np.column_stack([largest_idx, q8]).tobytes()

    # Decode
    raw = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, 4)
    dec_idx = raw[:, 0]
    dec_abc = raw[:, 1:4].astype(np.float32) / 255.0 * (2 * RANGE) - RANGE

    dec_quat = np.zeros((n, 4), dtype=np.float32)
    for i in range(4):
        m = dec_idx == i
        cols = [c for c in range(4) if c != i]
        dec_quat[m, cols[0]] = dec_abc[m, 0]
        dec_quat[m, cols[1]] = dec_abc[m, 1]
        dec_quat[m, cols[2]] = dec_abc[m, 2]
        sumsq = dec_abc[m, 0] ** 2 + dec_abc[m, 1] ** 2 + dec_abc[m, 2] ** 2
        dec_quat[m, i] = np.sqrt(np.clip(1.0 - sumsq, 0, 1))

    decoded = np.stack([dec_quat[:, 2], dec_quat[:, 0], -dec_quat[:, 1], dec_quat[:, 3]], axis=1)
    return encoded, decoded


# ---------------------------------------------------------------------------
# ScaleOpacity encodings
# ---------------------------------------------------------------------------

def scaleopacity_current_fp16(so_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Current: fp16 RGBA (baseline)."""
    encoded = so_fp16.astype(np.float16).tobytes()
    decoded = np.frombuffer(encoded, dtype=np.float16).reshape(-1, 4).astype(np.float32)
    return encoded, decoded


def scaleopacity_log_uint8(so_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Log-encode scale, linear-encode opacity as uint8.
    Scale: stored as exp(s), convert back to s = log(exp(s)), quantize to uint8.
    Opacity: already sigmoid, range [0,1], quantize to uint8.
    """
    so = so_fp16.astype(np.float32)
    # Channels: (exp_s2, exp_s0, exp_s1, sigmoid_opacity)
    exp_scales = so[:, :3]
    opacity = so[:, 3]

    # Convert back to log-scale
    log_scales = np.log(np.clip(exp_scales, 1e-10, None))

    # Find range for quantization
    s_min = np.percentile(log_scales, 0.1)
    s_max = np.percentile(log_scales, 99.9)

    # Quantize log-scale to uint8
    s_norm = (log_scales - s_min) / (s_max - s_min)
    s_q = (np.clip(s_norm, 0, 1) * 255).astype(np.uint8)

    # Quantize opacity to uint8
    o_q = (np.clip(opacity, 0, 1) * 255).astype(np.uint8)

    # Pack as 4 uint8
    encoded_arr = np.column_stack([s_q, o_q])
    # Also need to store s_min, s_max as header (8 bytes float32)
    header = struct.pack("<ff", s_min, s_max)
    encoded = header + encoded_arr.tobytes()

    # Decode
    s_decoded = s_q.astype(np.float32) / 255.0 * (s_max - s_min) + s_min
    exp_decoded = np.exp(s_decoded)
    o_decoded = o_q.astype(np.float32) / 255.0
    decoded = np.column_stack([exp_decoded, o_decoded])

    return encoded, decoded


def scaleopacity_log_uint16(so_fp16: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Log-encode scale as uint16, opacity as uint16. Same size as fp16 but potentially
    better compression due to uniform distribution after log transform."""
    so = so_fp16.astype(np.float32)
    exp_scales = so[:, :3]
    opacity = so[:, 3]

    log_scales = np.log(np.clip(exp_scales, 1e-10, None))
    s_min = np.percentile(log_scales, 0.01)
    s_max = np.percentile(log_scales, 99.99)

    s_norm = (log_scales - s_min) / (s_max - s_min)
    s_q = (np.clip(s_norm, 0, 1) * 65535).astype(np.uint16)
    o_q = (np.clip(opacity, 0, 1) * 65535).astype(np.uint16)

    encoded_arr = np.column_stack([s_q, o_q])
    header = struct.pack("<ff", s_min, s_max)
    encoded = header + encoded_arr.tobytes()

    s_decoded = s_q.astype(np.float32) / 65535.0 * (s_max - s_min) + s_min
    exp_decoded = np.exp(s_decoded)
    o_decoded = o_q.astype(np.float32) / 65535.0
    decoded = np.column_stack([exp_decoded, o_decoded])

    return encoded, decoded


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def quaternion_angular_error(original_fp16: np.ndarray, decoded: np.ndarray, n_gaussians: int) -> dict:
    """Compute angular error between original and decoded quaternions."""
    orig = original_fp16[:n_gaussians].astype(np.float32)
    dec = decoded[:n_gaussians].astype(np.float32)

    # Normalize both
    orig_norm = orig / (np.linalg.norm(orig, axis=1, keepdims=True) + 1e-10)
    dec_norm = dec / (np.linalg.norm(dec, axis=1, keepdims=True) + 1e-10)

    # Angular error: angle = 2 * arccos(|dot(q1, q2)|)
    dot = np.abs(np.sum(orig_norm * dec_norm, axis=1))
    dot = np.clip(dot, 0, 1)
    angles_rad = 2 * np.arccos(dot)
    angles_deg = np.degrees(angles_rad)

    return {
        "mean_deg": float(np.mean(angles_deg)),
        "max_deg": float(np.max(angles_deg)),
        "p99_deg": float(np.percentile(angles_deg, 99)),
        "p999_deg": float(np.percentile(angles_deg, 99.9)),
    }


def relative_error(original: np.ndarray, decoded: np.ndarray, n_gaussians: int) -> dict:
    """Compute relative error for scale/opacity."""
    orig = original[:n_gaussians].astype(np.float32)
    dec = decoded[:n_gaussians].astype(np.float32)

    # Only compare non-zero values
    mask = np.abs(orig) > 1e-6
    if mask.sum() == 0:
        return {"mean_rel": 0, "max_rel": 0, "p99_rel": 0}

    rel = np.abs(orig[mask] - dec[mask]) / (np.abs(orig[mask]) + 1e-10)
    return {
        "mean_rel": float(np.mean(rel)),
        "max_rel": float(np.max(rel)),
        "p99_rel": float(np.percentile(rel, 99)),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(gsd_path: str, num_frames: int = 5):
    print(f"Loading GSD: {gsd_path}")
    header, offsets = read_gsd_header(gsd_path)

    def bpp(prec): return 16 if prec == 0 else 8
    pixels = header["textureWidth"] * header["textureHeight"]
    n_gaussians = header.get("gaussianCount", pixels)
    raw_size_full = pixels * (
        bpp(header["positionPrecision"]) +
        bpp(header["rotationPrecision"]) +
        bpp(header["scaleOpacityPrecision"]) +
        bpp(header.get("shPrecision", 1))
    )

    rot_bpp = bpp(header["rotationPrecision"])
    so_bpp = bpp(header["scaleOpacityPrecision"])

    print(f"  Pixels: {pixels} ({header['textureWidth']}x{header['textureHeight']})")
    print(f"  Gaussians: {n_gaussians}")
    print(f"  Rotation: fp16 ({rot_bpp} bpp), ScaleOpacity: fp16 ({so_bpp} bpp)")
    print()

    # Load frames
    frames = []
    for i in range(min(num_frames, header["frameCount"])):
        comp = read_compressed_frame(gsd_path, offsets[i])
        raw = lz4.block.decompress(comp, uncompressed_size=raw_size_full)
        tex = extract_textures(raw, header)
        frames.append(tex)
        print(f"  Frame {i} loaded")
    print()

    # =========================================================================
    # Rotation encoding benchmark
    # =========================================================================
    print("=" * 75)
    print("ROTATION ENCODING BENCHMARK")
    print("=" * 75)

    rot_encodings = {
        "fp16 RGBA (current)": rotation_current_fp16,
        "uint8 RGBA": rotation_uint8,
        "uint16 RGBA": rotation_uint16,
        "smallest-3, 10-bit": rotation_smallest_three_10bit,
        "smallest-3, 8-bit": rotation_smallest_three_8bit,
    }

    print(f"\n{'Encoding':<25} {'Raw':>7} {'Shuffled+LZ4':>13} {'Ratio':>7} "
          f"{'vs curr':>8} {'Mean°':>7} {'P99°':>7} {'Max°':>7}")
    print("-" * 100)

    baseline_lz4_size = None

    for name, enc_fn in rot_encodings.items():
        raw_sizes = []
        lz4_sizes = []
        errors = []

        for frame in frames:
            encoded, decoded = enc_fn(frame["rotation_fp16"])
            raw_sizes.append(len(encoded))

            # Determine bpp for shuffle
            n_pixels = frame["pixels"]
            enc_bpp = len(encoded) // n_pixels if len(encoded) >= n_pixels else len(encoded) // n_pixels + 1
            if enc_bpp == 0:
                enc_bpp = 1

            # Shuffle and compress
            if len(encoded) % enc_bpp == 0 and enc_bpp > 1:
                shuffled = pixel_shuffle(encoded, enc_bpp)
            else:
                shuffled = encoded
            compressed = compress_lz4(shuffled)
            lz4_sizes.append(len(compressed))

            err = quaternion_angular_error(frame["rotation_fp16"], decoded, n_gaussians)
            errors.append(err)

        avg_raw = np.mean(raw_sizes)
        avg_lz4 = np.mean(lz4_sizes)
        ratio = avg_lz4 / (pixels * rot_bpp)
        avg_err = np.mean([e["mean_deg"] for e in errors])
        p99_err = np.mean([e["p99_deg"] for e in errors])
        max_err = np.max([e["max_deg"] for e in errors])

        if baseline_lz4_size is None:
            baseline_lz4_size = avg_lz4
            vs_curr = "baseline"
        else:
            savings = (1 - avg_lz4 / baseline_lz4_size) * 100
            vs_curr = f"-{savings:.1f}%"

        print(f"{name:<25} {avg_raw/1024/1024:>6.2f}M {avg_lz4/1024/1024:>11.2f}M {ratio:>6.1%} "
              f"{vs_curr:>8} {avg_err:>6.3f} {p99_err:>6.3f} {max_err:>6.3f}")

    # =========================================================================
    # ScaleOpacity encoding benchmark
    # =========================================================================
    print()
    print("=" * 75)
    print("SCALE+OPACITY ENCODING BENCHMARK")
    print("=" * 75)

    so_encodings = {
        "fp16 RGBA (current)": scaleopacity_current_fp16,
        "log uint8 RGBA": scaleopacity_log_uint8,
        "log uint16 RGBA": scaleopacity_log_uint16,
    }

    print(f"\n{'Encoding':<25} {'Raw':>7} {'Shuffled+LZ4':>13} {'Ratio':>7} "
          f"{'vs curr':>8} {'Mean rel':>9} {'P99 rel':>9}")
    print("-" * 85)

    baseline_so_lz4 = None

    for name, enc_fn in so_encodings.items():
        raw_sizes = []
        lz4_sizes = []
        errors = []

        for frame in frames:
            encoded, decoded = enc_fn(frame["scaleOpacity_fp16"])
            raw_sizes.append(len(encoded))

            n_pixels = frame["pixels"]
            if "uint8" in name:
                enc_bpp = 4
            elif "uint16" in name:
                enc_bpp = 8
            else:
                enc_bpp = so_bpp

            data_to_compress = encoded
            # Strip header bytes for log encodings
            if name.startswith("log"):
                data_to_compress = encoded[8:]  # skip 8-byte header

            if len(data_to_compress) % enc_bpp == 0 and enc_bpp > 1:
                shuffled = pixel_shuffle(data_to_compress, enc_bpp)
            else:
                shuffled = data_to_compress
            compressed = compress_lz4(shuffled)
            lz4_sizes.append(len(compressed))

            err = relative_error(
                frame["scaleOpacity_fp16"],
                decoded,
                n_gaussians,
            )
            errors.append(err)

        avg_raw = np.mean(raw_sizes)
        avg_lz4 = np.mean(lz4_sizes)
        ratio = avg_lz4 / (pixels * so_bpp)
        avg_err = np.mean([e["mean_rel"] for e in errors])
        p99_err = np.mean([e["p99_rel"] for e in errors])

        if baseline_so_lz4 is None:
            baseline_so_lz4 = avg_lz4
            vs_curr = "baseline"
        else:
            savings = (1 - avg_lz4 / baseline_so_lz4) * 100
            vs_curr = f"-{savings:.1f}%"

        print(f"{name:<25} {avg_raw/1024/1024:>6.2f}M {avg_lz4/1024/1024:>11.2f}M {ratio:>6.1%} "
              f"{vs_curr:>8} {avg_err:>8.5f} {p99_err:>8.5f}")

    # =========================================================================
    # Combined impact estimate
    # =========================================================================
    print()
    print("=" * 75)
    print("COMBINED IMPACT ESTIMATE")
    print("=" * 75)

    # Current frame breakdown (from first compression benchmark)
    pos_lz4 = np.mean([len(compress_lz4(f["position_shuffled"])) for f in frames])
    rot_current_lz4 = baseline_lz4_size
    so_current_lz4 = baseline_so_lz4
    sh0_lz4 = np.mean([len(compress_lz4(f["sh0_shuffled"])) for f in frames])

    current_total = pos_lz4 + rot_current_lz4 + so_current_lz4 + sh0_lz4

    print(f"\nCurrent frame (shuffle+LZ4):")
    print(f"  position:     {pos_lz4/1024/1024:.2f} MB")
    print(f"  rotation:     {rot_current_lz4/1024/1024:.2f} MB")
    print(f"  scaleOpacity: {so_current_lz4/1024/1024:.2f} MB")
    print(f"  sh0:          {sh0_lz4/1024/1024:.2f} MB")
    print(f"  TOTAL:        {current_total/1024/1024:.2f} MB ({current_total/(pixels*(bpp(header['positionPrecision'])+rot_bpp+so_bpp+bpp(header.get('shPrecision',1)))):.1%} of raw)")

    # Best rotation encoding
    for name, enc_fn in rot_encodings.items():
        if name == "fp16 RGBA (current)":
            continue
        lz4_sizes = []
        for frame in frames:
            encoded, _ = enc_fn(frame["rotation_fp16"])
            n_pixels = frame["pixels"]
            enc_bpp = len(encoded) // n_pixels
            if enc_bpp == 0:
                enc_bpp = 1
            if len(encoded) % enc_bpp == 0 and enc_bpp > 1:
                shuffled = pixel_shuffle(encoded, enc_bpp)
            else:
                shuffled = encoded
            lz4_sizes.append(len(compress_lz4(shuffled)))
        avg = np.mean(lz4_sizes)

        # Best SO encoding
        for so_name, so_fn in so_encodings.items():
            if so_name == "fp16 RGBA (current)":
                continue
            so_lz4_sizes = []
            for frame in frames:
                enc, _ = so_fn(frame["scaleOpacity_fp16"])
                data = enc[8:] if so_name.startswith("log") else enc
                so_enc_bpp = 4 if "uint8" in so_name else 8
                if len(data) % so_enc_bpp == 0 and so_enc_bpp > 1:
                    sh = pixel_shuffle(data, so_enc_bpp)
                else:
                    sh = data
                so_lz4_sizes.append(len(compress_lz4(sh)))
            so_avg = np.mean(so_lz4_sizes)

            new_total = pos_lz4 + avg + so_avg + sh0_lz4
            savings = (1 - new_total / current_total) * 100

            print(f"\n  With [{name}] + [{so_name}]:")
            print(f"    rotation:     {avg/1024/1024:.2f} MB")
            print(f"    scaleOpacity: {so_avg/1024/1024:.2f} MB")
            print(f"    TOTAL:        {new_total/1024/1024:.2f} MB  (saves {savings:.1f}%)")
            if header["frameCount"] > 0:
                total_file = new_total * header["frameCount"]
                curr_file = current_total * header["frameCount"]
                print(f"    Full file:    {total_file/1024/1024/1024:.2f} GB  (was {curr_file/1024/1024/1024:.2f} GB)")


if __name__ == "__main__":
    gsd_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\4dgs-data\fish-2\fish-2.gsd"
    run_benchmark(gsd_path)
