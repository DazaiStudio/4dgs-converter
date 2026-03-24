"""Full encoding benchmark: position, rotation, scaleOpacity, sh0.

Tests all textures with various encodings, then measures combined impact.
"""

import json
import struct
import sys

import lz4.block
import zstandard as zstd
import numpy as np


# ---------------------------------------------------------------------------
# GSD reader
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


def pixel_unshuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr) // bpp
    return arr.reshape(bpp, n).T.reshape(-1).tobytes()


def pixel_shuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bpp).T.reshape(-1).tobytes()


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def compress_zstd3(data: bytes) -> bytes:
    return zstd.ZstdCompressor(level=3).compress(data)


# ---------------------------------------------------------------------------
# Extract textures
# ---------------------------------------------------------------------------

def get_bpp(prec):
    return 16 if prec == 0 else 8


def extract_textures(raw_data: bytes, header: dict) -> dict:
    w = header["textureWidth"]
    h = header["textureHeight"]
    pixels = w * h

    pos_bpp = get_bpp(header["positionPrecision"])
    rot_bpp = get_bpp(header["rotationPrecision"])
    so_bpp = get_bpp(header["scaleOpacityPrecision"])
    sh_bpp = get_bpp(header.get("shPrecision", header.get("SHPrecision", 1)))

    offset = 0
    tex = {"pixels": pixels}

    # Each section is already shuffled in GSD
    tex["pos_shuffled"] = raw_data[offset:offset + pixels * pos_bpp]
    offset += pixels * pos_bpp
    tex["rot_shuffled"] = raw_data[offset:offset + pixels * rot_bpp]
    offset += pixels * rot_bpp
    tex["so_shuffled"] = raw_data[offset:offset + pixels * so_bpp]
    offset += pixels * so_bpp
    tex["sh0_shuffled"] = raw_data[offset:offset + pixels * sh_bpp]
    offset += pixels * sh_bpp

    # Unshuffle to get raw pixel data, then parse
    pos_raw = pixel_unshuffle(tex["pos_shuffled"], pos_bpp)
    rot_raw = pixel_unshuffle(tex["rot_shuffled"], rot_bpp)
    so_raw = pixel_unshuffle(tex["so_shuffled"], so_bpp)
    sh0_raw = pixel_unshuffle(tex["sh0_shuffled"], sh_bpp)

    # Parse position as fp32 (precision=0 → 16bpp → fp32 RGBA)
    tex["pos_fp32"] = np.frombuffer(pos_raw, dtype=np.float32).reshape(-1, 4)
    # Parse rotation as fp16
    tex["rot_fp16"] = np.frombuffer(rot_raw, dtype=np.float16).reshape(-1, 4)
    # Parse scaleOpacity as fp16
    tex["so_fp16"] = np.frombuffer(so_raw, dtype=np.float16).reshape(-1, 4)
    # Parse sh0 as fp16
    tex["sh0_fp16"] = np.frombuffer(sh0_raw, dtype=np.float16).reshape(-1, 4)

    return tex


# ---------------------------------------------------------------------------
# Encoding functions — each returns (encoded_bytes, bpp_for_shuffle)
# ---------------------------------------------------------------------------

# --- Position ---

def pos_fp32(data: np.ndarray) -> tuple[bytes, int]:
    """Current: fp32 RGBA, 16 bpp."""
    return data.astype(np.float32).tobytes(), 16


def pos_fp16(data: np.ndarray) -> tuple[bytes, int]:
    """fp16 RGBA, 8 bpp."""
    return data.astype(np.float16).tobytes(), 8


def pos_fixed16_normalize(data: np.ndarray, n_gaussians: int) -> tuple[bytes, int]:
    """Per-frame min/max normalized uint16 RGBA, 8 bpp.
    Store min/max as 8-byte header per channel (or global)."""
    vals = data[:n_gaussians].astype(np.float32)
    # Per-channel min/max for better precision
    ch_min = vals.min(axis=0)
    ch_max = vals.max(axis=0)
    ch_range = ch_max - ch_min
    ch_range = np.where(ch_range == 0, 1.0, ch_range)

    # Normalize all pixels (including padding)
    all_vals = data.astype(np.float32)
    normalized = (all_vals - ch_min) / ch_range
    normalized = np.clip(normalized, 0, 1)
    quantized = (normalized * 65535).astype(np.uint16)

    # 32 bytes header (4 channels × 2 floats × 4 bytes)
    header = np.concatenate([ch_min, ch_max]).astype(np.float32).tobytes()
    return header + quantized.tobytes(), 8  # same bpp as fp16


def pos_fixed16_log(data: np.ndarray, n_gaussians: int) -> tuple[bytes, int]:
    """Position with log-space encoding for better dynamic range.
    Sign bit + 15-bit log magnitude per component. Packed as uint16 RGBA."""
    vals = data.astype(np.float32)

    # For each component, encode sign + log(|v| + epsilon)
    sign = (vals >= 0).astype(np.uint16)
    magnitude = np.abs(vals) + 1e-7
    log_mag = np.log(magnitude)

    # Find range of log values from active gaussians
    active_log = log_mag[:n_gaussians]
    lmin = active_log.min()
    lmax = active_log.max()
    lrange = lmax - lmin
    if lrange == 0:
        lrange = 1.0

    # Quantize to 15-bit [0, 32767]
    normalized = (log_mag - lmin) / lrange
    normalized = np.clip(normalized, 0, 1)
    q15 = (normalized * 32767).astype(np.uint16)

    # Pack: bit15 = sign, bits[0:14] = quantized log magnitude
    packed = (sign << 15) | q15

    header = struct.pack("<ff", lmin, lmax)
    return header + packed.astype(np.uint16).tobytes(), 8


# --- Rotation ---

def rot_fp16(data: np.ndarray) -> tuple[bytes, int]:
    """Current: fp16 RGBA, 8 bpp."""
    return data.astype(np.float16).tobytes(), 8


def rot_uint8(data: np.ndarray) -> tuple[bytes, int]:
    """uint8 RGBA: map [-1,1] -> [0,255], 4 bpp."""
    q = ((data.astype(np.float32) + 1.0) / 2.0 * 255).astype(np.uint8)
    return np.clip(q, 0, 255).tobytes(), 4


# --- ScaleOpacity ---

def so_fp16(data: np.ndarray) -> tuple[bytes, int]:
    """Current: fp16 RGBA, 8 bpp."""
    return data.astype(np.float16).tobytes(), 8


def so_log_uint8(data: np.ndarray, n_gaussians: int) -> tuple[bytes, int]:
    """Log-encode scale uint8, linear-encode opacity uint8, 4 bpp."""
    so = data.astype(np.float32)
    exp_scales = so[:, :3]
    opacity = so[:, 3]

    log_scales = np.log(np.clip(exp_scales, 1e-10, None))
    active_log = log_scales[:n_gaussians]
    s_min = np.percentile(active_log, 0.1)
    s_max = np.percentile(active_log, 99.9)
    s_range = s_max - s_min
    if s_range == 0:
        s_range = 1.0

    s_norm = (log_scales - s_min) / s_range
    s_q = (np.clip(s_norm, 0, 1) * 255).astype(np.uint8)
    o_q = (np.clip(opacity, 0, 1) * 255).astype(np.uint8)

    header = struct.pack("<ff", s_min, s_max)
    return header + np.column_stack([s_q, o_q]).tobytes(), 4


# --- SH0 ---

def sh0_fp16(data: np.ndarray) -> tuple[bytes, int]:
    """Current: fp16 RGBA, 8 bpp."""
    return data.astype(np.float16).tobytes(), 8


def sh0_uint8(data: np.ndarray, n_gaussians: int) -> tuple[bytes, int]:
    """uint8 RGBA with per-channel min/max normalization, 4 bpp."""
    vals = data[:n_gaussians].astype(np.float32)
    ch_min = vals.min(axis=0)
    ch_max = vals.max(axis=0)
    ch_range = ch_max - ch_min
    ch_range = np.where(ch_range == 0, 1.0, ch_range)

    all_vals = data.astype(np.float32)
    normalized = (all_vals - ch_min) / ch_range
    q = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)

    header = np.concatenate([ch_min, ch_max]).astype(np.float32).tobytes()
    return header + q.tobytes(), 4


def sh0_uint16(data: np.ndarray, n_gaussians: int) -> tuple[bytes, int]:
    """uint16 RGBA with per-channel min/max normalization, 8 bpp (same as fp16)."""
    vals = data[:n_gaussians].astype(np.float32)
    ch_min = vals.min(axis=0)
    ch_max = vals.max(axis=0)
    ch_range = ch_max - ch_min
    ch_range = np.where(ch_range == 0, 1.0, ch_range)

    all_vals = data.astype(np.float32)
    normalized = (all_vals - ch_min) / ch_range
    q = (np.clip(normalized, 0, 1) * 65535).astype(np.uint16)

    header = np.concatenate([ch_min, ch_max]).astype(np.float32).tobytes()
    return header + q.tobytes(), 8


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def position_error(orig: np.ndarray, encoded: bytes, bpp: int, n_gauss: int, enc_name: str) -> dict:
    """Decode and compute position error."""
    orig_f32 = orig[:n_gauss].astype(np.float32)

    if "fp32" in enc_name:
        dec = np.frombuffer(encoded, dtype=np.float32).reshape(-1, 4)[:n_gauss]
    elif "fp16" in enc_name:
        dec = np.frombuffer(encoded, dtype=np.float16).reshape(-1, 4)[:n_gauss].astype(np.float32)
    elif "fixed16_normalize" in enc_name:
        header = encoded[:32]
        ch_minmax = np.frombuffer(header, dtype=np.float32)
        ch_min = ch_minmax[:4]
        ch_max = ch_minmax[4:]
        ch_range = ch_max - ch_min
        q = np.frombuffer(encoded[32:], dtype=np.uint16).reshape(-1, 4)[:n_gauss]
        dec = q.astype(np.float32) / 65535.0 * ch_range + ch_min
    elif "fixed16_log" in enc_name:
        lmin, lmax = struct.unpack("<ff", encoded[:8])
        packed = np.frombuffer(encoded[8:], dtype=np.uint16).reshape(-1, 4)[:n_gauss]
        sign = (packed >> 15).astype(np.float32) * 2 - 1  # 1 -> +1, 0 -> -1
        q15 = (packed & 0x7FFF).astype(np.float32) / 32767.0
        log_mag = q15 * (lmax - lmin) + lmin
        dec = sign * (np.exp(log_mag) - 1e-7)
    else:
        return {"mean_abs": -1, "max_abs": -1, "p99_abs": -1}

    diff = np.abs(orig_f32 - dec)
    return {
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "p99_abs": float(np.percentile(diff, 99)),
    }


def rotation_angular_error(orig_fp16: np.ndarray, dec: np.ndarray, n_gauss: int) -> dict:
    orig = orig_fp16[:n_gauss].astype(np.float32)
    orig_norm = orig / (np.linalg.norm(orig, axis=1, keepdims=True) + 1e-10)
    dec_norm = dec[:n_gauss] / (np.linalg.norm(dec[:n_gauss], axis=1, keepdims=True) + 1e-10)
    dot = np.abs(np.sum(orig_norm * dec_norm, axis=1))
    angles = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
    return {"mean_deg": float(np.mean(angles)), "p99_deg": float(np.percentile(angles, 99))}


def relative_error(orig: np.ndarray, dec: np.ndarray, n_gauss: int) -> dict:
    o = orig[:n_gauss].astype(np.float32).ravel()
    d = dec[:n_gauss].astype(np.float32).ravel()
    mask = np.abs(o) > 1e-6
    if mask.sum() == 0:
        return {"mean_rel": 0, "p99_rel": 0}
    rel = np.abs(o[mask] - d[mask]) / (np.abs(o[mask]) + 1e-10)
    return {"mean_rel": float(np.mean(rel)), "p99_rel": float(np.percentile(rel, 99))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(gsd_path: str, num_frames: int = 5):
    header, offsets = read_gsd_header(gsd_path)
    pixels = header["textureWidth"] * header["textureHeight"]
    n_gauss = header.get("gaussianCount", pixels)

    pos_bpp_orig = get_bpp(header["positionPrecision"])
    rot_bpp_orig = get_bpp(header["rotationPrecision"])
    so_bpp_orig = get_bpp(header["scaleOpacityPrecision"])
    sh_bpp_orig = get_bpp(header.get("shPrecision", 1))
    raw_frame_size = pixels * (pos_bpp_orig + rot_bpp_orig + so_bpp_orig + sh_bpp_orig)

    print(f"GSD: {gsd_path}")
    print(f"  {header['frameCount']} frames, {header['textureWidth']}x{header['textureHeight']}, "
          f"{n_gauss} gaussians, raw={raw_frame_size/1024/1024:.1f} MB/frame")
    print()

    # Load frames
    frames = []
    for i in range(min(num_frames, header["frameCount"])):
        comp = read_compressed_frame(gsd_path, offsets[i])
        raw = lz4.block.decompress(comp, uncompressed_size=raw_frame_size)
        frames.append(extract_textures(raw, header))
    print(f"Loaded {len(frames)} frames\n")

    # =========================================================================
    # Per-texture encoding tests
    # =========================================================================

    def test_encoding(name, tex_key, encode_fn, error_fn, compressors):
        """Test one encoding across all frames."""
        results = {c_name: [] for c_name in compressors}
        errors = []

        for f in frames:
            data = f[tex_key]
            if "n_gaussians" in encode_fn.__code__.co_varnames:
                encoded, enc_bpp = encode_fn(data, n_gauss)
            else:
                encoded, enc_bpp = encode_fn(data)

            # Strip header for compression (header is tiny, skip it)
            payload = encoded
            if len(encoded) > pixels * enc_bpp:
                payload = encoded[len(encoded) - pixels * enc_bpp:]

            shuffled = pixel_shuffle(payload, enc_bpp) if enc_bpp > 1 else payload

            for c_name, c_fn in compressors.items():
                compressed = c_fn(shuffled)
                results[c_name].append(len(compressed))

            # Error
            if error_fn:
                err = error_fn(f, encoded, enc_bpp, name)
                errors.append(err)

        return results, errors

    compressors = {"LZ4": compress_lz4, "Zstd3": compress_zstd3}

    # --- POSITION ---
    print("=" * 80)
    print("POSITION ENCODING")
    print("=" * 80)
    pos_raw_size = pixels * pos_bpp_orig

    pos_encs = {
        "fp32 (current)": (pos_fp32, lambda f, e, b, n: position_error(f["pos_fp32"], e, b, n_gauss, n)),
        "fp16": (pos_fp16, lambda f, e, b, n: position_error(f["pos_fp32"], e, b, n_gauss, n)),
        "fixed16 normalize": (
            lambda d: pos_fixed16_normalize(d, n_gauss),
            lambda f, e, b, n: position_error(f["pos_fp32"], e, b, n_gauss, "fixed16_normalize"),
        ),
        "fixed16 log": (
            lambda d: pos_fixed16_log(d, n_gauss),
            lambda f, e, b, n: position_error(f["pos_fp32"], e, b, n_gauss, "fixed16_log"),
        ),
    }

    print(f"\n{'Encoding':<22} {'LZ4 MB':>8} {'LZ4 %':>7} {'Zstd3 MB':>9} {'Zstd3 %':>7} {'Mean err':>9} {'P99 err':>9} {'Max err':>9}")
    print("-" * 90)

    for enc_name, (enc_fn, err_fn) in pos_encs.items():
        results = {c: [] for c in compressors}
        errors = []
        for f in frames:
            if "n_gaussians" in enc_fn.__code__.co_varnames:
                encoded, enc_bpp = enc_fn(f["pos_fp32"])
            else:
                encoded, enc_bpp = enc_fn(f["pos_fp32"])

            payload = encoded
            if len(encoded) > pixels * enc_bpp + 8:  # has header
                payload = encoded[len(encoded) - pixels * enc_bpp:]

            shuffled = pixel_shuffle(payload, enc_bpp)
            for c_name, c_fn in compressors.items():
                results[c_name].append(len(c_fn(shuffled)))

            if err_fn:
                err = err_fn(f, encoded, enc_bpp, enc_name)
                errors.append(err)

        lz4_avg = np.mean(results["LZ4"])
        zstd_avg = np.mean(results["Zstd3"])
        mean_err = np.mean([e["mean_abs"] for e in errors]) if errors else 0
        p99_err = np.mean([e["p99_abs"] for e in errors]) if errors else 0
        max_err = np.max([e["max_abs"] for e in errors]) if errors else 0

        print(f"{enc_name:<22} {lz4_avg/1e6:>7.2f} {lz4_avg/pos_raw_size:>6.1%} "
              f"{zstd_avg/1e6:>8.2f} {zstd_avg/pos_raw_size:>6.1%} "
              f"{mean_err:>9.6f} {p99_err:>9.6f} {max_err:>9.6f}")

    # --- ROTATION ---
    print(f"\n{'=' * 80}")
    print("ROTATION ENCODING")
    print("=" * 80)
    rot_raw_size = pixels * rot_bpp_orig

    rot_encs = {"fp16 (current)": rot_fp16, "uint8": rot_uint8}

    print(f"\n{'Encoding':<22} {'LZ4 MB':>8} {'LZ4 %':>7} {'Zstd3 MB':>9} {'Zstd3 %':>7} {'Mean°':>7} {'P99°':>7}")
    print("-" * 70)

    for enc_name, enc_fn in rot_encs.items():
        results = {c: [] for c in compressors}
        errors = []
        for f in frames:
            encoded, enc_bpp = enc_fn(f["rot_fp16"])
            shuffled = pixel_shuffle(encoded, enc_bpp)
            for c_name, c_fn in compressors.items():
                results[c_name].append(len(c_fn(shuffled)))

            dec = np.frombuffer(encoded, dtype=np.uint8 if "uint8" in enc_name else np.float16).reshape(-1, 4)
            if "uint8" in enc_name:
                dec = dec.astype(np.float32) / 255.0 * 2.0 - 1.0
            else:
                dec = dec.astype(np.float32)
            errors.append(rotation_angular_error(f["rot_fp16"], dec, n_gauss))

        lz4_avg = np.mean(results["LZ4"])
        zstd_avg = np.mean(results["Zstd3"])
        print(f"{enc_name:<22} {lz4_avg/1e6:>7.2f} {lz4_avg/rot_raw_size:>6.1%} "
              f"{zstd_avg/1e6:>8.2f} {zstd_avg/rot_raw_size:>6.1%} "
              f"{np.mean([e['mean_deg'] for e in errors]):>6.3f} "
              f"{np.mean([e['p99_deg'] for e in errors]):>6.3f}")

    # --- SCALE+OPACITY ---
    print(f"\n{'=' * 80}")
    print("SCALE+OPACITY ENCODING")
    print("=" * 80)
    so_raw_size = pixels * so_bpp_orig

    so_encs = {"fp16 (current)": so_fp16, "log uint8": lambda d: so_log_uint8(d, n_gauss)}

    print(f"\n{'Encoding':<22} {'LZ4 MB':>8} {'LZ4 %':>7} {'Zstd3 MB':>9} {'Zstd3 %':>7} {'Mean rel':>9} {'P99 rel':>9}")
    print("-" * 75)

    for enc_name, enc_fn in so_encs.items():
        results = {c: [] for c in compressors}
        errors = []
        for f in frames:
            encoded, enc_bpp = enc_fn(f["so_fp16"])
            payload = encoded
            if len(encoded) > pixels * enc_bpp:
                payload = encoded[len(encoded) - pixels * enc_bpp:]
            shuffled = pixel_shuffle(payload, enc_bpp)
            for c_name, c_fn in compressors.items():
                results[c_name].append(len(c_fn(shuffled)))

            # Quick decode for error
            if "log" in enc_name:
                s_min, s_max = struct.unpack("<ff", encoded[:8])
                raw_bytes = encoded[8:]
                arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 4)
                s_dec = arr[:, :3].astype(np.float32) / 255.0 * (s_max - s_min) + s_min
                exp_dec = np.exp(s_dec)
                o_dec = arr[:, 3:].astype(np.float32) / 255.0
                dec = np.column_stack([exp_dec, o_dec])
            else:
                dec = np.frombuffer(encoded, dtype=np.float16).reshape(-1, 4).astype(np.float32)
            errors.append(relative_error(f["so_fp16"], dec, n_gauss))

        lz4_avg = np.mean(results["LZ4"])
        zstd_avg = np.mean(results["Zstd3"])
        print(f"{enc_name:<22} {lz4_avg/1e6:>7.2f} {lz4_avg/so_raw_size:>6.1%} "
              f"{zstd_avg/1e6:>8.2f} {zstd_avg/so_raw_size:>6.1%} "
              f"{np.mean([e['mean_rel'] for e in errors]):>8.5f} "
              f"{np.mean([e['p99_rel'] for e in errors]):>8.5f}")

    # --- SH0 ---
    print(f"\n{'=' * 80}")
    print("SH0 ENCODING")
    print("=" * 80)
    sh_raw_size = pixels * sh_bpp_orig

    sh_encs = {
        "fp16 (current)": sh0_fp16,
        "uint8 normalize": lambda d: sh0_uint8(d, n_gauss),
        "uint16 normalize": lambda d: sh0_uint16(d, n_gauss),
    }

    print(f"\n{'Encoding':<22} {'LZ4 MB':>8} {'LZ4 %':>7} {'Zstd3 MB':>9} {'Zstd3 %':>7} {'Mean rel':>9} {'P99 rel':>9}")
    print("-" * 75)

    for enc_name, enc_fn in sh_encs.items():
        results = {c: [] for c in compressors}
        errors = []
        for f in frames:
            encoded, enc_bpp = enc_fn(f["sh0_fp16"])
            payload = encoded
            if len(encoded) > pixels * enc_bpp:
                payload = encoded[len(encoded) - pixels * enc_bpp:]
            shuffled = pixel_shuffle(payload, enc_bpp)
            for c_name, c_fn in compressors.items():
                results[c_name].append(len(c_fn(shuffled)))

            # Decode for error
            if "uint8" in enc_name:
                hdr = np.frombuffer(encoded[:32], dtype=np.float32)
                ch_min, ch_max = hdr[:4], hdr[4:]
                q = np.frombuffer(encoded[32:], dtype=np.uint8).reshape(-1, 4)
                dec = q.astype(np.float32) / 255.0 * (ch_max - ch_min) + ch_min
            elif "uint16" in enc_name:
                hdr = np.frombuffer(encoded[:32], dtype=np.float32)
                ch_min, ch_max = hdr[:4], hdr[4:]
                q = np.frombuffer(encoded[32:], dtype=np.uint16).reshape(-1, 4)
                dec = q.astype(np.float32) / 65535.0 * (ch_max - ch_min) + ch_min
            else:
                dec = np.frombuffer(encoded, dtype=np.float16).reshape(-1, 4).astype(np.float32)
            errors.append(relative_error(f["sh0_fp16"], dec, n_gauss))

        lz4_avg = np.mean(results["LZ4"])
        zstd_avg = np.mean(results["Zstd3"])
        print(f"{enc_name:<22} {lz4_avg/1e6:>7.2f} {lz4_avg/sh_raw_size:>6.1%} "
              f"{zstd_avg/1e6:>8.2f} {zstd_avg/sh_raw_size:>6.1%} "
              f"{np.mean([e['mean_rel'] for e in errors]):>8.5f} "
              f"{np.mean([e['p99_rel'] for e in errors]):>8.5f}")

    # =========================================================================
    # COMBINED: all combinations
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("COMBINED ESTIMATES (LZ4)")
    print("=" * 80)

    # Collect per-frame sizes for each encoding
    def avg_lz4_size(tex_key, enc_fn, orig_bpp):
        sizes = []
        for f in frames:
            data = f[tex_key]
            if "n_gaussians" in enc_fn.__code__.co_varnames:
                encoded, enc_bpp = enc_fn(data, n_gauss)
            else:
                encoded, enc_bpp = enc_fn(data)
            payload = encoded
            if len(encoded) > pixels * enc_bpp + 8:
                payload = encoded[len(encoded) - pixels * enc_bpp:]
            shuffled = pixel_shuffle(payload, enc_bpp)
            sizes.append(len(compress_lz4(shuffled)))
        return np.mean(sizes)

    def avg_zstd_size(tex_key, enc_fn, orig_bpp):
        sizes = []
        for f in frames:
            data = f[tex_key]
            if "n_gaussians" in enc_fn.__code__.co_varnames:
                encoded, enc_bpp = enc_fn(data, n_gauss)
            else:
                encoded, enc_bpp = enc_fn(data)
            payload = encoded
            if len(encoded) > pixels * enc_bpp + 8:
                payload = encoded[len(encoded) - pixels * enc_bpp:]
            shuffled = pixel_shuffle(payload, enc_bpp)
            sizes.append(len(compress_zstd3(shuffled)))
        return np.mean(sizes)

    # Current baseline
    pos_curr = avg_lz4_size("pos_fp32", pos_fp32, pos_bpp_orig)
    rot_curr = avg_lz4_size("rot_fp16", rot_fp16, rot_bpp_orig)
    so_curr = avg_lz4_size("so_fp16", so_fp16, so_bpp_orig)
    sh_curr = avg_lz4_size("sh0_fp16", sh0_fp16, sh_bpp_orig)
    current_total = pos_curr + rot_curr + so_curr + sh_curr

    print(f"\nCurrent (all fp16/fp32 + LZ4): {current_total/1e6:.2f} MB/frame")
    print(f"  pos={pos_curr/1e6:.2f}  rot={rot_curr/1e6:.2f}  so={so_curr/1e6:.2f}  sh0={sh_curr/1e6:.2f}")

    # Combos
    combos = [
        ("A: rot uint8 + so log8",
         pos_fp32, rot_uint8, lambda d: so_log_uint8(d, n_gauss), sh0_fp16),
        ("B: A + pos fp16",
         pos_fp16, rot_uint8, lambda d: so_log_uint8(d, n_gauss), sh0_fp16),
        ("C: B + sh0 uint8",
         pos_fp16, rot_uint8, lambda d: so_log_uint8(d, n_gauss), lambda d: sh0_uint8(d, n_gauss)),
        ("D: A + pos fixed16",
         lambda d: pos_fixed16_normalize(d, n_gauss), rot_uint8,
         lambda d: so_log_uint8(d, n_gauss), sh0_fp16),
        ("E: D + sh0 uint8",
         lambda d: pos_fixed16_normalize(d, n_gauss), rot_uint8,
         lambda d: so_log_uint8(d, n_gauss), lambda d: sh0_uint8(d, n_gauss)),
    ]

    print(f"\n{'Combo':<30} {'LZ4 MB':>8} {'vs curr':>8} {'Zstd3 MB':>9} {'vs curr':>8} {'480fr GB':>9}")
    print("-" * 80)

    for combo_name, p_fn, r_fn, s_fn, h_fn in combos:
        p = avg_lz4_size("pos_fp32", p_fn, pos_bpp_orig)
        r = avg_lz4_size("rot_fp16", r_fn, rot_bpp_orig)
        s = avg_lz4_size("so_fp16", s_fn, so_bpp_orig)
        h = avg_lz4_size("sh0_fp16", h_fn, sh_bpp_orig)
        total_lz4 = p + r + s + h

        pz = avg_zstd_size("pos_fp32", p_fn, pos_bpp_orig)
        rz = avg_zstd_size("rot_fp16", r_fn, rot_bpp_orig)
        sz = avg_zstd_size("so_fp16", s_fn, so_bpp_orig)
        hz = avg_zstd_size("sh0_fp16", h_fn, sh_bpp_orig)
        total_zstd = pz + rz + sz + hz

        savings_lz4 = (1 - total_lz4 / current_total) * 100
        savings_zstd = (1 - total_zstd / current_total) * 100
        file_lz4 = total_lz4 * header["frameCount"] / 1e9

        print(f"{combo_name:<30} {total_lz4/1e6:>7.2f} {savings_lz4:>+7.1f}% "
              f"{total_zstd/1e6:>8.2f} {savings_zstd:>+7.1f}% "
              f"{file_lz4:>8.2f}")


if __name__ == "__main__":
    gsd_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\4dgs-data\fish-2\fish-2.gsd"
    run(gsd_path)
