"""GSD Compression Benchmark.

Reads frames from an existing .gsd file, decompresses them,
then recompresses with various strategies to compare:
  - LZ4 (current baseline)
  - Zstd at various levels
  - XOR delta + LZ4
  - XOR delta + Zstd
  - Bitshuffle variants

Measures both compression ratio and decompression speed.
"""

import json
import struct
import sys
import time
from pathlib import Path

import lz4.block
import numpy as np
import zstandard as zstd


# ---------------------------------------------------------------------------
# GSD reader helpers
# ---------------------------------------------------------------------------

def read_gsd_header(path: str) -> tuple[dict, list[int]]:
    """Read GSD header and return (header_dict, frame_offsets)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GSD1", f"Bad magic: {magic}"
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
    """Read a single compressed frame blob from GSD file."""
    with open(path, "rb") as f:
        f.seek(offset)
        comp_size = struct.unpack("<I", f.read(4))[0]
        return f.read(comp_size)


def calc_raw_frame_size(header: dict) -> int:
    """Calculate uncompressed frame size in bytes."""
    w = header["textureWidth"]
    h = header["textureHeight"]
    pixels = w * h

    def bpp(prec): return 16 if prec == 0 else 8

    pos_bpp = bpp(header["positionPrecision"])
    rot_bpp = bpp(header["rotationPrecision"])
    so_bpp = bpp(header["scaleOpacityPrecision"])
    sh_bpp = bpp(header.get("shPrecision", header.get("SHPrecision", 1)))

    sh_deg = header.get("shDegree", 0)
    sh_count = {0: 1, 1: 3, 2: 7, 3: 12}.get(sh_deg, 1)

    return pixels * (pos_bpp + rot_bpp + so_bpp + sh_bpp * sh_count)


# ---------------------------------------------------------------------------
# Compression strategies
# ---------------------------------------------------------------------------

def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def decompress_lz4(data: bytes, raw_size: int) -> bytes:
    return lz4.block.decompress(data, uncompressed_size=raw_size)


def compress_zstd(data: bytes, level: int = 3) -> bytes:
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def decompress_zstd(data: bytes) -> bytes:
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


def pixel_shuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bpp).T.reshape(-1).tobytes()


def pixel_unshuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr) // bpp
    return arr.reshape(bpp, n).T.reshape(-1).tobytes()


def xor_delta(current: bytes, previous: bytes) -> bytes:
    """XOR two byte buffers."""
    a = np.frombuffer(current, dtype=np.uint8)
    b = np.frombuffer(previous, dtype=np.uint8)
    return np.bitwise_xor(a, b).tobytes()


def xor_restore(delta: bytes, previous: bytes) -> bytes:
    """Restore frame from XOR delta."""
    a = np.frombuffer(delta, dtype=np.uint8)
    b = np.frombuffer(previous, dtype=np.uint8)
    return np.bitwise_xor(a, b).tobytes()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_frame(raw_data: bytes, raw_size: int, label: str,
                    compress_fn, decompress_fn, iterations: int = 5) -> dict:
    """Benchmark a compression strategy on a single frame."""
    compressed = compress_fn(raw_data)
    comp_size = len(compressed)

    # Warm up
    decompress_fn(compressed)

    # Time decompression
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = decompress_fn(compressed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return {
        "label": label,
        "compressed_size": comp_size,
        "ratio": comp_size / raw_size,
        "decompress_ms_avg": np.mean(times),
        "decompress_ms_min": np.min(times),
        "decompress_ms_max": np.max(times),
    }


def run_benchmark(gsd_path: str, num_frames: int = 20, sample_step: int = 1):
    """Run full compression benchmark on a GSD file."""
    print(f"Loading GSD: {gsd_path}")
    header, offsets = read_gsd_header(gsd_path)
    raw_size = calc_raw_frame_size(header)
    total_frames = header["frameCount"]

    print(f"  Frames: {total_frames}, Texture: {header['textureWidth']}x{header['textureHeight']}")
    print(f"  Raw frame size: {raw_size / 1024 / 1024:.2f} MB")
    print(f"  SH degree: {header.get('shDegree', 0)}")
    print(f"  Precisions: pos={header['positionPrecision']} rot={header['rotationPrecision']} "
          f"so={header['scaleOpacityPrecision']} sh={header.get('shPrecision', '?')}")
    print()

    # Select frames to test
    frame_indices = list(range(0, min(total_frames, num_frames * sample_step), sample_step))[:num_frames]
    print(f"Testing {len(frame_indices)} frames: {frame_indices[:5]}...{frame_indices[-3:]}")
    print()

    # Decompress all selected frames
    print("Decompressing frames from GSD...")
    raw_frames = {}
    for i, fi in enumerate(frame_indices):
        comp_data = read_compressed_frame(gsd_path, offsets[fi])
        raw_frames[fi] = decompress_lz4(comp_data, raw_size)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(frame_indices)} done")
    print()

    # =========================================================================
    # Test 1: Single-frame compression (no delta)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: Single-frame compression (shuffled data)")
    print("=" * 70)

    strategies_single = {
        "LZ4 (baseline)": (compress_lz4, lambda d: decompress_lz4(d, raw_size)),
        "Zstd level 1": (lambda d: compress_zstd(d, 1), decompress_zstd),
        "Zstd level 3": (lambda d: compress_zstd(d, 3), decompress_zstd),
        "Zstd level 6": (lambda d: compress_zstd(d, 6), decompress_zstd),
        "Zstd level 9": (lambda d: compress_zstd(d, 9), decompress_zstd),
        "Zstd level 15": (lambda d: compress_zstd(d, 15), decompress_zstd),
    }

    results_single = {name: [] for name in strategies_single}

    for fi in frame_indices:
        data = raw_frames[fi]  # already shuffled from GSD
        for name, (cfn, dfn) in strategies_single.items():
            r = benchmark_frame(data, raw_size, name, cfn, dfn, iterations=3)
            results_single[name].append(r)

    print(f"\n{'Strategy':<25} {'Ratio':>8} {'Size/frame':>12} {'Decomp ms':>10} {'Decomp min':>10}")
    print("-" * 70)
    for name, results in results_single.items():
        avg_ratio = np.mean([r["ratio"] for r in results])
        avg_size = np.mean([r["compressed_size"] for r in results])
        avg_ms = np.mean([r["decompress_ms_avg"] for r in results])
        min_ms = np.min([r["decompress_ms_min"] for r in results])
        print(f"{name:<25} {avg_ratio:>7.1%} {avg_size/1024/1024:>10.2f} MB {avg_ms:>8.1f} ms {min_ms:>8.1f} ms")

    # =========================================================================
    # Test 2: Re-shuffle then compress (unshuffle GSD data, re-shuffle, compress)
    # This tests if our shuffle is already optimal or if bitshuffle could help
    # =========================================================================

    # =========================================================================
    # Test 3: XOR Delta compression
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 2: XOR Delta compression (consecutive frames)")
    print("=" * 70)

    # Need consecutive frames for delta
    consec_start = 0
    consec_count = min(20, total_frames)
    consec_indices = list(range(consec_start, consec_start + consec_count))

    # Load consecutive frames
    print(f"Loading {consec_count} consecutive frames starting at {consec_start}...")
    consec_frames = {}
    for fi in consec_indices:
        if fi in raw_frames:
            consec_frames[fi] = raw_frames[fi]
        else:
            comp_data = read_compressed_frame(gsd_path, offsets[fi])
            consec_frames[fi] = decompress_lz4(comp_data, raw_size)
    print()

    delta_strategies = {
        "XOR delta + LZ4": (compress_lz4, lambda d: decompress_lz4(d, raw_size)),
        "XOR delta + Zstd 1": (lambda d: compress_zstd(d, 1), decompress_zstd),
        "XOR delta + Zstd 3": (lambda d: compress_zstd(d, 3), decompress_zstd),
        "XOR delta + Zstd 9": (lambda d: compress_zstd(d, 9), decompress_zstd),
    }

    # Also test: shuffle the delta before compressing
    delta_shuffle_strategies = {
        "XOR delta + shuffle + LZ4": (compress_lz4, lambda d: decompress_lz4(d, raw_size)),
        "XOR delta + shuffle + Zstd 3": (lambda d: compress_zstd(d, 3), decompress_zstd),
    }

    results_delta = {name: [] for name in delta_strategies}
    results_delta_shuffle = {name: [] for name in delta_shuffle_strategies}
    results_keyframe = []  # keyframe (non-delta) sizes for comparison

    # Calculate bpp for shuffle
    def get_bpp(prec): return 16 if prec == 0 else 8
    pos_bpp = get_bpp(header["positionPrecision"])
    rot_bpp = get_bpp(header["rotationPrecision"])
    so_bpp = get_bpp(header["scaleOpacityPrecision"])
    sh_bpp = get_bpp(header.get("shPrecision", header.get("SHPrecision", 1)))
    pixels = header["textureWidth"] * header["textureHeight"]

    # For delta+shuffle: we need to unshuffle first, XOR, then re-shuffle
    # But GSD data is already shuffled. For XOR delta on shuffled data,
    # we can XOR directly (shuffle is linear, XOR is bitwise - order doesn't matter)
    # shuffle(A) XOR shuffle(B) == shuffle(A XOR B)
    # So XOR on shuffled data IS valid and equivalent!

    for i, fi in enumerate(consec_indices[1:], 1):
        prev_fi = consec_indices[i - 1]
        curr_data = consec_frames[fi]
        prev_data = consec_frames[prev_fi]

        delta_data = xor_delta(curr_data, prev_data)

        # Verify correctness
        restored = xor_restore(delta_data, prev_data)
        assert restored == curr_data, "XOR delta restore failed!"

        for name, (cfn, dfn) in delta_strategies.items():
            r = benchmark_frame(delta_data, raw_size, name, cfn, dfn, iterations=3)
            results_delta[name].append(r)

        # For delta+shuffle: unshuffle delta, re-shuffle with different params
        # Actually, since data is already shuffled, XOR of shuffled = shuffled XOR
        # Let's just test with the delta data as-is (it's already in shuffled domain)
        for name, (cfn, dfn) in delta_shuffle_strategies.items():
            r = benchmark_frame(delta_data, raw_size, name, cfn, dfn, iterations=3)
            results_delta_shuffle[name].append(r)

    # Print results
    print(f"\n{'Strategy':<35} {'Ratio':>8} {'Size/frame':>12} {'Decomp ms':>10}")
    print("-" * 70)

    # Baseline for comparison
    baseline_ratios = [r["ratio"] for r in results_single["LZ4 (baseline)"]]
    print(f"{'LZ4 no-delta (baseline)':<35} {np.mean(baseline_ratios):>7.1%} "
          f"{np.mean([r['compressed_size'] for r in results_single['LZ4 (baseline)']])/1024/1024:>10.2f} MB "
          f"{np.mean([r['decompress_ms_avg'] for r in results_single['LZ4 (baseline)']]):>8.1f} ms")
    print("-" * 70)

    for name, results in results_delta.items():
        avg_ratio = np.mean([r["ratio"] for r in results])
        avg_size = np.mean([r["compressed_size"] for r in results])
        avg_ms = np.mean([r["decompress_ms_avg"] for r in results])
        print(f"{name:<35} {avg_ratio:>7.1%} {avg_size/1024/1024:>10.2f} MB {avg_ms:>8.1f} ms")

    for name, results in results_delta_shuffle.items():
        avg_ratio = np.mean([r["ratio"] for r in results])
        avg_size = np.mean([r["compressed_size"] for r in results])
        avg_ms = np.mean([r["decompress_ms_avg"] for r in results])
        print(f"{name:<35} {avg_ratio:>7.1%} {avg_size/1024/1024:>10.2f} MB {avg_ms:>8.1f} ms")

    # =========================================================================
    # Test 3: Per-texture analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 3: Per-texture compression analysis (frame 0)")
    print("=" * 70)

    frame0 = consec_frames[0]
    # Split into textures (data is shuffled, each texture section is contiguous)
    tex_sizes = [
        ("position", pixels * pos_bpp),
        ("rotation", pixels * rot_bpp),
        ("scaleOpacity", pixels * so_bpp),
        ("sh0", pixels * sh_bpp),
    ]

    offset = 0
    print(f"\n{'Texture':<15} {'Raw MB':>8} {'LZ4':>8} {'LZ4 %':>7} {'Zstd3':>8} {'Zstd3 %':>7} {'Zstd9':>8} {'Zstd9 %':>7}")
    print("-" * 75)
    for tex_name, tex_size in tex_sizes:
        tex_data = frame0[offset:offset + tex_size]
        offset += tex_size

        lz4_comp = compress_lz4(tex_data)
        zstd3_comp = compress_zstd(tex_data, 3)
        zstd9_comp = compress_zstd(tex_data, 9)

        print(f"{tex_name:<15} {tex_size/1024/1024:>7.2f} "
              f"{len(lz4_comp)/1024/1024:>7.2f} {len(lz4_comp)/tex_size:>6.1%} "
              f"{len(zstd3_comp)/1024/1024:>7.2f} {len(zstd3_comp)/tex_size:>6.1%} "
              f"{len(zstd9_comp)/1024/1024:>7.2f} {len(zstd9_comp)/tex_size:>6.1%}")

    # Per-texture delta analysis
    print(f"\n--- Per-texture XOR Delta (frame 1 vs frame 0) ---")
    frame1 = consec_frames[1]
    offset0 = 0
    offset1 = 0
    print(f"{'Texture':<15} {'Delta LZ4':>8} {'D+LZ4 %':>7} {'Delta Zstd3':>10} {'D+Z3 %':>7} {'Zero%':>7}")
    print("-" * 60)
    for tex_name, tex_size in tex_sizes:
        tex0 = frame0[offset0:offset0 + tex_size]
        tex1 = frame1[offset1:offset1 + tex_size]
        offset0 += tex_size
        offset1 += tex_size

        delta = xor_delta(tex1, tex0)
        zero_ratio = np.count_nonzero(np.frombuffer(delta, dtype=np.uint8) == 0) / len(delta)

        dlz4 = compress_lz4(delta)
        dzstd3 = compress_zstd(delta, 3)

        print(f"{tex_name:<15} "
              f"{len(dlz4)/1024/1024:>7.2f} {len(dlz4)/tex_size:>6.1%} "
              f"{len(dzstd3)/1024/1024:>9.2f} {len(dzstd3)/tex_size:>6.1%} "
              f"{zero_ratio:>6.1%}")

    # =========================================================================
    # Test 4: Keyframe interval analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 4: Keyframe interval analysis (Zstd 3)")
    print("=" * 70)
    print("Simulating different keyframe intervals with XOR delta + Zstd 3")

    for kf_interval in [1, 5, 10, 24, 30]:
        total_comp = 0
        for i, fi in enumerate(consec_indices):
            curr_data = consec_frames[fi]
            if i % kf_interval == 0:
                # Keyframe: compress as-is
                comp = compress_zstd(curr_data, 3)
            else:
                # Delta frame
                prev_fi = consec_indices[i - 1]
                delta = xor_delta(curr_data, consec_frames[prev_fi])
                comp = compress_zstd(delta, 3)
            total_comp += len(comp)

        avg_comp = total_comp / len(consec_indices)
        avg_ratio = avg_comp / raw_size
        print(f"  KF every {kf_interval:>2} frames: avg {avg_comp/1024/1024:.2f} MB/frame ({avg_ratio:.1%} of raw)")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Raw frame size: {raw_size / 1024 / 1024:.2f} MB")
    print(f"Current (shuffle+LZ4): ~{np.mean(baseline_ratios):.1%} of raw")
    print()
    print("Best single-frame alternatives:")
    for name, results in sorted(results_single.items(), key=lambda x: np.mean([r["ratio"] for r in x[1]])):
        avg_ratio = np.mean([r["ratio"] for r in results])
        avg_ms = np.mean([r["decompress_ms_avg"] for r in results])
        print(f"  {name:<25} {avg_ratio:.1%}  ({avg_ms:.1f} ms decompress)")

    print()
    print("Best delta alternatives:")
    all_delta = {**results_delta, **results_delta_shuffle}
    for name, results in sorted(all_delta.items(), key=lambda x: np.mean([r["ratio"] for r in x[1]])):
        avg_ratio = np.mean([r["ratio"] for r in results])
        avg_ms = np.mean([r["decompress_ms_avg"] for r in results])
        print(f"  {name:<35} {avg_ratio:.1%}  ({avg_ms:.1f} ms decompress)")


if __name__ == "__main__":
    gsd_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\4dgs-data\fish-2\fish-2.gsd"
    run_benchmark(gsd_path)
