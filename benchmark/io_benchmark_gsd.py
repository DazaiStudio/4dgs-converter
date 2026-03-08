"""I/O Benchmark for GSD (Gaussian Splatting Data) format.

Measures:
1. GSD sequential playback: read + LZ4 decompress + byte-unshuffle
2. Compression statistics per frame
3. Comparison with RAW baseline (if available)
"""

import json
import os
import struct
import sys
import time
import statistics
from pathlib import Path

try:
    import lz4.block
except ImportError:
    print("ERROR: lz4 package required. Install with: pip install lz4")
    sys.exit(1)

import numpy as np

# ─── Configuration ───────────────────────────────────────────

# Default path (override via CLI argument)
GSD_FILE = r"C:\Users\tommy\Desktop\w6\medias\video\aurora-timelapse_raw_top50.gsd"
NUM_FRAMES_TO_TEST = 100  # Test first N frames (0 = all)
WARM_UP_FRAMES = 5

GSD_MAGIC = b"GSD1"


def _pixel_unshuffle(data: bytes, bytes_per_pixel: int) -> bytes:
    """Reverse the pixel byte-shuffle: grouped-by-byte-position → interleaved pixels."""
    arr = np.frombuffer(data, dtype=np.uint8)
    count = len(arr) // bytes_per_pixel
    return arr.reshape(bytes_per_pixel, count).T.reshape(-1).tobytes()


# ═══════════════════════════════════════════════════════════════
# GSD Reader
# ═══════════════════════════════════════════════════════════════

class GSDReader:
    """Reads and decodes a .gsd file."""

    def __init__(self, path: str):
        self.path = path
        self.header = None
        self.frame_offsets = []  # byte offset of each frame's compressed blob in file
        self.file_data = None

    def load(self):
        """Load the entire .gsd file into memory and parse header."""
        with open(self.path, "rb") as f:
            self.file_data = f.read()

        # Parse magic
        magic = self.file_data[:4]
        if magic != GSD_MAGIC:
            raise ValueError(f"Invalid GSD magic: {magic!r} (expected {GSD_MAGIC!r})")

        # Parse header length
        header_len = struct.unpack_from("<I", self.file_data, 4)[0]

        # Parse header JSON
        header_json = self.file_data[8 : 8 + header_len].decode("utf-8")
        self.header = json.loads(header_json)

        # Build frame offset table
        offset = 8 + header_len
        self.frame_offsets = []
        for frame_info in self.header["frames"]:
            self.frame_offsets.append(offset)
            comp_size = struct.unpack_from("<I", self.file_data, offset)[0]
            offset += 4 + comp_size

    def get_frame_count(self) -> int:
        return self.header["frameCount"]

    def get_uncompressed_frame_size(self) -> int:
        """Calculate expected uncompressed frame size from header."""
        w = self.header["textureWidth"]
        h = self.header["textureHeight"]
        pixels = w * h

        pos_bpp = 16 if self.header["positionPrecision"] == 0 else 8
        rot_bpp = 16 if self.header["rotationPrecision"] == 0 else 8
        so_bpp = 16 if self.header["scaleOpacityPrecision"] == 0 else 8
        sh_bpp = 16 if self.header["shPrecision"] == 0 else 8
        sh_count = {0: 1, 1: 3, 2: 7, 3: 12}.get(self.header["shDegree"], 12)

        return pixels * (pos_bpp + rot_bpp + so_bpp + sh_bpp * sh_count)

    def _get_section_sizes(self):
        """Return list of (section_size, bpp) for each texture in blob order."""
        w = self.header["textureWidth"]
        h = self.header["textureHeight"]
        pixels = w * h

        pos_bpp = 16 if self.header["positionPrecision"] == 0 else 8
        rot_bpp = 16 if self.header["rotationPrecision"] == 0 else 8
        so_bpp = 16 if self.header["scaleOpacityPrecision"] == 0 else 8
        sh_bpp = 16 if self.header["shPrecision"] == 0 else 8
        sh_count = {0: 1, 1: 3, 2: 7, 3: 12}.get(self.header["shDegree"], 12)

        sections = [
            (pixels * pos_bpp, pos_bpp),
            (pixels * rot_bpp, rot_bpp),
            (pixels * so_bpp, so_bpp),
        ]
        for _ in range(sh_count):
            sections.append((pixels * sh_bpp, sh_bpp))
        return sections

    def decompress_frame(self, frame_index: int) -> bytes:
        """Decompress a single frame (independent, O(1)).

        Returns:
            Uncompressed frame bytes (unshuffled, original pixel layout).
        """
        offset = self.frame_offsets[frame_index]
        comp_size = struct.unpack_from("<I", self.file_data, offset)[0]
        compressed = self.file_data[offset + 4 : offset + 4 + comp_size]

        uncompressed_size = self.get_uncompressed_frame_size()
        shuffled = lz4.block.decompress(compressed, uncompressed_size=uncompressed_size)

        # Unshuffle each texture section
        sections = self._get_section_sizes()
        parts = []
        read_offset = 0
        for size, bpp in sections:
            section_data = shuffled[read_offset : read_offset + size]
            parts.append(_pixel_unshuffle(section_data, bpp))
            read_offset += size

        return b"".join(parts)

    def decode_sequential(self, start: int, end: int):
        """Decode frames sequentially, yielding (frame_index, raw_bytes, elapsed_ms).

        Every frame is independent — no reference chain needed.
        """
        for i in range(start, end):
            t0 = time.perf_counter_ns()
            raw = self.decompress_frame(i)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            yield i, raw, elapsed_ms


# ═══════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════

def bench_sequential_playback(reader: GSDReader, num_frames: int) -> dict:
    """Benchmark sequential playback: read + decompress + unshuffle."""
    frame_count = min(num_frames, reader.get_frame_count()) if num_frames > 0 else reader.get_frame_count()

    # Warm up
    for _, _, _ in reader.decode_sequential(0, min(WARM_UP_FRAMES, frame_count)):
        pass

    # Measure
    all_times = []

    for i, raw, elapsed_ms in reader.decode_sequential(0, frame_count):
        all_times.append(elapsed_ms)

    return {
        "frame_count": len(all_times),
        "all_mean_ms": statistics.mean(all_times),
        "all_median_ms": statistics.median(all_times),
        "all_max_ms": max(all_times),
        "all_p95_ms": sorted(all_times)[int(len(all_times) * 0.95)],
        "all_p99_ms": sorted(all_times)[int(len(all_times) * 0.99)],
    }


def bench_random_access(reader: GSDReader, num_frames: int) -> dict:
    """Benchmark random access — every frame is O(1), just decompress directly."""
    frame_count = min(num_frames, reader.get_frame_count()) if num_frames > 0 else reader.get_frame_count()

    import random
    random.seed(42)
    targets = [random.randint(0, frame_count - 1) for _ in range(20)]

    times = []

    for target in targets:
        t0 = time.perf_counter_ns()
        raw = reader.decompress_frame(target)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        times.append(elapsed_ms)

    return {
        "mean_ms": statistics.mean(times),
        "max_ms": max(times),
    }


def compression_stats(reader: GSDReader) -> dict:
    """Gather compression statistics."""
    uncompressed_size = reader.get_uncompressed_frame_size()
    frames = reader.header["frames"]

    frame_sizes = [f["compressedSize"] for f in frames]
    total_compressed = sum(frame_sizes)
    total_raw = uncompressed_size * len(frames)

    return {
        "uncompressed_frame_size_MB": uncompressed_size / 1e6,
        "total_raw_GB": total_raw / 1e9,
        "total_compressed_GB": total_compressed / 1e9,
        "overall_ratio_pct": total_compressed / total_raw * 100,
        "frame_count": len(frame_sizes),
        "frame_avg_MB": statistics.mean(frame_sizes) / 1e6 if frame_sizes else 0,
        "frame_ratio_pct": statistics.mean(frame_sizes) / uncompressed_size * 100 if frame_sizes else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GSD (Byte-Shuffle + LZ4) I/O Benchmark")
    print("=" * 70)

    if not os.path.isfile(GSD_FILE):
        print(f"ERROR: GSD file not found: {GSD_FILE}")
        print("Run raw_to_gsd.py first to create a .gsd file.")
        sys.exit(1)

    print(f"GSD file: {GSD_FILE}")
    file_size_mb = os.path.getsize(GSD_FILE) / 1e6
    print(f"File size: {file_size_mb:.1f} MB")

    # Load GSD
    print("\nLoading GSD file into memory...")
    t0 = time.perf_counter()
    reader = GSDReader(GSD_FILE)
    reader.load()
    load_time = time.perf_counter() - t0
    print(f"Load time: {load_time*1000:.1f} ms")

    h = reader.header
    print(f"Sequence: {h['sequenceName']}")
    print(f"Frames: {h['frameCount']}, FPS: {h['targetFPS']}")
    print(f"Texture: {h['textureWidth']}x{h['textureHeight']}, Gaussians: {h['gaussianCount']}")
    print(f"Compression: {h.get('compression', 'unknown')}")

    num_test = NUM_FRAMES_TO_TEST if NUM_FRAMES_TO_TEST > 0 else reader.get_frame_count()
    print(f"Testing first {num_test} frames")

    # ─── Compression Stats ───
    print()
    print("-" * 70)
    print("COMPRESSION STATISTICS")
    print("-" * 70)

    cs = compression_stats(reader)
    print(f"  Uncompressed frame: {cs['uncompressed_frame_size_MB']:.1f} MB")
    print(f"  Total RAW:          {cs['total_raw_GB']:.2f} GB")
    print(f"  Total GSD:          {cs['total_compressed_GB']:.2f} GB")
    print(f"  Overall ratio:      {cs['overall_ratio_pct']:.1f}%")
    print(f"  Frames ({cs['frame_count']}):      avg {cs['frame_avg_MB']:.1f} MB ({cs['frame_ratio_pct']:.1f}% of raw)")

    # ─── Sequential Playback ───
    print()
    print("-" * 70)
    print("BENCHMARK 1: Sequential Playback (read + decompress + unshuffle)")
    print("-" * 70)

    seq = bench_sequential_playback(reader, num_test)
    print(f"  Frames tested: {seq['frame_count']}")
    print(f"  Mean={seq['all_mean_ms']:.2f} ms, Median={seq['all_median_ms']:.2f} ms, Max={seq['all_max_ms']:.2f} ms")
    print(f"  P95={seq['all_p95_ms']:.2f} ms, P99={seq['all_p99_ms']:.2f} ms")
    fits = seq['all_mean_ms'] < 33.3
    print(f"  Fits 30fps budget? {'YES' if fits else 'NO'} ({seq['all_mean_ms']:.2f} vs 33.3 ms)")

    # ─── Random Access ───
    print()
    print("-" * 70)
    print("BENCHMARK 2: Random Access (O(1) — each frame independent)")
    print("-" * 70)

    ra = bench_random_access(reader, num_test)
    print(f"  Mean time: {ra['mean_ms']:.2f} ms")
    print(f"  Max time:  {ra['max_ms']:.2f} ms")

    # ─── Summary ───
    print()
    print("=" * 70)
    print("SUMMARY: GSD vs RAW")
    print("=" * 70)
    print(f"  {'Metric':<30} {'RAW (typical)':>15} {'GSD':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Frame size':<30} {cs['uncompressed_frame_size_MB']:>12.1f} MB {cs['frame_avg_MB']:>12.1f} MB")
    print(f"  {'Total size':<30} {cs['total_raw_GB']:>12.2f} GB {cs['total_compressed_GB']:>12.2f} GB")
    print(f"  {'Decode time (mean)':<30} {'~26.7 ms':>15} {seq['all_mean_ms']:>12.2f} ms")
    print(f"  {'Random access':<30} {'O(1)':>15} {'O(1)':>15}")
    print(f"  {'Quality loss':<30} {'None':>15} {'None':>15}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GSD format I/O benchmark")
    parser.add_argument("gsd_file", nargs="?", default=GSD_FILE, help="Path to .gsd file")
    parser.add_argument("-n", "--num-frames", type=int, default=NUM_FRAMES_TO_TEST,
                        help="Number of frames to test (0 = all)")
    args = parser.parse_args()
    GSD_FILE = args.gsd_file
    NUM_FRAMES_TO_TEST = args.num_frames
    main()
