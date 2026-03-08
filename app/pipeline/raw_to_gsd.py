"""RAW to GSD (Gaussian Splatting Data) converter.

Reads a RAW sequence (sequence.json + per-frame .bin files) and produces a
single .gsd file using Byte-Shuffle + LZ4 compression.

Byte-shuffle: Each texture's pixels are rearranged so that bytes at the same
position within each pixel (e.g. all exponent bytes) are grouped together.
This creates long runs of similar values that LZ4 compresses very well.
Every frame is independent — no keyframes or delta encoding needed.
This is completely lossless.

File format (.gsd):
  [4B]  Magic "GSD1" (little-endian)
  [4B]  Header JSON length (uint32 LE)
  [NB]  Header JSON (UTF-8)
  [Frame blobs, sequential...]
    Each: [4B compressed_size (uint32 LE)] [NB LZ4 compressed data]
"""

import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np

try:
    import lz4.block
except ImportError:
    print("ERROR: lz4 package required. Install with: pip install lz4")
    sys.exit(1)

# Matches BIN_FILES order in io_benchmark.py and UE loader
BIN_FILES = [
    "position.bin", "rotation.bin", "scaleOpacity.bin",
    "sh_0.bin", "sh_1.bin", "sh_2.bin", "sh_3.bin",
    "sh_4.bin", "sh_5.bin", "sh_6.bin", "sh_7.bin",
    "sh_8.bin", "sh_9.bin", "sh_10.bin", "sh_11.bin",
]

GSD_MAGIC = b"GSD1"


def _get_sh_texture_count(sh_degree: int) -> int:
    """Number of SH textures for a given SH degree (matches UE GetSHTextureCount)."""
    return {0: 1, 1: 3, 2: 7, 3: 12}.get(sh_degree, 12)


def _pixel_shuffle(data: bytes, bytes_per_pixel: int) -> bytes:
    """Rearrange pixel data so same byte-positions are grouped together.

    For bpp=16: byte layout [p0b0 p0b1 ... p0b15 p1b0 p1b1 ... p1b15 ...]
    becomes [p0b0 p1b0 p2b0 ... p0b1 p1b1 p2b1 ... ... p0b15 p1b15 p2b15 ...]
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bytes_per_pixel).T.reshape(-1).tobytes()


def _get_bpp(precision: int) -> int:
    """Get bytes-per-pixel for a precision value (0=Full/16bpp, 1=Half/8bpp)."""
    return 16 if precision == 0 else 8


def _load_and_shuffle_frame(frame_folder: str, sh_count: int, meta: dict) -> bytes:
    """Load all .bin files for a frame, shuffle each, and concatenate."""
    bpp_list = [
        _get_bpp(meta["positionPrecision"]),
        _get_bpp(meta["rotationPrecision"]),
        _get_bpp(meta["scaleOpacityPrecision"]),
    ] + [_get_bpp(meta["shPrecision"])] * sh_count

    bin_names = list(BIN_FILES[:3]) + [f"sh_{i}.bin" for i in range(sh_count)]

    parts = []
    for bin_file, bpp in zip(bin_names, bpp_list):
        path = os.path.join(frame_folder, bin_file)
        with open(path, "rb") as f:
            raw = f.read()
        parts.append(_pixel_shuffle(raw, bpp))

    return b"".join(parts)


def convert_raw_to_gsd(
    raw_folder: str,
    output_path: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Convert a RAW sequence to a single .gsd file.

    Args:
        raw_folder: Path to RAW sequence folder containing sequence.json.
        output_path: Output .gsd file path.
        progress_callback: Callback for log messages.
        frame_progress_callback: Callback (current_frame, total_frames).

    Returns:
        Stats dict with compression info.
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Load sequence.json
    seq_path = os.path.join(raw_folder, "sequence.json")
    with open(seq_path) as f:
        seq_meta = json.load(f)

    frame_count = seq_meta["frameCount"]
    sh_degree = seq_meta.get("shDegree", 3)
    sh_count = _get_sh_texture_count(sh_degree)
    frame_folders = seq_meta["frameFolders"]

    log(f"Sequence: {seq_meta['sequenceName']}, {frame_count} frames, SH degree {sh_degree} ({sh_count} SH textures)")

    # Load first frame's metadata to get texture dimensions and precision
    meta0_path = os.path.join(raw_folder, frame_folders[0], "metadata.json")
    with open(meta0_path) as f:
        meta0 = json.load(f)

    tex_w = meta0["textureWidth"]
    tex_h = meta0["textureHeight"]
    gaussian_count = meta0["gaussianCount"]

    log(f"Texture: {tex_w}x{tex_h}, {gaussian_count} gaussians")

    # Build GSD header
    header = {
        "version": 1,
        "compression": "shuffle_lz4",
        "sequenceName": seq_meta["sequenceName"],
        "frameCount": frame_count,
        "targetFPS": seq_meta["targetFPS"],
        "shDegree": sh_degree,
        "textureWidth": tex_w,
        "textureHeight": tex_h,
        "gaussianCount": gaussian_count,
        "positionPrecision": meta0["positionPrecision"],
        "rotationPrecision": meta0["rotationPrecision"],
        "scaleOpacityPrecision": meta0["scaleOpacityPrecision"],
        "shPrecision": meta0["shPrecision"],
        "minPosition": meta0["minPosition"],
        "maxPosition": meta0["maxPosition"],
        "frames": [],  # Will be filled during encoding
    }

    # Encode frames
    t0 = time.time()
    frame_blobs = []  # list of compressed_bytes
    total_raw_size = 0
    total_compressed_size = 0

    for i in range(frame_count):
        frame_folder = os.path.join(raw_folder, frame_folders[i])
        blob = _load_and_shuffle_frame(frame_folder, sh_count, meta0)

        if i == 0:
            expected_size = len(blob)
            log(f"Uncompressed frame size: {expected_size / 1e6:.1f} MB")
        elif len(blob) != expected_size:
            raise ValueError(
                f"Frame {i} blob size {len(blob)} != expected {expected_size}. "
                f"All frames must have identical texture dimensions."
            )

        # Compress shuffled blob with LZ4 (raw block, no framing)
        compressed = lz4.block.compress(blob, store_size=False)

        frame_blobs.append(compressed)
        header["frames"].append({
            "compressedSize": len(compressed),
        })

        total_raw_size += len(blob)
        total_compressed_size += len(compressed)

        if frame_progress_callback:
            frame_progress_callback(i + 1, frame_count)

        if (i + 1) % 50 == 0 or i == frame_count - 1:
            ratio = total_compressed_size / total_raw_size * 100
            log(f"  Encoded {i + 1}/{frame_count} frames ({ratio:.1f}% of original)")

    encode_time = time.time() - t0

    # Write .gsd file
    log(f"\nWriting {output_path}...")
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        # Magic
        f.write(GSD_MAGIC)
        # Header length
        f.write(struct.pack("<I", len(header_json)))
        # Header JSON
        f.write(header_json)
        # Frame blobs
        for compressed in frame_blobs:
            f.write(struct.pack("<I", len(compressed)))
            f.write(compressed)

    file_size = os.path.getsize(output_path)

    # Stats
    frame_sizes = [len(c) for c in frame_blobs]

    stats = {
        "frame_count": frame_count,
        "uncompressed_frame_size": expected_size,
        "total_raw_size": total_raw_size,
        "total_compressed_size": total_compressed_size,
        "file_size": file_size,
        "overall_ratio": total_compressed_size / total_raw_size,
        "avg_frame_size": sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0,
        "encode_time": encode_time,
    }

    log(f"\n{'='*60}")
    log(f"GSD Compression Stats (Byte-Shuffle + LZ4)")
    log(f"{'='*60}")
    log(f"  Frames:          {frame_count}")
    log(f"  Raw frame size:  {expected_size / 1e6:.1f} MB")
    log(f"  Total RAW:       {total_raw_size / 1e9:.2f} GB")
    log(f"  Total GSD:       {file_size / 1e9:.2f} GB")
    log(f"  Overall ratio:   {stats['overall_ratio']*100:.1f}%")
    log(f"  Avg frame:       {stats['avg_frame_size'] / 1e6:.1f} MB ({stats['avg_frame_size'] / expected_size * 100:.1f}%)")
    log(f"  Encode time:     {encode_time:.1f}s ({encode_time / frame_count * 1000:.0f}ms/frame)")
    log(f"  Output file:     {output_path}")

    return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert RAW sequence to GSD (Byte-Shuffle + LZ4)")
    parser.add_argument("raw_folder", help="Path to RAW sequence folder (containing sequence.json)")
    parser.add_argument("-o", "--output", help="Output .gsd file path (default: <raw_folder>.gsd)")
    args = parser.parse_args()

    output = args.output
    if not output:
        output = args.raw_folder.rstrip("/\\") + ".gsd"

    convert_raw_to_gsd(
        raw_folder=args.raw_folder,
        output_path=output,
        progress_callback=print,
    )


if __name__ == "__main__":
    main()
