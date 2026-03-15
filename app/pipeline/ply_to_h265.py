"""PLY sequence → H.265 encoded MP4 streams for web/UE5 streaming."""

import math
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order


# --- Quantization ---

def quantize_position(pos: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Quantize (N,3) float32 position → uint16 → split into high/low uint8."""
    norm = (pos - bounds_min) / (bounds_max - bounds_min)
    u16 = (norm * 65535).clip(0, 65535).astype(np.uint16)
    high = (u16 >> 8).astype(np.uint8)
    low = (u16 & 0xFF).astype(np.uint8)
    return high, low


def dequantize_position(high: np.ndarray, low: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Reconstruct float32 position from high/low uint8 (for verification)."""
    u16 = high.astype(np.uint16) << 8 | low.astype(np.uint16)
    norm = u16.astype(np.float32) / 65535.0
    return bounds_min + norm * (bounds_max - bounds_min)


def quantize_uint8(data: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Quantize float data to uint8 using min-max normalization."""
    norm = (data - bounds_min) / (bounds_max - bounds_min)
    return (norm * 255).clip(0, 255).astype(np.uint8)


def dequantize_uint8(data: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Reconstruct float from uint8 (for verification)."""
    norm = data.astype(np.float32) / 255.0
    return bounds_min + norm * (bounds_max - bounds_min)


# --- Activation & Bounds ---

def activate_gaussians(gaussians: dict) -> dict:
    """Apply activations to raw PLY attributes (no coordinate transform).

    Coordinate space: COLMAP (raw PLY values). The COLMAP-to-rendering
    transform (axis swaps, scaling) is deferred to the playback shader.

    Quaternion order: (W, X, Y, Z) — matches PLY convention where
    rot_0=W, rot_1=X, rot_2=Y, rot_3=Z.

    Returns dict with:
      position: (N, 3) float32 — raw COLMAP coords (x, y, z)
      rotation: (N, 4) float32 — normalized quaternion (W, X, Y, Z)
      scale_opacity: (N, 4) float32 — [exp(s0), exp(s1), exp(s2), sigmoid(opa)]
      sh_dc: (N, 3) float32 — DC color coefficients
    """
    rot = gaussians["rotation"].copy()
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)

    scale = np.exp(gaussians["scale"])
    opacity = 1.0 / (1.0 + np.exp(-gaussians["opacity"]))
    so = np.column_stack([scale, opacity])

    return {
        "position": gaussians["position"],
        "rotation": rot,
        "scale_opacity": so,
        "sh_dc": gaussians["sh_dc"],
    }


def compute_global_bounds(ply_paths: list[Path], sample_ratio: float = 0.1,
                          points_per_frame: int = 50000, seed: int = 42) -> dict:
    """Sample frames to compute global min/max per attribute with percentile clipping.

    Rotation uses fixed bounds [-1, 1] (quaternion components are always in
    this range after normalization), not percentile-based, to avoid clipping.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    n_sample = max(3, int(len(ply_paths) * sample_ratio))
    sampled = rng.sample(list(ply_paths), min(n_sample, len(ply_paths)))

    all_vals = {"position": [], "scale_opacity": [], "sh_dc": []}

    for path in sampled:
        gaussians = load_gaussian_ply(str(path))
        activated = activate_gaussians(gaussians)
        n = len(activated["position"])
        idx = np_rng.choice(n, min(points_per_frame, n), replace=False)

        for key in all_vals:
            all_vals[key].append(activated[key][idx])

    bounds = {}
    for key in all_vals:
        merged = np.concatenate(all_vals[key])
        bounds[key] = {
            "min": np.percentile(merged, 0.1, axis=0).astype(np.float32),
            "max": np.percentile(merged, 99.9, axis=0).astype(np.float32),
        }

    # Rotation: fixed bounds [-1, 1] (quaternion components always in this range)
    bounds["rotation"] = {
        "min": np.full(4, -1.0, dtype=np.float32),
        "max": np.full(4, 1.0, dtype=np.float32),
    }

    return bounds


# --- Channel Tiling ---

def _tile_horizontal(arrays: list[np.ndarray], grid_h: int, grid_w: int) -> np.ndarray:
    """Tile multiple (N,) uint8 arrays horizontally into one row of grid_w-wide columns."""
    cols = [arr.reshape(grid_h, grid_w) for arr in arrays]
    return np.concatenate(cols, axis=1)


def tile_stream_position(pos_high: np.ndarray, pos_low: np.ndarray,
                         grid_h: int, grid_w: int) -> np.ndarray:
    """Tile position high+low → (2*grid_h, 3*grid_w) grayscale.

    Layout:
      Row 0: [posHi_X | posHi_Y | posHi_Z]
      Row 1: [posLo_X | posLo_Y | posLo_Z]
    """
    row_hi = _tile_horizontal([pos_high[:, i] for i in range(3)], grid_h, grid_w)
    row_lo = _tile_horizontal([pos_low[:, i] for i in range(3)], grid_h, grid_w)
    return np.concatenate([row_hi, row_lo], axis=0)


def tile_stream_motion(rot: np.ndarray, so: np.ndarray,
                       grid_h: int, grid_w: int) -> np.ndarray:
    """Tile rotation XYZW + scaleOpacity XY → (2*grid_h, 3*grid_w) grayscale.

    Layout:
      Row 0: [rot_X | rot_Y | rot_Z]
      Row 1: [rot_W | so_X  | so_Y ]
    """
    row0 = _tile_horizontal([rot[:, 0], rot[:, 1], rot[:, 2]], grid_h, grid_w)
    row1 = _tile_horizontal([rot[:, 3], so[:, 0], so[:, 1]], grid_h, grid_w)
    return np.concatenate([row0, row1], axis=0)


def tile_stream_appearance(so_zw: np.ndarray, sh_dc: np.ndarray,
                           grid_h: int, grid_w: int) -> np.ndarray:
    """Tile scaleOpacity ZW + shDC RGB → (grid_h, 5*grid_w) grayscale.

    Layout:
      Row 0: [so_Z | so_W | shDC_R | shDC_G | shDC_B]
    """
    return _tile_horizontal(
        [so_zw[:, 0], so_zw[:, 1], sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]],
        grid_h, grid_w,
    )


# --- ffmpeg H.265 Encoding ---

def start_encoder(width: int, height: int, fps: int, crf: int,
                  output_path: str) -> subprocess.Popen:
    """Start ffmpeg H.265 encoder reading raw YUV420p frames from stdin.

    Input: YUV420p raw frames via stdin (Y=data, U=V=128 neutral).
    We feed YUV420p directly instead of grayscale to AVOID ffmpeg's
    gray→yuv420p conversion which maps 0-255 to limited range 16-235,
    destroying ~14% of quantization precision.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-tag:v", "hev1",
        "-x265-params", f"keyint={fps}:min-keyint={fps}:log-level=error:no-sao=1:deblock=0,0",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


# Pre-allocate neutral UV planes (reused across frames to avoid allocation)
_uv_cache: dict[tuple[int, int], bytes] = {}


def write_frame(proc: subprocess.Popen, frame: np.ndarray):
    """Write one frame as YUV420p: Y=data, U=V=128 (neutral chroma)."""
    h, w = frame.shape
    # Y plane: our actual data
    proc.stdin.write(frame.tobytes())
    # U + V planes: half resolution, filled with 128 (neutral)
    uv_key = (h // 2, w // 2)
    if uv_key not in _uv_cache:
        uv_size = (h // 2) * (w // 2)
        _uv_cache[uv_key] = b'\x80' * uv_size  # 128 = 0x80
    uv_bytes = _uv_cache[uv_key]
    proc.stdin.write(uv_bytes)  # U plane
    proc.stdin.write(uv_bytes)  # V plane


def finish_encoder(proc: subprocess.Popen):
    """Close stdin and wait for ffmpeg to finish. Returns stderr output."""
    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr.decode()}")
    return stderr.decode()


# --- Manifest Generation ---

def write_manifest(output_dir: str, sequence_name: str, frame_count: int,
                   fps: int, grid_w: int, grid_h: int,
                   gaussian_count: int, bounds: dict):
    """Write manifest.json with quantization bounds and stream metadata."""
    def _to_list(arr):
        return [round(float(x), 6) for x in arr]

    manifest = {
        "version": 1,
        "format": "4dgs-h265",
        "sequenceName": sequence_name,
        "frameCount": frame_count,
        "targetFPS": fps,
        "duration": round(frame_count / fps, 2),
        "shDegree": 0,
        "gridWidth": grid_w,
        "gridHeight": grid_h,
        "gaussianCount": gaussian_count,
        "requiredCodec": "hev1.1.6.L150.B0",
        "coordinateSpace": "colmap",
        "quaternionOrder": "wxyz",
        "quantization": {
            "position": {
                "precision": "uint16",
                "min": _to_list(bounds["position"]["min"]),
                "max": _to_list(bounds["position"]["max"]),
            },
            "rotation": {
                "precision": "uint8",
                "min": _to_list(bounds["rotation"]["min"]),
                "max": _to_list(bounds["rotation"]["max"]),
            },
            "scaleOpacity": {
                "precision": "uint8",
                "min": _to_list(bounds["scale_opacity"]["min"]),
                "max": _to_list(bounds["scale_opacity"]["max"]),
            },
            "shDC": {
                "precision": "uint8",
                "min": _to_list(bounds["sh_dc"]["min"]),
                "max": _to_list(bounds["sh_dc"]["max"]),
            },
        },
        "streams": {
            "position": {
                "file": "stream_position.mp4",
                "width": grid_w * 3,
                "height": grid_h * 2,
                "channels": 6,
            },
            "motion": {
                "file": "stream_motion.mp4",
                "width": grid_w * 3,
                "height": grid_h * 2,
                "channels": 6,
            },
            "appearance": {
                "file": "stream_appearance.mp4",
                "width": grid_w * 5,
                "height": grid_h,
                "channels": 5,
            },
        },
    }

    out_path = Path(output_dir) / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)


# --- Main Pipeline ---

H265_LEVEL_5_MAX_PIXELS = 8_912_896


def process_frame(ply_path: Path, bounds: dict, grid_h: int, grid_w: int):
    """Process one PLY frame -> 3 tiled grayscale stream images.

    Returns:
        (stream_pos, stream_mot, stream_app) -- each a uint8 ndarray
    """
    gaussians = load_gaussian_ply(str(ply_path))
    activated = activate_gaussians(gaussians)
    n = len(activated["position"])

    # Morton sort for spatial coherence + temporal consistency
    sorted_indices, _, _ = sort_3d_morton_order(activated["position"])

    # Pad to grid size with appropriate default values
    pad_n = grid_h * grid_w

    def _sort_and_pad(arr, pad_value=0.0):
        sorted_arr = arr[sorted_indices]
        if len(sorted_arr) < pad_n:
            padded = np.full((pad_n, arr.shape[1]), pad_value, dtype=arr.dtype)
            padded[:len(sorted_arr)] = sorted_arr
            return padded
        return sorted_arr[:pad_n]

    pos = _sort_and_pad(activated["position"])
    rot_sorted = activated["rotation"][sorted_indices]
    # Pad rotation with identity quaternion (W=1, X=0, Y=0, Z=0)
    if len(rot_sorted) < pad_n:
        rot = np.zeros((pad_n, 4), dtype=rot_sorted.dtype)
        rot[:len(rot_sorted)] = rot_sorted
        rot[len(rot_sorted):, 0] = 1.0  # W=1 for identity
    else:
        rot = rot_sorted[:pad_n]
    so = _sort_and_pad(activated["scale_opacity"])
    sh_dc = _sort_and_pad(activated["sh_dc"])

    # Quantize
    pos_high, pos_low = quantize_position(pos, bounds["position"]["min"], bounds["position"]["max"])
    rot_q = quantize_uint8(rot, bounds["rotation"]["min"], bounds["rotation"]["max"])
    so_q = quantize_uint8(so, bounds["scale_opacity"]["min"], bounds["scale_opacity"]["max"])
    sh_q = quantize_uint8(sh_dc, bounds["sh_dc"]["min"], bounds["sh_dc"]["max"])

    # Tile
    stream_pos = tile_stream_position(pos_high, pos_low, grid_h, grid_w)
    stream_mot = tile_stream_motion(rot_q, so_q, grid_h, grid_w)
    stream_app = tile_stream_appearance(so_q[:, 2:4], sh_q, grid_h, grid_w)

    return stream_pos, stream_mot, stream_app


def convert_ply_to_h265(ply_dir: str, output_dir: str, fps: int = 24,
                        crf_position: int = 18, crf_motion: int = 22,
                        crf_appearance: int = 24,
                        callback=None):
    """Convert PLY sequence to 3 H.265 MP4 streams + manifest.json."""
    ply_dir = Path(ply_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover PLY files
    ply_paths = sorted(ply_dir.glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No PLY files found in {ply_dir}")
    frame_count = len(ply_paths)

    # Determine grid size from first frame
    sample = load_gaussian_ply(str(ply_paths[0]))
    gaussian_count = len(sample["position"])
    grid_size = math.ceil(math.sqrt(gaussian_count))
    grid_size = ((grid_size + 15) // 16) * 16
    grid_h = grid_w = grid_size

    # Validate against H.265 Level 5.0
    pos_w, pos_h = grid_w * 3, grid_h * 2
    mot_w, mot_h = grid_w * 3, grid_h * 2
    app_w, app_h = grid_w * 5, grid_h
    max_pixels = max(pos_w * pos_h, mot_w * mot_h, app_w * app_h)
    if max_pixels > H265_LEVEL_5_MAX_PIXELS:
        raise ValueError(
            f"Stream dimensions ({max_pixels} pixels) exceed H.265 Level 5.0 "
            f"limit ({H265_LEVEL_5_MAX_PIXELS}). Reduce gaussian count or grid size."
        )

    # Pass 1: Compute global quantization bounds
    if callback:
        callback(-1, frame_count)
    bounds = compute_global_bounds(ply_paths)

    # Start 3 parallel ffmpeg encoders
    enc_pos = start_encoder(pos_w, pos_h, fps, crf_position,
                            str(output_dir / "stream_position.mp4"))
    enc_mot = start_encoder(mot_w, mot_h, fps, crf_motion,
                            str(output_dir / "stream_motion.mp4"))
    enc_app = start_encoder(app_w, app_h, fps, crf_appearance,
                            str(output_dir / "stream_appearance.mp4"))

    # Pass 2: Process frames sequentially
    encoders = [enc_pos, enc_mot, enc_app]
    try:
        for i, ply_path in enumerate(ply_paths):
            stream_pos, stream_mot, stream_app = process_frame(
                ply_path, bounds, grid_h, grid_w
            )
            write_frame(enc_pos, stream_pos)
            write_frame(enc_mot, stream_mot)
            write_frame(enc_app, stream_app)
            if callback:
                callback(i, frame_count)
    finally:
        errors = []
        for enc in encoders:
            try:
                finish_encoder(enc)
            except RuntimeError as e:
                errors.append(e)
        if errors:
            raise RuntimeError(f"ffmpeg encoding errors: {errors}")

    # Write manifest
    write_manifest(
        output_dir=str(output_dir),
        sequence_name=ply_dir.name,
        frame_count=frame_count,
        fps=fps,
        grid_w=grid_w, grid_h=grid_h,
        gaussian_count=gaussian_count,
        bounds=bounds,
    )

    return {
        "frame_count": frame_count,
        "grid_size": grid_size,
        "gaussian_count": gaussian_count,
        "output_dir": str(output_dir),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert PLY sequence to H.265 streams")
    parser.add_argument("--input", required=True, help="Directory with frame_NNNN.ply files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--crf-position", type=int, default=18)
    parser.add_argument("--crf-motion", type=int, default=22)
    parser.add_argument("--crf-appearance", type=int, default=24)
    args = parser.parse_args()

    def progress(i, total):
        if i == -1:
            print(f"Computing quantization bounds from {total} frames...")
        else:
            print(f"  Frame {i+1}/{total}", end="\r")

    print(f"Converting PLY sequence: {args.input}")
    result = convert_ply_to_h265(
        ply_dir=args.input,
        output_dir=args.output,
        fps=args.fps,
        crf_position=args.crf_position,
        crf_motion=args.crf_motion,
        crf_appearance=args.crf_appearance,
        callback=progress,
    )
    print(f"\nDone! {result['frame_count']} frames encoded to {result['output_dir']}")
    print(f"  Grid: {result['grid_size']}x{result['grid_size']}, Gaussians: {result['gaussian_count']}")

    # Report file sizes
    out = Path(result["output_dir"])
    total = 0
    for mp4 in sorted(out.glob("*.mp4")):
        sz = mp4.stat().st_size
        total += sz
        print(f"  {mp4.name}: {sz / 1024 / 1024:.1f} MB")
    print(f"  Total: {total / 1024 / 1024:.1f} MB")
    if result['frame_count'] > 0:
        print(f"  Bitrate: {total * 8 * args.fps / result['frame_count'] / 1e6:.1f} Mbps")


if __name__ == "__main__":
    main()
