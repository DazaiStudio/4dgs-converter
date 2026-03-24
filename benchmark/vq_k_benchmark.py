"""VQ K-value benchmark for GSD v2 attributes.

Tests different VQ K values (256, 512, 1024, 2048, 4096) for rotation, scale,
and sh_dc on real SHARP data. Measures compression size, reconstruction error,
and encode time for each combination.
"""

import math
import os
import sys
import time

import lz4.block
import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order

PLY_FOLDER = r"D:\4dgs-data\nature\ANIMAL_camel\ply"
FRAME_INDICES = [0, 1, 2]  # 3 frames
K_VALUES = [256, 512, 1024, 2048, 4096]


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def vq_encode(data: np.ndarray, k: int, n_sample: int = 50000):
    """Vector quantize data. Returns (codebook, indices, encode_time_s)."""
    n = len(data)
    sample_idx = np.random.choice(n, min(n_sample, n), replace=False)
    t0 = time.perf_counter()
    km = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, random_state=42)
    km.fit(data[sample_idx])
    indices = km.predict(data)
    t1 = time.perf_counter()
    codebook = km.cluster_centers_.astype(np.float32)
    # uint8 for K<=256, uint16 otherwise
    if k <= 256:
        indices = indices.astype(np.uint8)
    else:
        indices = indices.astype(np.uint16)
    return codebook, indices, t1 - t0


def compressed_size(codebook: np.ndarray, indices: np.ndarray) -> int:
    """Total bytes: codebook + LZ4-compressed indices."""
    codebook_bytes = codebook.nbytes  # K * dim * 4 (float32)
    idx_compressed = compress_lz4(indices.tobytes())
    return codebook_bytes + len(idx_compressed)


def rotation_angular_error(original: np.ndarray, reconstructed: np.ndarray):
    """Angular error in degrees between quaternions. Returns (mean, p50, p99, max)."""
    o = original / (np.linalg.norm(original, axis=1, keepdims=True) + 1e-10)
    r = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-10)
    dot = np.abs(np.sum(o * r, axis=1))
    angles = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
    return float(np.mean(angles)), float(np.median(angles)), float(np.percentile(angles, 99)), float(np.max(angles))


def mean_abs_error(original: np.ndarray, reconstructed: np.ndarray):
    """Mean absolute error. Returns (mean, p50, p99, max)."""
    diff = np.abs(original - reconstructed)
    per_row = np.mean(diff, axis=1) if diff.ndim > 1 else diff
    return float(np.mean(per_row)), float(np.median(per_row)), float(np.percentile(per_row, 99)), float(np.max(per_row))


def load_frames():
    """Load 3 frames, return list of sorted attribute dicts."""
    ply_files = sorted([
        os.path.join(PLY_FOLDER, f) for f in os.listdir(PLY_FOLDER)
        if f.endswith(".ply")
    ])
    print(f"Found {len(ply_files)} PLY files in {PLY_FOLDER}")
    print(f"Loading frames: {FRAME_INDICES}\n")

    frames = []
    for fi in FRAME_INDICES:
        g = load_gaussian_ply(ply_files[fi])
        n = len(g["position"])
        sorted_idx, _, _ = sort_3d_morton_order(g["position"])

        # Normalize rotation
        rot = g["rotation"][sorted_idx].astype(np.float32)
        rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
        rot_norm = np.where(rot_norm == 0, 1.0, rot_norm)
        rot = rot / rot_norm

        frames.append({
            "rotation": rot,                                        # (N, 4)
            "scale": g["scale"][sorted_idx].astype(np.float32),     # (N, 3) pre-exp
            "sh_dc": g["sh_dc"][sorted_idx].astype(np.float32),     # (N, 3)
            "n_gaussians": n,
        })
        print(f"  Frame {fi}: {n:,} gaussians")
    print()
    return frames


def benchmark_attribute(frames, attr_name, k_values, error_fn):
    """Run VQ benchmark for one attribute across all K values.

    Returns list of result dicts, one per K value (averaged over frames).
    """
    results = []
    for k in k_values:
        frame_sizes = []
        frame_errors = []
        frame_times = []
        frame_raw_sizes = []

        for frame in frames:
            data = frame[attr_name]
            raw_size = data.nbytes

            codebook, indices, enc_time = vq_encode(data, k)
            reconstructed = codebook[indices]
            size = compressed_size(codebook, indices)

            err = error_fn(data, reconstructed)

            frame_sizes.append(size)
            frame_errors.append(err)
            frame_times.append(enc_time)
            frame_raw_sizes.append(raw_size)

        avg_size = np.mean(frame_sizes)
        avg_raw = np.mean(frame_raw_sizes)
        avg_time = np.mean(frame_times)
        # Average each error metric across frames
        avg_err = tuple(np.mean([e[i] for e in frame_errors]) for i in range(4))

        idx_bytes = 1 if k <= 256 else 2
        n = frames[0]["n_gaussians"]
        dim = frames[0][attr_name].shape[1]
        codebook_size = k * dim * 4  # float32
        idx_raw = n * idx_bytes
        idx_compressed_avg = avg_size - codebook_size

        results.append({
            "k": k,
            "avg_size_bytes": avg_size,
            "avg_raw_bytes": avg_raw,
            "ratio_pct": avg_size / avg_raw * 100,
            "codebook_bytes": codebook_size,
            "idx_compressed_bytes": idx_compressed_avg,
            "idx_dtype": f"uint{idx_bytes*8}",
            "avg_encode_s": avg_time,
            "err_mean": avg_err[0],
            "err_p50": avg_err[1],
            "err_p99": avg_err[2],
            "err_max": avg_err[3],
        })

    return results


def print_table(attr_name, results, error_unit):
    """Print a formatted table for one attribute."""
    print(f"\n{'='*90}")
    print(f"  {attr_name.upper()}  (error unit: {error_unit})")
    print(f"{'='*90}")
    print(f"  {'K':>5}  {'Size':>10}  {'Ratio':>7}  {'CB':>8}  {'Idx(LZ4)':>10}  "
          f"{'Idx Type':>8}  {'Enc(s)':>7}  {'Mean':>8}  {'P50':>8}  {'P99':>8}  {'Max':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*10}  "
          f"{'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        size_str = f"{r['avg_size_bytes']/1024:.1f} KB"
        ratio_str = f"{r['ratio_pct']:.1f}%"
        cb_str = f"{r['codebook_bytes']/1024:.1f} KB"
        idx_str = f"{r['idx_compressed_bytes']/1024:.1f} KB"
        print(f"  {r['k']:>5}  {size_str:>10}  {ratio_str:>7}  {cb_str:>8}  {idx_str:>10}  "
              f"{r['idx_dtype']:>8}  {r['avg_encode_s']:>7.2f}  "
              f"{r['err_mean']:>8.4f}  {r['err_p50']:>8.4f}  {r['err_p99']:>8.4f}  {r['err_max']:>8.4f}")

    # Highlight the sweet spot
    print()
    baseline = results[0]  # K=256
    for r in results[1:]:
        size_delta = (r['avg_size_bytes'] - baseline['avg_size_bytes']) / baseline['avg_size_bytes'] * 100
        err_delta = (r['err_mean'] - baseline['err_mean']) / baseline['err_mean'] * 100 if baseline['err_mean'] > 0 else 0
        print(f"  K={r['k']:>4} vs K=256: size {size_delta:+.1f}%, error {err_delta:+.1f}%")


def main():
    np.random.seed(42)
    print("VQ K-Value Benchmark for GSD v2")
    print(f"K values: {K_VALUES}")
    print(f"Frames: {len(FRAME_INDICES)}")
    print()

    frames = load_frames()
    n = frames[0]["n_gaussians"]
    print(f"Gaussians per frame: {n:,}")
    print(f"Raw sizes per frame:")
    print(f"  rotation (N,4) fp32: {n*4*4/1024:.1f} KB")
    print(f"  scale    (N,3) fp32: {n*3*4/1024:.1f} KB")
    print(f"  sh_dc    (N,3) fp32: {n*3*4/1024:.1f} KB")
    print()

    # Rotation
    print("Benchmarking ROTATION (quaternion, 4D)...")
    rot_results = benchmark_attribute(frames, "rotation", K_VALUES, rotation_angular_error)

    # Scale
    print("Benchmarking SCALE (pre-exp, 3D)...")
    scale_results = benchmark_attribute(frames, "scale", K_VALUES, mean_abs_error)

    # SH DC
    print("Benchmarking SH_DC (3D)...")
    sh_results = benchmark_attribute(frames, "sh_dc", K_VALUES, mean_abs_error)

    # Print results
    print_table("rotation", rot_results, "degrees")
    print_table("scale", scale_results, "abs (pre-exp)")
    print_table("sh_dc", sh_results, "abs")

    # Combined summary
    print(f"\n{'='*90}")
    print(f"  COMBINED SUMMARY (all 3 attributes, average per frame)")
    print(f"{'='*90}")
    print(f"  {'K':>5}  {'Rot Size':>10}  {'Sca Size':>10}  {'SH Size':>10}  {'Total':>10}  "
          f"{'Rot Err':>8}  {'Sca Err':>8}  {'SH Err':>8}  {'Enc Time':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for i, k in enumerate(K_VALUES):
        rr = rot_results[i]
        sr = scale_results[i]
        shr = sh_results[i]
        total = rr['avg_size_bytes'] + sr['avg_size_bytes'] + shr['avg_size_bytes']
        total_time = rr['avg_encode_s'] + sr['avg_encode_s'] + shr['avg_encode_s']
        print(f"  {k:>5}  {rr['avg_size_bytes']/1024:>8.1f}KB  {sr['avg_size_bytes']/1024:>8.1f}KB  "
              f"{shr['avg_size_bytes']/1024:>8.1f}KB  {total/1024:>8.1f}KB  "
              f"{rr['err_mean']:>7.3f}d  {sr['err_mean']:>8.4f}  {shr['err_mean']:>8.4f}  "
              f"{total_time:>7.2f}s")

    # Raw total for comparison
    raw_total = n * (4*4 + 3*4 + 3*4)  # rot + scale + sh_dc
    print(f"\n  Raw total (rot+scale+sh): {raw_total/1024:.1f} KB")
    for i, k in enumerate(K_VALUES):
        total = rot_results[i]['avg_size_bytes'] + scale_results[i]['avg_size_bytes'] + sh_results[i]['avg_size_bytes']
        print(f"  K={k:>4}: {total/1024:.1f} KB = {total/raw_total*100:.1f}% of raw")

    print("\nDone.")


if __name__ == "__main__":
    main()
