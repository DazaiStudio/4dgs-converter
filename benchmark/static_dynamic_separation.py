"""Static/Dynamic separation prototype.

Classifies scene regions as static vs dynamic by comparing
per-voxel gaussian statistics across frames from ml-sharp.

Pipeline:
1. Define voxel grid over scene bounding box
2. For each frame, assign gaussians to voxels
3. Compute per-voxel statistics (mean position, mean SH, occupancy)
4. Compare variance across frames → low variance = static
5. Measure static ratio and estimate compression impact
"""

import os
import sys
import time

import numpy as np
import lz4.block

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.utils.ply_reader import load_gaussian_ply


def pixel_shuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bpp).T.reshape(-1).tobytes()


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def load_frame_compact(ply_path: str) -> dict:
    """Load only what we need: position, sh_dc, opacity, scale, rotation."""
    g = load_gaussian_ply(ply_path)
    return {
        "position": g["position"].astype(np.float32),
        "sh_dc": g["sh_dc"].astype(np.float32),
        "opacity": (1.0 / (1.0 + np.exp(-g["opacity"]))).astype(np.float32),  # sigmoid
        "scale": np.exp(g["scale"]).astype(np.float32),  # activated
        "rotation": g["rotation"].astype(np.float32),
    }


def compute_scene_bounds(frames: list[dict], padding: float = 0.1) -> tuple:
    """Compute scene bounding box from all frames."""
    all_mins = []
    all_maxs = []
    for f in frames:
        pos = f["position"]
        all_mins.append(pos.min(axis=0))
        all_maxs.append(pos.max(axis=0))
    scene_min = np.min(all_mins, axis=0) - padding
    scene_max = np.max(all_maxs, axis=0) + padding
    return scene_min, scene_max


def assign_to_voxels(positions: np.ndarray, scene_min: np.ndarray,
                     voxel_size: float) -> np.ndarray:
    """Assign each gaussian to a voxel. Returns (N, 3) int32 voxel indices."""
    return ((positions - scene_min) / voxel_size).astype(np.int32)


def build_voxel_stats(frame: dict, scene_min: np.ndarray,
                      voxel_size: float, grid_shape: tuple) -> dict:
    """For each occupied voxel, compute aggregate statistics."""
    voxel_idx = assign_to_voxels(frame["position"], scene_min, voxel_size)

    # Clamp to grid bounds
    for d in range(3):
        voxel_idx[:, d] = np.clip(voxel_idx[:, d], 0, grid_shape[d] - 1)

    # Flatten voxel index to 1D
    vi = voxel_idx.astype(np.int64)
    flat_idx = (vi[:, 0] * grid_shape[1] * grid_shape[2] +
                vi[:, 1] * grid_shape[2] +
                vi[:, 2]).astype(np.intp)

    n_voxels = int(np.prod(np.array(grid_shape, dtype=np.int64)))

    # Per-voxel: count, mean position, mean sh_dc, mean opacity
    counts = np.bincount(flat_idx, minlength=n_voxels)

    # Weighted by opacity for more meaningful stats
    opacity = frame["opacity"]

    mean_pos = np.zeros((n_voxels, 3), dtype=np.float64)
    mean_sh = np.zeros((n_voxels, 3), dtype=np.float64)
    mean_opacity = np.zeros(n_voxels, dtype=np.float64)

    for d in range(3):
        mean_pos[:, d] = np.bincount(flat_idx, weights=frame["position"][:, d] * opacity,
                                     minlength=n_voxels)
        mean_sh[:, d] = np.bincount(flat_idx, weights=frame["sh_dc"][:, d] * opacity,
                                    minlength=n_voxels)
    mean_opacity = np.bincount(flat_idx, weights=opacity, minlength=n_voxels)

    # Normalize by total opacity weight
    mask = mean_opacity > 1e-6
    for d in range(3):
        mean_pos[mask, d] /= mean_opacity[mask]
        mean_sh[mask, d] /= mean_opacity[mask]

    return {
        "counts": counts,
        "mean_pos": mean_pos.astype(np.float32),
        "mean_sh": mean_sh.astype(np.float32),
        "mean_opacity": mean_opacity.astype(np.float32),
        "flat_idx": flat_idx,
        "mask": mask,  # occupied voxels
    }


def classify_voxels(all_stats: list[dict], grid_shape: tuple,
                    pos_threshold: float = 0.005,
                    sh_threshold: float = 0.05,
                    occupancy_threshold: float = 0.3) -> dict:
    """Classify voxels as static vs dynamic based on temporal variance."""
    n_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
    n_frames = len(all_stats)

    # Stack per-voxel stats across frames
    all_counts = np.stack([s["counts"] for s in all_stats])  # (F, V)
    all_pos = np.stack([s["mean_pos"] for s in all_stats])    # (F, V, 3)
    all_sh = np.stack([s["mean_sh"] for s in all_stats])      # (F, V, 3)
    all_opacity = np.stack([s["mean_opacity"] for s in all_stats])  # (F, V)

    # Voxels that are occupied in most frames
    occupied_rate = np.mean(all_counts > 0, axis=0)  # (V,)
    consistently_occupied = occupied_rate >= occupancy_threshold

    # Temporal variance of position (only for consistently occupied)
    pos_var = np.zeros(n_voxels, dtype=np.float32)
    sh_var = np.zeros(n_voxels, dtype=np.float32)
    opacity_var = np.zeros(n_voxels, dtype=np.float32)
    count_var = np.zeros(n_voxels, dtype=np.float32)

    co_mask = consistently_occupied
    if co_mask.sum() > 0:
        # Position variance: mean of per-axis std across frames
        pos_std = np.std(all_pos[:, co_mask, :], axis=0)  # (occupied, 3)
        pos_var[co_mask] = np.mean(pos_std, axis=1)

        # SH variance
        sh_std = np.std(all_sh[:, co_mask, :], axis=0)
        sh_var[co_mask] = np.mean(sh_std, axis=1)

        # Opacity variance
        opacity_var[co_mask] = np.std(all_opacity[:, co_mask], axis=0)

        # Count variance (normalized)
        mean_counts = np.mean(all_counts[:, co_mask], axis=0)
        mean_counts = np.where(mean_counts == 0, 1, mean_counts)
        count_var[co_mask] = np.std(all_counts[:, co_mask], axis=0) / mean_counts

    # Classify: static = low variance across all metrics
    is_static = (co_mask &
                 (pos_var < pos_threshold) &
                 (sh_var < sh_threshold))

    is_dynamic = co_mask & ~is_static
    is_empty = ~co_mask

    return {
        "is_static": is_static,
        "is_dynamic": is_dynamic,
        "is_empty": is_empty,
        "pos_var": pos_var,
        "sh_var": sh_var,
        "opacity_var": opacity_var,
        "count_var": count_var,
        "occupied_rate": occupied_rate,
        "consistently_occupied": consistently_occupied,
    }


def estimate_compression(frames: list[dict], classification: dict,
                         scene_min: np.ndarray, voxel_size: float,
                         grid_shape: tuple) -> dict:
    """Estimate compression with static/dynamic separation."""
    is_static = classification["is_static"]
    is_dynamic = classification["is_dynamic"]

    results = []

    for fi, frame in enumerate(frames):
        voxel_idx = assign_to_voxels(frame["position"], scene_min, voxel_size)
        for d in range(3):
            voxel_idx[:, d] = np.clip(voxel_idx[:, d], 0, grid_shape[d] - 1)
        vi = voxel_idx.astype(np.int64)
        flat_idx = (vi[:, 0] * grid_shape[1] * grid_shape[2] +
                    vi[:, 1] * grid_shape[2] +
                    vi[:, 2]).astype(np.intp)

        # Classify each gaussian
        g_static = is_static[flat_idx]
        g_dynamic = is_dynamic[flat_idx]

        n_total = len(frame["position"])
        n_static = g_static.sum()
        n_dynamic = g_dynamic.sum()
        n_empty = n_total - n_static - n_dynamic  # in rarely-occupied voxels

        # Compress dynamic gaussians only (static stored once)
        if n_dynamic > 0:
            dyn_pos = frame["position"][g_dynamic].astype(np.float16)
            dyn_rot = frame["rotation"][g_dynamic].astype(np.float16)
            dyn_scale = frame["scale"][g_dynamic].astype(np.float16)
            dyn_sh = frame["sh_dc"][g_dynamic].astype(np.float16)
            dyn_opacity = frame["opacity"][g_dynamic].astype(np.float16)

            # Pack and compress
            dyn_blob = b"".join([
                pixel_shuffle(dyn_pos.tobytes(), dyn_pos.shape[1] * 2),
                pixel_shuffle(dyn_rot.tobytes(), dyn_rot.shape[1] * 2),
                pixel_shuffle(dyn_scale.tobytes(), dyn_scale.shape[1] * 2),
                pixel_shuffle(dyn_sh.tobytes(), dyn_sh.shape[1] * 2),
                pixel_shuffle(dyn_opacity.reshape(-1, 1).tobytes(), 2),
            ])
            dyn_compressed = len(compress_lz4(dyn_blob))

            # Also need voxel assignment indices for dynamic gaussians
            dyn_voxel_idx = flat_idx[g_dynamic].astype(np.uint32).tobytes()
            dyn_idx_compressed = len(compress_lz4(pixel_shuffle(dyn_voxel_idx, 4)))
        else:
            dyn_compressed = 0
            dyn_idx_compressed = 0

        # Gaussians in rarely-occupied voxels - also need to store per-frame
        if n_empty > 0:
            emp_pos = frame["position"][~g_static & ~g_dynamic].astype(np.float16)
            emp_rot = frame["rotation"][~g_static & ~g_dynamic].astype(np.float16)
            emp_scale = frame["scale"][~g_static & ~g_dynamic].astype(np.float16)
            emp_sh = frame["sh_dc"][~g_static & ~g_dynamic].astype(np.float16)
            emp_opacity = frame["opacity"][~g_static & ~g_dynamic].astype(np.float16)

            emp_blob = b"".join([
                pixel_shuffle(emp_pos.tobytes(), emp_pos.shape[1] * 2),
                pixel_shuffle(emp_rot.tobytes(), emp_rot.shape[1] * 2),
                pixel_shuffle(emp_scale.tobytes(), emp_scale.shape[1] * 2),
                pixel_shuffle(emp_sh.tobytes(), emp_sh.shape[1] * 2),
                pixel_shuffle(emp_opacity.reshape(-1, 1).tobytes(), 2),
            ])
            emp_compressed = len(compress_lz4(emp_blob))
        else:
            emp_compressed = 0

        per_frame_size = dyn_compressed + dyn_idx_compressed + emp_compressed

        results.append({
            "frame": fi,
            "n_total": n_total,
            "n_static": int(n_static),
            "n_dynamic": int(n_dynamic),
            "n_empty_voxel": int(n_empty),
            "static_pct": n_static / n_total * 100,
            "dynamic_pct": n_dynamic / n_total * 100,
            "per_frame_compressed_mb": per_frame_size / 1e6,
        })

    return results


def run(ply_folder: str, num_frames: int = 20, voxel_sizes=None):
    if voxel_sizes is None:
        voxel_sizes = [0.5, 1.0, 2.0]

    ply_files = sorted([
        os.path.join(ply_folder, f) for f in os.listdir(ply_folder)
        if f.endswith(".ply")
    ])
    total_ply = len(ply_files)
    print(f"Found {total_ply} PLY files")

    # Load frames (sample evenly for better temporal coverage)
    step = max(1, total_ply // num_frames)
    sample_indices = list(range(0, total_ply, step))[:num_frames]
    print(f"Sampling {len(sample_indices)} frames: {sample_indices[:5]}...{sample_indices[-3:]}")

    frames = []
    for i, idx in enumerate(sample_indices):
        t0 = time.perf_counter()
        f = load_frame_compact(ply_files[idx])
        t1 = time.perf_counter()
        frames.append(f)
        if (i + 1) % 5 == 0:
            print(f"  Loaded {i+1}/{len(sample_indices)} ({t1-t0:.1f}s each)")
    print()

    # Compute full-frame baseline
    print("=" * 80)
    print("BASELINE: full frame compression (fp16 + shuffle + LZ4)")
    print("=" * 80)

    baseline_sizes = []
    for f in frames[:3]:
        blob = b"".join([
            pixel_shuffle(f["position"].astype(np.float16).tobytes(), 6),
            pixel_shuffle(f["rotation"].astype(np.float16).tobytes(), 8),
            pixel_shuffle(np.exp(f["scale"]).astype(np.float16).tobytes() if False else f["scale"].astype(np.float16).tobytes(), 6),
            pixel_shuffle(f["sh_dc"].astype(np.float16).tobytes(), 6),
            pixel_shuffle(f["opacity"].reshape(-1, 1).astype(np.float16).tobytes(), 2),
        ])
        comp = len(compress_lz4(blob))
        baseline_sizes.append(comp)
    baseline_avg = np.mean(baseline_sizes)
    print(f"  Average frame: {baseline_avg/1e6:.2f} MB (fp16 all, shuffle+LZ4)")
    print()

    # Scene bounds
    scene_min, scene_max = compute_scene_bounds(frames)
    scene_size = scene_max - scene_min
    print(f"Scene bounds: min={scene_min}, max={scene_max}")
    print(f"Scene size: {scene_size}")
    print()

    # Test different voxel sizes
    for voxel_size in voxel_sizes:
        grid_shape = tuple(np.ceil(scene_size / voxel_size).astype(np.int64))
        n_voxels = int(np.prod(np.array(grid_shape, dtype=np.int64)))

        print("=" * 80)
        print(f"VOXEL SIZE = {voxel_size} → Grid {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]} = {n_voxels:,} voxels")
        print("=" * 80)

        # Build per-frame voxel stats
        print("Building voxel stats...")
        t0 = time.perf_counter()
        all_stats = []
        for f in frames:
            stats = build_voxel_stats(f, scene_min, voxel_size, grid_shape)
            all_stats.append(stats)
        t1 = time.perf_counter()
        print(f"  Done in {t1-t0:.1f}s")

        # Classify
        classification = classify_voxels(all_stats, grid_shape)

        n_static = classification["is_static"].sum()
        n_dynamic = classification["is_dynamic"].sum()
        n_empty = classification["is_empty"].sum()
        n_occupied = n_static + n_dynamic

        print(f"\n  Voxel classification:")
        print(f"    Static:  {n_static:>8,} ({n_static/n_voxels*100:.1f}% of all, {n_static/max(n_occupied,1)*100:.1f}% of occupied)")
        print(f"    Dynamic: {n_dynamic:>8,} ({n_dynamic/n_voxels*100:.1f}% of all, {n_dynamic/max(n_occupied,1)*100:.1f}% of occupied)")
        print(f"    Empty:   {n_empty:>8,} ({n_empty/n_voxels*100:.1f}%)")

        # Estimate compression
        print(f"\n  Per-frame gaussian classification:")
        comp_results = estimate_compression(frames, classification, scene_min, voxel_size, grid_shape)

        for r in comp_results[:5]:
            print(f"    Frame {r['frame']:>3}: static={r['static_pct']:.1f}% dynamic={r['dynamic_pct']:.1f}% "
                  f"per_frame={r['per_frame_compressed_mb']:.2f} MB")
        if len(comp_results) > 5:
            print(f"    ...")

        # Compute static layer size (stored once)
        # Use frame 0's static gaussians as the canonical static layer
        f0 = frames[0]
        voxel_idx0 = assign_to_voxels(f0["position"], scene_min, voxel_size)
        for d in range(3):
            voxel_idx0[:, d] = np.clip(voxel_idx0[:, d], 0, grid_shape[d] - 1)
        vi0 = voxel_idx0.astype(np.int64)
        flat_idx0 = (vi0[:, 0] * grid_shape[1] * grid_shape[2] +
                     vi0[:, 1] * grid_shape[2] +
                     vi0[:, 2]).astype(np.intp)
        static_mask0 = classification["is_static"][flat_idx0]

        static_blob = b"".join([
            pixel_shuffle(f0["position"][static_mask0].astype(np.float16).tobytes(), 6),
            pixel_shuffle(f0["rotation"][static_mask0].astype(np.float16).tobytes(), 8),
            pixel_shuffle(f0["scale"][static_mask0].astype(np.float16).tobytes(), 6),
            pixel_shuffle(f0["sh_dc"][static_mask0].astype(np.float16).tobytes(), 6),
            pixel_shuffle(f0["opacity"][static_mask0].reshape(-1, 1).astype(np.float16).tobytes(), 2),
        ])
        static_compressed = len(compress_lz4(static_blob))

        avg_per_frame = np.mean([r["per_frame_compressed_mb"] for r in comp_results])
        avg_static_pct = np.mean([r["static_pct"] for r in comp_results])

        total_480 = static_compressed / 1e6 + 480 * avg_per_frame
        baseline_480 = 480 * baseline_avg / 1e6

        print(f"\n  ESTIMATE (480 frames):")
        print(f"    Static layer:     {static_compressed/1e6:.2f} MB (stored once, {static_mask0.sum():,} gaussians)")
        print(f"    Dynamic per-frame: {avg_per_frame:.2f} MB avg ({100-avg_static_pct:.1f}% of gaussians)")
        print(f"    Total:            {total_480/1024:.2f} GB")
        print(f"    Baseline:         {baseline_480/1024:.2f} GB")
        print(f"    Savings:          {(1-total_480/baseline_480)*100:.1f}%")

        # Variance distribution
        occ = classification["consistently_occupied"]
        if occ.sum() > 0:
            pv = classification["pos_var"][occ]
            sv = classification["sh_var"][occ]
            print(f"\n  Variance distribution (occupied voxels):")
            print(f"    Position var: p10={np.percentile(pv,10):.5f} p50={np.percentile(pv,50):.5f} "
                  f"p90={np.percentile(pv,90):.5f} p99={np.percentile(pv,99):.5f}")
            print(f"    SH var:       p10={np.percentile(sv,10):.5f} p50={np.percentile(sv,50):.5f} "
                  f"p90={np.percentile(sv,90):.5f} p99={np.percentile(sv,99):.5f}")
        print()


if __name__ == "__main__":
    ply_folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\4dgs-data\fish-2\ply"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run(ply_folder, num_frames)
