"""Post-deformation alignment test.

Takes per-frame independent PLY files from ml-sharp, attempts to establish
gaussian correspondence across frames via KD-tree spatial matching,
then measures delta compressibility.

Goal: determine if post-alignment delta encoding can dramatically reduce storage.
"""

import sys
import os
import time

import numpy as np
from scipy.spatial import cKDTree
import lz4.block

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.utils.ply_reader import load_gaussian_ply


def pixel_shuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bpp).T.reshape(-1).tobytes()


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def load_frame(ply_path: str) -> dict:
    """Load PLY and return key attributes as float32 arrays."""
    g = load_gaussian_ply(ply_path)
    # Normalize quaternion
    rot = g["rotation"]
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot_norm = np.where(rot_norm == 0, 1.0, rot_norm)
    rot = rot / rot_norm

    return {
        "position": g["position"].astype(np.float32),     # (N, 3)
        "rotation": rot.astype(np.float32),                # (N, 4)
        "scale": g["scale"].astype(np.float32),            # (N, 3)
        "opacity": g["opacity"].astype(np.float32),        # (N,)
        "sh_dc": g["sh_dc"].astype(np.float32),            # (N, 3)
    }


def match_frames(canonical: dict, target: dict, max_dist: float = 0.05):
    """Match target gaussians to canonical using KD-tree nearest neighbor.

    Returns:
        matched_canon_idx: indices into canonical for matched pairs
        matched_target_idx: indices into target for matched pairs
        unmatched_target_idx: target indices with no good match
        match_distances: distances for matched pairs
    """
    t0 = time.perf_counter()
    tree = cKDTree(canonical["position"])
    distances, indices = tree.query(target["position"], k=1, workers=-1)
    t1 = time.perf_counter()

    mask = distances < max_dist
    matched_target_idx = np.where(mask)[0]
    matched_canon_idx = indices[mask]
    unmatched_target_idx = np.where(~mask)[0]
    match_distances = distances[mask]

    print(f"  KD-tree match: {t1-t0:.2f}s, "
          f"matched={len(matched_target_idx)} ({len(matched_target_idx)/len(target['position'])*100:.1f}%), "
          f"unmatched={len(unmatched_target_idx)}, "
          f"mean_dist={np.mean(match_distances):.6f}, "
          f"max_dist={np.max(match_distances):.6f}")

    return matched_canon_idx, matched_target_idx, unmatched_target_idx, match_distances


def compute_deltas(canonical: dict, target: dict,
                   canon_idx: np.ndarray, target_idx: np.ndarray) -> dict:
    """Compute per-attribute deltas for matched gaussians."""
    deltas = {}
    deltas["position"] = target["position"][target_idx] - canonical["position"][canon_idx]
    deltas["rotation"] = target["rotation"][target_idx] - canonical["rotation"][canon_idx]
    deltas["scale"] = target["scale"][target_idx] - canonical["scale"][canon_idx]
    deltas["opacity"] = target["opacity"][target_idx] - canonical["opacity"][canon_idx]
    deltas["sh_dc"] = target["sh_dc"][target_idx] - canonical["sh_dc"][canon_idx]
    return deltas


def measure_compression(data: np.ndarray, label: str, bpp: int = None):
    """Measure shuffle+LZ4 compression of an array."""
    raw = data.astype(np.float16).tobytes()
    raw_size = len(raw)

    if bpp is None:
        bpp = data.shape[1] * 2 if data.ndim > 1 else 2  # fp16

    shuffled = pixel_shuffle(raw, bpp)
    compressed = compress_lz4(shuffled)
    ratio = len(compressed) / raw_size

    return {
        "label": label,
        "raw_size": raw_size,
        "compressed_size": len(compressed),
        "ratio": ratio,
    }


def run_test(ply_folder: str, num_frames: int = 10, max_dist_options=None):
    if max_dist_options is None:
        max_dist_options = [0.01, 0.02, 0.05, 0.1, 0.2]

    # Find PLY files
    ply_files = sorted([
        os.path.join(ply_folder, f) for f in os.listdir(ply_folder)
        if f.endswith(".ply")
    ])
    print(f"Found {len(ply_files)} PLY files in {ply_folder}")
    print(f"Testing first {num_frames} frames\n")

    # Load canonical (frame 0)
    print("Loading canonical frame (frame 0)...")
    canonical = load_frame(ply_files[0])
    n_canon = len(canonical["position"])
    print(f"  {n_canon} gaussians\n")

    # Load subsequent frames
    targets = []
    for i in range(1, min(num_frames, len(ply_files))):
        print(f"Loading frame {i}...")
        target = load_frame(ply_files[i])
        print(f"  {len(target['position'])} gaussians")
        targets.append((i, target))
    print()

    # =========================================================================
    # Test 1: Match rate at different distance thresholds
    # =========================================================================
    print("=" * 80)
    print("TEST 1: Match rate vs distance threshold")
    print("=" * 80)

    # Use frame 1 as test case
    _, test_target = targets[0]
    print(f"\nFrame 0 → Frame 1:")
    for max_d in max_dist_options:
        tree = cKDTree(canonical["position"])
        distances, indices = tree.query(test_target["position"], k=1, workers=-1)
        match_rate = np.mean(distances < max_d) * 100
        mean_d = np.mean(distances[distances < max_d]) if np.any(distances < max_d) else 0
        print(f"  threshold={max_d:.3f}: match={match_rate:.1f}%, mean_dist={mean_d:.6f}")

    # Also test frame 0 → frame 10, frame 0 → frame 24 (1 second)
    for skip_name, skip_idx in [("frame 5", 4), ("frame 10", 9)]:
        if skip_idx < len(targets):
            _, far_target = targets[skip_idx]
            print(f"\nFrame 0 → {skip_name}:")
            tree = cKDTree(canonical["position"])
            distances, _ = tree.query(far_target["position"], k=1, workers=-1)
            for max_d in max_dist_options:
                match_rate = np.mean(distances < max_d) * 100
                print(f"  threshold={max_d:.3f}: match={match_rate:.1f}%")

    # =========================================================================
    # Test 2: Delta compressibility (best threshold)
    # =========================================================================
    print()
    print("=" * 80)
    print("TEST 2: Delta compression (threshold=0.05)")
    print("=" * 80)

    best_threshold = 0.05

    # Baseline: full frame compression (no delta)
    print("\n--- Baseline: full frame, no delta (fp16 + shuffle + LZ4) ---")
    frame1 = targets[0][1]
    baseline_sizes = {}
    for attr in ["position", "rotation", "scale", "sh_dc"]:
        data = frame1[attr]
        r = measure_compression(data, attr)
        baseline_sizes[attr] = r["compressed_size"]
        print(f"  {attr:<12}: {r['raw_size']/1e6:.2f} MB raw → {r['compressed_size']/1e6:.2f} MB LZ4 ({r['ratio']:.1%})")

    opacity_r = measure_compression(frame1["opacity"].reshape(-1, 1), "opacity")
    baseline_sizes["opacity"] = opacity_r["compressed_size"]
    print(f"  {'opacity':<12}: {opacity_r['raw_size']/1e6:.2f} MB raw → {opacity_r['compressed_size']/1e6:.2f} MB LZ4 ({opacity_r['ratio']:.1%})")

    total_baseline = sum(baseline_sizes.values())
    print(f"  TOTAL:       {total_baseline/1e6:.2f} MB")

    # Delta compression for consecutive frames
    print(f"\n--- Delta compression (frame N vs frame 0, threshold={best_threshold}) ---")

    print(f"\n{'Frame':<8} {'Match%':>7} {'Δpos MB':>8} {'Δrot MB':>8} {'Δscale':>8} "
          f"{'Δsh MB':>8} {'Unmatch':>8} {'TOTAL':>8} {'vs base':>8}")
    print("-" * 80)

    for frame_idx, target in targets:
        canon_idx, target_idx, unmatched_idx, _ = match_frames(
            canonical, target, max_dist=best_threshold)

        match_pct = len(target_idx) / len(target["position"]) * 100

        # Compress deltas for matched
        deltas = compute_deltas(canonical, target, canon_idx, target_idx)

        delta_sizes = {}
        for attr in ["position", "rotation", "scale", "sh_dc"]:
            d = deltas[attr]
            r = measure_compression(d, attr)
            delta_sizes[attr] = r["compressed_size"]

        d_op = deltas["opacity"].reshape(-1, 1)
        r_op = measure_compression(d_op, "opacity")
        delta_sizes["opacity"] = r_op["compressed_size"]

        # Compress unmatched (full data, no delta)
        unmatched_size = 0
        if len(unmatched_idx) > 0:
            for attr in ["position", "rotation", "scale", "sh_dc"]:
                um_data = target[attr][unmatched_idx]
                um_r = measure_compression(um_data, attr)
                unmatched_size += um_r["compressed_size"]
            um_op = target["opacity"][unmatched_idx].reshape(-1, 1)
            unmatched_size += measure_compression(um_op, "opacity")["compressed_size"]

        # Index storage: matched indices as uint32
        index_data = canon_idx.astype(np.uint32).tobytes()
        index_compressed = len(compress_lz4(pixel_shuffle(index_data, 4)))

        total_delta = sum(delta_sizes.values()) + unmatched_size + index_compressed
        savings = (1 - total_delta / total_baseline) * 100

        print(f"  {frame_idx:<6} {match_pct:>6.1f}% "
              f"{delta_sizes['position']/1e6:>7.2f} {delta_sizes['rotation']/1e6:>7.2f} "
              f"{delta_sizes['scale']/1e6:>7.2f} {delta_sizes['sh_dc']/1e6:>7.2f} "
              f"{unmatched_size/1e6:>7.2f} {total_delta/1e6:>7.2f} {savings:>+7.1f}%")

    # =========================================================================
    # Test 3: Sliding window (frame N-1 → frame N) vs fixed canonical
    # =========================================================================
    print()
    print("=" * 80)
    print("TEST 3: Sliding window (prev frame as reference) vs fixed canonical")
    print("=" * 80)

    print(f"\n{'Frame':<8} {'Mode':<15} {'Match%':>7} {'Delta MB':>9} {'vs base':>8}")
    print("-" * 55)

    prev_frame = canonical
    for frame_idx, target in targets:
        # Fixed canonical (frame 0)
        c_idx, t_idx, um_idx, _ = match_frames(canonical, target, max_dist=best_threshold)
        match_fixed = len(t_idx) / len(target["position"]) * 100
        deltas_fixed = compute_deltas(canonical, target, c_idx, t_idx)
        size_fixed = 0
        for attr in ["position", "rotation", "scale", "sh_dc"]:
            size_fixed += measure_compression(deltas_fixed[attr], attr)["compressed_size"]
        size_fixed += measure_compression(deltas_fixed["opacity"].reshape(-1, 1), "opacity")["compressed_size"]
        # Add unmatched
        for attr in ["position", "rotation", "scale", "sh_dc"]:
            if len(um_idx) > 0:
                size_fixed += measure_compression(target[attr][um_idx], attr)["compressed_size"]

        # Sliding window (prev frame)
        c_idx2, t_idx2, um_idx2, _ = match_frames(prev_frame, target, max_dist=best_threshold)
        match_slide = len(t_idx2) / len(target["position"]) * 100
        deltas_slide = compute_deltas(prev_frame, target, c_idx2, t_idx2)
        size_slide = 0
        for attr in ["position", "rotation", "scale", "sh_dc"]:
            size_slide += measure_compression(deltas_slide[attr], attr)["compressed_size"]
        size_slide += measure_compression(deltas_slide["opacity"].reshape(-1, 1), "opacity")["compressed_size"]
        for attr in ["position", "rotation", "scale", "sh_dc"]:
            if len(um_idx2) > 0:
                size_slide += measure_compression(target[attr][um_idx2], attr)["compressed_size"]

        print(f"  {frame_idx:<6} {'fixed(f0)':<15} {match_fixed:>6.1f}% {size_fixed/1e6:>8.2f} "
              f"{(1-size_fixed/total_baseline)*100:>+7.1f}%")
        print(f"  {'':<6} {'slide(f{n-1})':<15} {match_slide:>6.1f}% {size_slide/1e6:>8.2f} "
              f"{(1-size_slide/total_baseline)*100:>+7.1f}%")

        prev_frame = target

    # =========================================================================
    # Test 4: Position-only analysis (understand the distribution)
    # =========================================================================
    print()
    print("=" * 80)
    print("TEST 4: Position delta statistics (frame 0 → frame 1)")
    print("=" * 80)

    c_idx, t_idx, _, _ = match_frames(canonical, targets[0][1], max_dist=best_threshold)
    deltas = compute_deltas(canonical, targets[0][1], c_idx, t_idx)

    for attr in ["position", "rotation", "scale", "sh_dc", "opacity"]:
        d = deltas[attr]
        print(f"\n  {attr}:")
        print(f"    shape: {d.shape}")
        print(f"    mean abs: {np.mean(np.abs(d)):.6f}")
        print(f"    max abs:  {np.max(np.abs(d)):.6f}")
        print(f"    std:      {np.std(d):.6f}")
        print(f"    % near zero (<1e-3): {np.mean(np.abs(d) < 1e-3)*100:.1f}%")
        print(f"    % near zero (<1e-2): {np.mean(np.abs(d) < 1e-2)*100:.1f}%")


if __name__ == "__main__":
    ply_folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\4dgs-data\fish-2\ply"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_test(ply_folder, num_frames)
