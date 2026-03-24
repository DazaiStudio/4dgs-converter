"""Deep analysis of PLY gaussian data.

Analyze the statistical properties of ml-sharp output to find
custom compression opportunities:
- Attribute distributions and entropy
- Spatial autocorrelation (after Morton sort)
- Attribute correlations
- Clustering potential (how many "types" of gaussians?)
- Predictability from neighbors
"""

import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import lz4.block

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order


def pixel_shuffle(data: bytes, bpp: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape(-1, bpp).T.reshape(-1).tobytes()


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def entropy_bits(data: np.ndarray) -> float:
    """Compute Shannon entropy in bits per element."""
    flat = data.ravel()
    _, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-15))


def analyze_distribution(name: str, data: np.ndarray):
    """Analyze and print distribution statistics."""
    flat = data.ravel()
    print(f"\n  [{name}] shape={data.shape}, dtype={data.dtype}")
    print(f"    range:  [{flat.min():.6f}, {flat.max():.6f}]")
    print(f"    mean:   {flat.mean():.6f}  std: {flat.std():.6f}")
    print(f"    median: {np.median(flat):.6f}")

    # Percentiles
    pcts = [1, 5, 25, 50, 75, 95, 99]
    vals = np.percentile(flat, pcts)
    pct_str = "  ".join([f"p{p}={v:.4f}" for p, v in zip(pcts, vals)])
    print(f"    percentiles: {pct_str}")

    # Check for concentration
    near_zero = np.mean(np.abs(flat) < 0.01) * 100
    print(f"    near zero (<0.01): {near_zero:.1f}%")

    # Unique value analysis (quantized to fp16)
    fp16 = flat.astype(np.float16)
    n_unique = len(np.unique(fp16))
    print(f"    unique fp16 values: {n_unique:,} (of {len(fp16):,})")

    # Entropy of fp16 bytes
    raw_bytes = fp16.tobytes()
    byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    ent = entropy_bits(byte_arr)
    print(f"    byte entropy: {ent:.2f} bits (max=8.0)")

    return {
        "min": float(flat.min()), "max": float(flat.max()),
        "mean": float(flat.mean()), "std": float(flat.std()),
        "near_zero_pct": near_zero, "n_unique_fp16": n_unique,
        "byte_entropy": ent,
    }


def analyze_spatial_autocorrelation(name: str, data: np.ndarray,
                                    sorted_indices: np.ndarray, n_sample: int = 100000):
    """After Morton sort, check if adjacent gaussians are similar."""
    sorted_data = data[sorted_indices]
    n = min(len(sorted_data) - 1, n_sample)

    # Difference between consecutive elements in Morton order
    diffs = sorted_data[1:n+1] - sorted_data[:n]
    abs_diffs = np.abs(diffs)

    mean_diff = np.mean(abs_diffs)
    mean_val = np.mean(np.abs(sorted_data[:n+1]))
    relative_diff = mean_diff / (mean_val + 1e-10)

    # How well can we predict from previous value?
    # If we store deltas instead of absolute values
    delta_bytes = diffs.astype(np.float16).tobytes()
    raw_bytes = sorted_data[:n+1].astype(np.float16).tobytes()

    delta_compressed = len(compress_lz4(pixel_shuffle(delta_bytes, diffs.shape[1] * 2 if diffs.ndim > 1 else 2)))
    raw_compressed = len(compress_lz4(pixel_shuffle(raw_bytes, (sorted_data.shape[1] * 2 if sorted_data.ndim > 1 else 2))))

    print(f"\n  [{name}] Spatial autocorrelation (Morton-sorted):")
    print(f"    mean |delta|:     {mean_diff:.6f}")
    print(f"    mean |value|:     {mean_val:.6f}")
    print(f"    relative delta:   {relative_diff:.4f} ({relative_diff*100:.1f}%)")
    print(f"    delta LZ4:        {delta_compressed/1024:.1f} KB")
    print(f"    raw LZ4:          {raw_compressed/1024:.1f} KB")
    print(f"    delta/raw ratio:  {delta_compressed/raw_compressed:.3f}")

    return {
        "mean_delta": float(mean_diff),
        "relative_delta": float(relative_diff),
        "delta_ratio": delta_compressed / raw_compressed,
    }


def analyze_correlations(gaussians: dict, sorted_indices: np.ndarray):
    """Check correlations between attributes."""
    print(f"\n  Cross-attribute correlations:")

    pos = gaussians["position"][sorted_indices]
    rot = gaussians["rotation"][sorted_indices]
    scale = gaussians["scale"][sorted_indices]
    opacity = gaussians["opacity"][sorted_indices]
    sh_dc = gaussians["sh_dc"][sorted_indices]

    # Scale vs opacity
    scale_mag = np.linalg.norm(scale, axis=1)
    corr_scale_op = np.corrcoef(scale_mag, opacity)[0, 1]
    print(f"    scale_magnitude vs opacity: r={corr_scale_op:.4f}")

    # Scale components correlation
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        r = np.corrcoef(scale[:, i], scale[:, j])[0, 1]
        print(f"    scale_{i} vs scale_{j}: r={r:.4f}")

    # Position vs SH (color depends on location?)
    for d in range(3):
        r = np.corrcoef(pos[:, d], sh_dc[:, 0])[0, 1]
        print(f"    pos_{d} vs sh_dc_0: r={r:.4f}")


def analyze_clustering(name: str, data: np.ndarray, max_k: int = 256,
                       n_sample: int = 50000):
    """How many clusters/codebook entries needed?"""
    from sklearn.cluster import MiniBatchKMeans

    # Sample for speed
    idx = np.random.choice(len(data), min(n_sample, len(data)), replace=False)
    sample = data[idx].astype(np.float32)

    print(f"\n  [{name}] Clustering analysis (K-means):")

    results = []
    for k in [8, 16, 32, 64, 128, 256]:
        if k > max_k:
            break
        km = MiniBatchKMeans(n_clusters=k, batch_size=1000, n_init=3, random_state=42)
        km.fit(sample)
        labels = km.predict(sample)
        centers = km.cluster_centers_

        # Reconstruction error
        reconstructed = centers[labels]
        mse = np.mean((sample - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        max_err = np.max(np.abs(sample - reconstructed))

        # Storage estimate
        codebook_size = k * data.shape[1] * 4  # float32
        index_bits = np.ceil(np.log2(k))
        index_size = len(data) * index_bits / 8  # for full dataset

        # Compare to raw fp16
        raw_size = len(data) * data.shape[1] * 2

        total_vq = codebook_size + index_size
        ratio = total_vq / raw_size

        print(f"    K={k:>3}: RMSE={rmse:.6f} max_err={max_err:.4f} "
              f"codebook={codebook_size/1024:.1f}KB + indices={index_size/1024:.1f}KB "
              f"= {total_vq/1024:.1f}KB ({ratio:.1%} of fp16)")
        results.append({"k": k, "rmse": rmse, "ratio": ratio})

    return results


def analyze_predictive_coding(name: str, data: np.ndarray,
                              sorted_indices: np.ndarray):
    """Test various predictive coding schemes on Morton-sorted data."""
    sorted_data = data[sorted_indices].astype(np.float32)
    n = len(sorted_data)

    bpp = sorted_data.shape[1] * 2 if sorted_data.ndim > 1 else 2

    # Raw (baseline)
    raw_fp16 = sorted_data.astype(np.float16).tobytes()
    raw_comp = len(compress_lz4(pixel_shuffle(raw_fp16, bpp)))

    # Delta (diff from previous)
    delta1 = np.zeros_like(sorted_data)
    delta1[0] = sorted_data[0]
    delta1[1:] = sorted_data[1:] - sorted_data[:-1]
    d1_fp16 = delta1.astype(np.float16).tobytes()
    d1_comp = len(compress_lz4(pixel_shuffle(d1_fp16, bpp)))

    # Double delta
    delta2 = np.zeros_like(delta1)
    delta2[0] = delta1[0]
    delta2[1:] = delta1[1:] - delta1[:-1]
    d2_fp16 = delta2.astype(np.float16).tobytes()
    d2_comp = len(compress_lz4(pixel_shuffle(d2_fp16, bpp)))

    # Predict from 2-neighbor average
    pred_avg = np.zeros_like(sorted_data)
    pred_avg[0] = sorted_data[0]
    pred_avg[1] = sorted_data[1]
    pred_avg[2:] = (sorted_data[:-2] + sorted_data[1:-1]) / 2
    residual_avg = sorted_data - pred_avg
    ra_fp16 = residual_avg.astype(np.float16).tobytes()
    ra_comp = len(compress_lz4(pixel_shuffle(ra_fp16, bpp)))

    # XOR on fp16 bytes (adjacent elements)
    fp16_arr = sorted_data.astype(np.float16)
    raw_u8 = np.frombuffer(fp16_arr.tobytes(), dtype=np.uint8).reshape(n, -1)
    xor_arr = np.zeros_like(raw_u8)
    xor_arr[0] = raw_u8[0]
    xor_arr[1:] = np.bitwise_xor(raw_u8[1:], raw_u8[:-1])
    xor_comp = len(compress_lz4(pixel_shuffle(xor_arr.tobytes(), bpp)))

    print(f"\n  [{name}] Predictive coding (Morton-sorted, shuffle+LZ4):")
    print(f"    Raw:               {raw_comp/1024:>8.1f} KB (baseline)")
    print(f"    Delta (prev):      {d1_comp/1024:>8.1f} KB ({d1_comp/raw_comp:.3f}x)")
    print(f"    Double delta:      {d2_comp/1024:>8.1f} KB ({d2_comp/raw_comp:.3f}x)")
    print(f"    Avg-2 residual:    {ra_comp/1024:>8.1f} KB ({ra_comp/raw_comp:.3f}x)")
    print(f"    XOR (prev, fp16):  {xor_comp/1024:>8.1f} KB ({xor_comp/raw_comp:.3f}x)")

    return {"raw": raw_comp, "delta": d1_comp, "xor": xor_comp,
            "delta_ratio": d1_comp / raw_comp, "xor_ratio": xor_comp / raw_comp}


def run(ply_path: str):
    print(f"Loading: {ply_path}")
    g = load_gaussian_ply(ply_path)
    n = len(g["position"])
    print(f"  {n:,} gaussians\n")

    # Activate
    opacity_act = 1.0 / (1.0 + np.exp(-g["opacity"]))
    scale_act = np.exp(g["scale"])

    # Morton sort
    print("Morton sorting...")
    sorted_indices, _, _ = sort_3d_morton_order(g["position"])
    print()

    # =========================================================================
    print("=" * 80)
    print("1. ATTRIBUTE DISTRIBUTIONS")
    print("=" * 80)

    attrs = {
        "position": g["position"],
        "rotation (raw)": g["rotation"],
        "scale (raw, pre-exp)": g["scale"],
        "scale (activated)": scale_act,
        "opacity (raw, pre-sigmoid)": g["opacity"],
        "opacity (activated)": opacity_act,
        "sh_dc": g["sh_dc"],
    }

    for name, data in attrs.items():
        analyze_distribution(name, data)

    # =========================================================================
    print(f"\n{'=' * 80}")
    print("2. SPATIAL AUTOCORRELATION (Morton-sorted)")
    print("=" * 80)

    spatial_results = {}
    for name, data in [
        ("position", g["position"]),
        ("rotation", g["rotation"]),
        ("scale (raw)", g["scale"]),
        ("opacity (raw)", g["opacity"].reshape(-1, 1)),
        ("sh_dc", g["sh_dc"]),
    ]:
        spatial_results[name] = analyze_spatial_autocorrelation(
            name, data, sorted_indices)

    # =========================================================================
    print(f"\n{'=' * 80}")
    print("3. PREDICTIVE CODING (intra-frame, Morton-sorted)")
    print("=" * 80)

    pred_results = {}
    for name, data in [
        ("position", g["position"]),
        ("rotation", g["rotation"]),
        ("scale (raw)", g["scale"]),
        ("opacity (raw)", g["opacity"].reshape(-1, 1)),
        ("sh_dc", g["sh_dc"]),
    ]:
        pred_results[name] = analyze_predictive_coding(name, data, sorted_indices)

    # =========================================================================
    print(f"\n{'=' * 80}")
    print("4. CROSS-ATTRIBUTE CORRELATIONS")
    print("=" * 80)

    analyze_correlations(g, sorted_indices)

    # =========================================================================
    print(f"\n{'=' * 80}")
    print("5. CLUSTERING / VECTOR QUANTIZATION POTENTIAL")
    print("=" * 80)

    try:
        for name, data in [
            ("rotation", g["rotation"]),
            ("scale (raw)", g["scale"]),
            ("sh_dc", g["sh_dc"]),
        ]:
            analyze_clustering(name, data[sorted_indices])
    except ImportError:
        print("  sklearn not available, skipping clustering analysis")

    # =========================================================================
    print(f"\n{'=' * 80}")
    print("6. COMBINED OPTIMAL ENCODING ESTIMATE")
    print("=" * 80)

    # For each attribute, pick the best encoding found
    print(f"\n  Best intra-frame encoding per attribute:")
    total_best = 0
    total_raw = 0
    for name in pred_results:
        r = pred_results[name]
        best_method = "raw"
        best_size = r["raw"]
        for method in ["delta", "xor"]:
            if r[method] < best_size:
                best_size = r[method]
                best_method = method
        total_best += best_size
        total_raw += r["raw"]
        print(f"    {name:<20}: {best_method:<10} {best_size/1024:.1f} KB ({best_size/r['raw']:.3f}x of raw)")

    print(f"\n    TOTAL best:   {total_best/1024/1024:.2f} MB")
    print(f"    TOTAL raw:    {total_raw/1024/1024:.2f} MB")
    print(f"    Improvement:  {(1-total_best/total_raw)*100:.1f}%")
    print(f"    480 frames:   {total_best/1024/1024*480:.2f} MB = {total_best/1024/1024/1024*480:.2f} GB")


if __name__ == "__main__":
    ply_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\4dgs-data\fish-2\ply\frame_0001.ply"
    run(ply_path)
