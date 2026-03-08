"""Binary content comparison between our output and reference RAW files."""

import os
import sys
import numpy as np

OUR_DIR = r"D:\4dgs-plugin\test\output_raw\frame_0000"
REF_DIR = r"D:\4dgs-plugin\ref\io-example\step4_ply-to-raw\fish-2_ply\frame_0000"


def compare_binary(fname, dtype):
    our_path = os.path.join(OUR_DIR, fname)
    ref_path = os.path.join(REF_DIR, fname)

    our_data = np.fromfile(our_path, dtype=dtype)
    ref_data = np.fromfile(ref_path, dtype=dtype)

    if our_data.shape != ref_data.shape:
        print(f"  {fname}: SHAPE MISMATCH {our_data.shape} vs {ref_data.shape}")
        return False

    exact_match = np.array_equal(our_data, ref_data)

    if exact_match:
        print(f"  {fname}: EXACT MATCH ({len(our_data)} values)")
        return True

    # Check how many differ
    diff_mask = our_data != ref_data
    n_diff = diff_mask.sum()
    pct = n_diff / len(our_data) * 100

    # For floats, check relative error
    finite_mask = np.isfinite(ref_data) & np.isfinite(our_data) & (ref_data != 0)
    if finite_mask.any():
        rel_err = np.abs((our_data[finite_mask] - ref_data[finite_mask]) / ref_data[finite_mask])
        max_rel = rel_err.max()
        mean_rel = rel_err.mean()
    else:
        max_rel = 0
        mean_rel = 0

    print(f"  {fname}: {n_diff}/{len(our_data)} differ ({pct:.3f}%), "
          f"max_rel_err={max_rel:.6e}, mean_rel_err={mean_rel:.6e}")

    # Show first few differences
    diff_indices = np.where(diff_mask)[0][:5]
    for idx in diff_indices:
        print(f"    [{idx}]: ours={our_data[idx]}, ref={ref_data[idx]}")

    return False


def main():
    print("=== Binary Content Comparison (Frame 0) ===\n")

    # Position is float32 (Full precision)
    print("Full precision (float32):")
    compare_binary("position.bin", np.float32)

    # Others are float16 (Half precision)
    print("\nHalf precision (float16):")
    for fname in ["rotation.bin", "scaleOpacity.bin"]:
        compare_binary(fname, np.float16)

    print("\nSH textures (float16):")
    for i in range(12):
        compare_binary(f"sh_{i}.bin", np.float16)


if __name__ == "__main__":
    main()
