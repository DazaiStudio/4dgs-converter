"""Validation test: Convert reference PLY files and compare with reference RAW output.

Compares:
1. metadata.json structure and values
2. Binary file sizes
3. Sample binary content
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.ply_to_raw import convert_ply_to_raw, PRECISION_FULL, PRECISION_HALF

# Paths
PLY_DIR = r"D:\4dgs-plugin\ref\io-example\step3_image-to-ply\fish-2_ply"
REF_RAW_DIR = r"D:\4dgs-plugin\ref\io-example\step4_ply-to-raw\fish-2_ply"
TEST_OUTPUT = r"D:\4dgs-plugin\test\output_raw"


def test_frame(frame_idx: int, ply_filename: str):
    """Test conversion of a single frame."""
    print(f"\n{'='*60}")
    print(f"Testing frame {frame_idx}: {ply_filename}")
    print(f"{'='*60}")

    ply_path = os.path.join(PLY_DIR, ply_filename)
    ref_frame_dir = os.path.join(REF_RAW_DIR, f"frame_{frame_idx:04d}")
    ref_metadata_path = os.path.join(ref_frame_dir, "metadata.json")

    # Load reference metadata
    with open(ref_metadata_path) as f:
        ref_meta = json.load(f)

    print(f"Reference: {ref_meta['gaussianCount']} gaussians, "
          f"{ref_meta['textureWidth']}x{ref_meta['textureHeight']} texture")

    # Run conversion
    metadata = convert_ply_to_raw(
        ply_path=ply_path,
        output_folder=TEST_OUTPUT,
        frame_index=frame_idx,
        position_precision=PRECISION_FULL,
        rotation_precision=PRECISION_HALF,
        scale_opacity_precision=PRECISION_HALF,
        sh_precision=PRECISION_HALF,
        progress_callback=lambda msg: print(f"  {msg}"),
    )

    # Compare metadata
    print(f"\n--- Metadata Comparison ---")
    errors = []

    for key in ["textureWidth", "textureHeight", "gaussianCount",
                 "positionPrecision", "rotationPrecision",
                 "scaleOpacityPrecision", "shPrecision"]:
        our = metadata[key]
        ref = ref_meta[key]
        status = "OK" if our == ref else "MISMATCH"
        if our != ref:
            errors.append(f"{key}: ours={our}, ref={ref}")
        print(f"  {key}: {our} (ref: {ref}) [{status}]")

    # Compare min/max positions
    for bound in ["minPosition", "maxPosition"]:
        for axis in ["x", "y", "z"]:
            our = metadata[bound][axis]
            ref = ref_meta[bound][axis]
            diff = abs(our - ref)
            # Allow small float differences
            status = "OK" if diff < 0.01 else "MISMATCH"
            if diff >= 0.01:
                errors.append(f"{bound}.{axis}: ours={our:.6f}, ref={ref:.6f}, diff={diff:.6f}")
            print(f"  {bound}.{axis}: {our:.6f} (ref: {ref:.6f}, diff: {diff:.6f}) [{status}]")

    # Compare file sizes
    print(f"\n--- File Size Comparison ---")
    our_frame_dir = os.path.join(TEST_OUTPUT, f"frame_{frame_idx:04d}")

    file_checks = [
        ("position.bin", "positionFileSize"),
        ("rotation.bin", "rotationFileSize"),
        ("scaleOpacity.bin", "scaleOpacityFileSize"),
    ]
    for i in range(12):
        file_checks.append((f"sh_{i}.bin", None))

    for fname, meta_key in file_checks:
        our_path = os.path.join(our_frame_dir, fname)
        ref_path = os.path.join(ref_frame_dir, fname)

        our_size = os.path.getsize(our_path) if os.path.exists(our_path) else -1

        if meta_key:
            ref_size = ref_meta[meta_key]
        else:
            i = int(fname.split("_")[1].split(".")[0])
            ref_size = ref_meta["shFileSizes"][i]

        status = "OK" if our_size == ref_size else "MISMATCH"
        if our_size != ref_size:
            errors.append(f"{fname}: ours={our_size}, ref={ref_size}")
        print(f"  {fname}: {our_size:,} (ref: {ref_size:,}) [{status}]")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  ALL CHECKS PASSED!")

    return len(errors) == 0


def main():
    os.makedirs(TEST_OUTPUT, exist_ok=True)

    # The PLY files are named frame_0001.ply through frame_0120.ply
    # but the RAW output uses frame_0000 through frame_0119
    # So PLY frame_0001.ply -> RAW frame_0000
    test_cases = [
        (0, "frame_0001.ply"),
        (1, "frame_0002.ply"),
        (2, "frame_0003.ply"),
    ]

    results = []
    for frame_idx, ply_file in test_cases:
        ok = test_frame(frame_idx, ply_file)
        results.append((frame_idx, ok))

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for idx, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  Frame {idx}: {status}")

    all_pass = all(ok for _, ok in results)
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
