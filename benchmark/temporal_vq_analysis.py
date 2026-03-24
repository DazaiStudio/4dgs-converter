"""Temporal VQ Analysis: measures per-frame VQ jitter across consecutive frames.

For each tracked gaussian across 10 consecutive frames, compares:
- The actual attribute change between consecutive frames (ground truth)
- The VQ reconstruction error per frame
- The "temporal jitter" caused by independent per-frame codebooks

Key question: Does per-frame VQ introduce temporal flickering even when
the underlying values change smoothly?
"""

import os
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order


VQ_K = 1024
NUM_FRAMES = 10
NUM_TRACKED = 500  # Track more gaussians for statistical significance
PLY_FOLDER = r"D:\4dgs-data\nature\ANIMAL_camel\ply"


def vq_encode(data: np.ndarray, k: int = VQ_K) -> tuple:
    """Vector quantize. Returns (codebook float32, indices, reconstructed)."""
    n = len(data)
    sample_idx = np.random.choice(n, min(100000, n), replace=False)
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=42)
    km.fit(data[sample_idx])
    indices = km.predict(data)
    codebook = km.cluster_centers_.astype(np.float32)
    reconstructed = codebook[indices]
    return codebook, indices, reconstructed


def analyze_v1_vs_v2_encoding(gaussians: dict):
    """Compare V1 and V2 encoding for a single frame to quantify per-attribute errors."""
    n = len(gaussians["position"])
    pos = gaussians["position"].astype(np.float32)
    rot = gaussians["rotation"].astype(np.float32)
    scale = gaussians["scale"].astype(np.float32)
    opacity = gaussians["opacity"].astype(np.float32)
    sh_dc = gaussians["sh_dc"].astype(np.float32)

    # Normalize rotation (both V1 and V2 do this)
    # V1: quat = (rot_1, rot_2, rot_3, rot_0), normalized
    qx, qy, qz, qw = rot[:, 1], rot[:, 2], rot[:, 3], rot[:, 0]
    qlen = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qlen = np.where(qlen == 0, 1.0, qlen)
    qx /= qlen; qy /= qlen; qz /= qlen; qw /= qlen

    # V1 packs rotation into texture as: (qz, qx, -qy, qw)
    v1_rot_tex = np.column_stack([qz, qx, -qy, qw])

    # V2 normalizes as (rot_0..3), norm, then stores raw (rot_0, rot_1, rot_2, rot_3)
    rot_v2 = rot.copy()
    rot_v2_norm = np.linalg.norm(rot_v2, axis=1, keepdims=True)
    rot_v2_norm = np.where(rot_v2_norm == 0, 1.0, rot_v2_norm)
    rot_v2 = rot_v2 / rot_v2_norm

    # ========== CRITICAL DIFFERENCE: V2 encoder DOES NOT do (Z,X,-Y) swizzle ==========
    # V2 stores the normalized quaternion as-is: (rot_0, rot_1, rot_2, rot_3) = (w, x, y, z)
    # But the renderer (V2 decoder) just does codebook[idx] → fp16 RGBA directly
    # So the renderer gets: R=rot_0(w), G=rot_1(x), B=rot_2(y), A=rot_3(z)
    #
    # V1 texture layout:   R=qz, G=qx, B=-qy, A=qw  (with Z,X,-Y swizzle)
    # V2 texture layout:   R=w,  G=x,  B=y,   A=z   (NO swizzle!)
    #
    # The shader reads rotation as (R, G, B, A) expecting (Z, X, -Y, W) order.
    # V2 gives it (W, X, Y, Z) instead!

    print("\n" + "="*70)
    print("V1 vs V2 ROTATION COORDINATE MISMATCH ANALYSIS")
    print("="*70)
    print(f"\nV1 rotation texture layout per pixel: (qz, qx, -qy, qw)")
    print(f"V2 rotation VQ stores raw normalized:  (rot_0, rot_1, rot_2, rot_3) = (w, x, y, z)")
    print(f"V2 decoder writes to fp16 RGBA:        (rot_0, rot_1, rot_2, rot_3)")
    print(f"  i.e. R=w, G=x, B=y, A=z")
    print(f"\n*** SHADER EXPECTS (R,G,B,A) = (Z, X, -Y, W) ***")
    print(f"*** V2 GIVES   (R,G,B,A) = (W, X,  Y, Z) ***")
    print(f"\n>>> THIS IS A CRITICAL BUG: quaternion channels are in the WRONG ORDER!")

    # Quantify: what does "wrong rotation" look like?
    sample = np.random.choice(n, min(10000, n), replace=False)
    v1_rgba = v1_rot_tex[sample]  # (Z, X, -Y, W)
    v2_rgba = rot_v2[sample]       # (W, X, Y, Z) ← wrong order

    # When the shader reads V2 thinking it's V1 format:
    # shader_Z = R_v2 = W (wrong! should be Z)
    # shader_X = G_v2 = X (correct by coincidence)
    # shader_Y = -B_v2 = -Y (but V2 stores Y, and shader negates B, so shader gets -Y = correct)
    # Wait, let me re-read the shader...

    # Actually the shader reads R,G,B,A directly as the quat components.
    # V1 stores: texel = (qz, qx, -qy, qw)
    # The shader uses R as quat.Z component, G as quat.X, etc.
    # If V2 stores texel = (w, x, y, z), shader interprets:
    #   "Z" = w, "X" = x, "Y_neg" = y (but shader expects -Y, gets +Y), "W" = z
    # So shader gets quat(Z=w, X=x, Y=-y, W=z) instead of quat(Z=z, X=x, Y=-y, W=w)
    # The W and Z channels are SWAPPED.

    print(f"\nDetailed channel mapping:")
    print(f"  Channel  V1 stores  V2 stores  Shader interprets V2 as")
    print(f"  R        quat.Z     quat.W     quat.Z (WRONG - it's W)")
    print(f"  G        quat.X     quat.X     quat.X (correct)")
    print(f"  B        -quat.Y    quat.Y     -quat.Y (WRONG SIGN - B is Y, shader expects -Y)")
    print(f"  A        quat.W     quat.Z     quat.W (WRONG - it's Z)")

    # Actually wait - need to re-examine more carefully.
    # V2 encoder stores normalized (rot_0, rot_1, rot_2, rot_3) where rot_0=W, rot_1=X, rot_2=Y, rot_3=Z
    # The VQ operates on this 4D vector.
    # The decoder writes codebook entry directly: E[0]=rot_0=W, E[1]=rot_1=X, E[2]=rot_2=Y, E[3]=rot_3=Z
    # But V1 writes texture as: (qz, qx, -qy, qw)
    # So V1's RGBA = (Z, X, -Y, W) but V2's RGBA = (W, X, Y, Z)

    # Compute actual angular error if we interpret V2 data through V1's assumptions
    # V1 assumes: quat = (X=G, Y=-B, Z=R, W=A)
    # With V2 data: quat_wrong = (X=rot_1, Y=-rot_2, Z=rot_0, W=rot_3)
    #   = (X=x, Y=-y, Z=w, W=z)  ← Z and W swapped, Y negated
    # Correct: quat = (X=x, Y=y, Z=z, W=w)

    quat_correct = np.column_stack([qx[sample], qy[sample], qz[sample], qw[sample]])  # XYZW
    # What shader gets from V2: X=G=x, Y=-B=-y, Z=R=w, W=A=z
    quat_v2_wrong = np.column_stack([
        rot_v2[sample, 1],      # G → X = x (correct)
        -rot_v2[sample, 2],     # -B → Y = -y (WRONG, should be +y... wait)
        rot_v2[sample, 0],      # R → Z = w (WRONG, should be z)
        rot_v2[sample, 3],      # A → W = z (WRONG, should be w)
    ])

    # Dot product between correct and wrong quaternions (cos of half-angle)
    dots = np.abs(np.sum(quat_correct * quat_v2_wrong, axis=1))
    dots = np.clip(dots, 0, 1)
    angles_deg = 2 * np.degrees(np.arccos(dots))

    print(f"\nAngular error from rotation mismatch:")
    print(f"  Mean:   {np.mean(angles_deg):.1f} degrees")
    print(f"  Median: {np.median(angles_deg):.1f} degrees")
    print(f"  p95:    {np.percentile(angles_deg, 95):.1f} degrees")
    print(f"  Max:    {np.max(angles_deg):.1f} degrees")

    # ========== SCALE ENCODING DIFFERENCE ==========
    print("\n" + "="*70)
    print("V1 vs V2 SCALE ENCODING DIFFERENCE")
    print("="*70)

    scale_activated = np.exp(scale)
    # V1: texture stores (exp(scale_2), exp(scale_0), exp(scale_1), opacity) — note the Z,X,Y swizzle!
    # V2 encoder: stores raw log-space (scale_0, scale_1, scale_2) in VQ
    # V2 decoder: applies exp() then packs as (exp(S[0]), exp(S[1]), exp(S[2]), opacity)
    # V1 packs:   (exp(scale_2), exp(scale_0), exp(scale_1), opacity)  ← Z,X,Y order
    # V2 packs:   (exp(S[0]),    exp(S[1]),    exp(S[2]),    opacity)  ← 0,1,2 order

    print(f"\nV1 ScaleOpacity texture: R=exp(scale_2), G=exp(scale_0), B=exp(scale_1), A=opacity")
    print(f"V2 decoder output:       R=exp(S[0]),     G=exp(S[1]),     B=exp(S[2]),     A=opacity")
    print(f"\n*** SCALE AXES ARE IN DIFFERENT ORDER ***")
    print(f"V1: (R,G,B) = (Z-axis, X-axis, Y-axis)")
    print(f"V2: (R,G,B) = (X-axis, Y-axis, Z-axis)")
    print(f"This means scale axes 0 and 2 are SWAPPED in V2!")

    v1_scale_rgb = np.column_stack([
        scale_activated[:, 2],  # R = exp(scale_2)
        scale_activated[:, 0],  # G = exp(scale_0)
        scale_activated[:, 1],  # B = exp(scale_1)
    ])
    v2_scale_rgb = np.column_stack([
        scale_activated[:, 0],  # R = exp(scale_0) — WRONG, should be scale_2
        scale_activated[:, 1],  # G = exp(scale_1) — WRONG, should be scale_0
        scale_activated[:, 2],  # B = exp(scale_2) — WRONG, should be scale_1
    ])

    scale_diff = np.abs(v1_scale_rgb[sample] - v2_scale_rgb[sample])
    print(f"\nScale channel errors (V2 decoded vs V1 expected):")
    print(f"  R channel: mean diff = {np.mean(scale_diff[:, 0]):.6f}")
    print(f"  G channel: mean diff = {np.mean(scale_diff[:, 1]):.6f}")
    print(f"  B channel: mean diff = {np.mean(scale_diff[:, 2]):.6f}")

    # ========== POSITION: fp32 vs fp16 ==========
    print("\n" + "="*70)
    print("POSITION PRECISION: V1 fp32 vs V2 fp16")
    print("="*70)

    # V1 default: position_precision=PRECISION_FULL (0) = fp32
    # V2: always fp16
    pos_fp32 = pos.astype(np.float32)
    pos_fp16 = pos.astype(np.float16).astype(np.float32)
    pos_err = np.abs(pos_fp32 - pos_fp16)

    # Position range
    pos_range = pos.max(axis=0) - pos.min(axis=0)
    print(f"\nPosition range: X=[{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}], "
          f"Y=[{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}], "
          f"Z=[{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}]")
    print(f"Position range size: ({pos_range[0]:.3f}, {pos_range[1]:.3f}, {pos_range[2]:.3f})")

    print(f"\nfp16 position error:")
    print(f"  Mean absolute: ({np.mean(pos_err[:, 0]):.6f}, {np.mean(pos_err[:, 1]):.6f}, {np.mean(pos_err[:, 2]):.6f})")
    print(f"  Max absolute:  ({np.max(pos_err[:, 0]):.6f}, {np.max(pos_err[:, 1]):.6f}, {np.max(pos_err[:, 2]):.6f})")
    print(f"  Relative to range: ({np.max(pos_err[:, 0])/pos_range[0]*100:.4f}%, "
          f"{np.max(pos_err[:, 1])/pos_range[1]*100:.4f}%, "
          f"{np.max(pos_err[:, 2])/pos_range[2]*100:.4f}%)")

    # fp16 max representable: 65504. Check if any positions exceed this
    pos_abs_max = np.abs(pos).max()
    print(f"\n  Max |position| = {pos_abs_max:.3f}")
    print(f"  fp16 max = 65504, fp16 precision at value V: ~V * 2^-10 = V * 0.000977")
    print(f"  At max position {pos_abs_max:.1f}: absolute precision = {pos_abs_max * 2**-10:.6f}")

    # ========== SH DC: VQ vs fp16 ==========
    print("\n" + "="*70)
    print("SH DC: V2 VQ ENCODING")
    print("="*70)

    # V1 packs SH DC as: texture[3] = (f_dc_0, f_dc_1, f_dc_2, sh_R[0])
    # V2 stores just (f_dc_0, f_dc_1, f_dc_2) in VQ, writes to RGBA with A=0
    # But the V1 texture[3] has sh_R[0] in the alpha channel!
    # V2's SH texture has A=0 always.
    print(f"\nV1 SH0 texture: (f_dc_0, f_dc_1, f_dc_2, sh_rest_r[0])")
    print(f"V2 SH0 output:  (f_dc_0, f_dc_1, f_dc_2, 0)")
    print(f"\n*** If SH degree > 0, V2 LOSES sh_rest_r[0] in the alpha channel ***")
    print(f"  (For SH degree 0 this is fine since sh_rest values are all zero)")


def main():
    np.random.seed(42)

    print("="*70)
    print("TEMPORAL VQ ANALYSIS")
    print(f"PLY folder: {PLY_FOLDER}")
    print(f"Frames: {NUM_FRAMES}, VQ K: {VQ_K}, Tracked gaussians: {NUM_TRACKED}")
    print("="*70)

    # List PLY files
    ply_files = sorted([f for f in os.listdir(PLY_FOLDER) if f.endswith('.ply')])[:NUM_FRAMES]
    print(f"\nLoading {len(ply_files)} frames...")

    # Load all frames
    frames = []
    for i, f in enumerate(ply_files):
        path = os.path.join(PLY_FOLDER, f)
        g = load_gaussian_ply(path)
        frames.append(g)
        print(f"  Frame {i}: {len(g['position'])} gaussians")

    # ========== Task 2: V1 vs V2 encoding analysis ==========
    analyze_v1_vs_v2_encoding(frames[0])

    # ========== Task 3: Temporal VQ jitter analysis ==========
    print("\n" + "="*70)
    print("TEMPORAL VQ JITTER ANALYSIS")
    print("="*70)

    n_gauss = len(frames[0]["position"])
    tracked_idx = np.random.choice(n_gauss, min(NUM_TRACKED, n_gauss), replace=False)

    # For each attribute, collect original and VQ-reconstructed values across frames
    attrs = {
        "rotation": {"dim": 4, "original": [], "reconstructed": []},
        "scale": {"dim": 3, "original": [], "reconstructed": []},
        "sh_dc": {"dim": 3, "original": [], "reconstructed": []},
    }

    print(f"\nRunning VQ K={VQ_K} on each frame independently...")
    t0 = time.time()

    for frame_i, g in enumerate(frames):
        # Normalize rotation (as V2 encoder does)
        rot = g["rotation"].astype(np.float32)
        rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
        rot_norm = np.where(rot_norm == 0, 1.0, rot_norm)
        rot_normed = rot / rot_norm

        scale = g["scale"].astype(np.float32)
        sh_dc = g["sh_dc"].astype(np.float32)

        # VQ encode each attribute independently (per-frame codebook)
        _, _, rot_recon = vq_encode(rot_normed, VQ_K)
        _, _, scale_recon = vq_encode(scale, VQ_K)
        _, _, sh_recon = vq_encode(sh_dc, VQ_K)

        # Track selected gaussians
        attrs["rotation"]["original"].append(rot_normed[tracked_idx])
        attrs["rotation"]["reconstructed"].append(rot_recon[tracked_idx])
        attrs["scale"]["original"].append(scale[tracked_idx])
        attrs["scale"]["reconstructed"].append(scale_recon[tracked_idx])
        attrs["sh_dc"]["original"].append(sh_dc[tracked_idx])
        attrs["sh_dc"]["reconstructed"].append(sh_recon[tracked_idx])

        print(f"  Frame {frame_i} done")

    elapsed = time.time() - t0
    print(f"VQ encoding: {elapsed:.1f}s total, {elapsed/NUM_FRAMES:.1f}s/frame")

    # Analyze temporal jitter
    for attr_name, attr_data in attrs.items():
        print(f"\n--- {attr_name.upper()} ---")
        orig = np.array(attr_data["original"])      # (num_frames, num_tracked, dim)
        recon = np.array(attr_data["reconstructed"]) # (num_frames, num_tracked, dim)

        # Per-frame reconstruction error
        per_frame_error = np.linalg.norm(orig - recon, axis=2)  # (num_frames, num_tracked)
        mean_recon_error = np.mean(per_frame_error)
        print(f"  Per-frame VQ reconstruction error (L2):")
        print(f"    Mean: {mean_recon_error:.6f}")
        print(f"    Std:  {np.std(per_frame_error):.6f}")
        print(f"    Max:  {np.max(per_frame_error):.6f}")

        # Temporal change in original values between consecutive frames
        orig_delta = np.linalg.norm(np.diff(orig, axis=0), axis=2)  # (num_frames-1, num_tracked)
        mean_orig_delta = np.mean(orig_delta)
        print(f"\n  Original temporal change (frame-to-frame L2 delta):")
        print(f"    Mean: {mean_orig_delta:.6f}")
        print(f"    Std:  {np.std(orig_delta):.6f}")

        # Temporal change in RECONSTRUCTED values between consecutive frames
        recon_delta = np.linalg.norm(np.diff(recon, axis=0), axis=2)  # (num_frames-1, num_tracked)
        mean_recon_delta = np.mean(recon_delta)
        print(f"\n  Reconstructed temporal change (frame-to-frame L2 delta):")
        print(f"    Mean: {mean_recon_delta:.6f}")
        print(f"    Std:  {np.std(recon_delta):.6f}")

        # Temporal jitter = how much VQ changes the reconstruction even when original is stable
        # For each pair of consecutive frames:
        #   jitter = |recon_delta - orig_delta| / orig_delta
        # If orig changes by X and recon changes by Y, jitter ratio = |Y-X|/X
        mask = orig_delta > 1e-8  # Avoid division by zero
        jitter_ratio = np.zeros_like(orig_delta)
        jitter_ratio[mask] = np.abs(recon_delta[mask] - orig_delta[mask]) / orig_delta[mask]
        print(f"\n  Temporal jitter ratio |recon_delta - orig_delta| / orig_delta:")
        print(f"    Mean:   {np.mean(jitter_ratio[mask]):.4f}")
        print(f"    Median: {np.median(jitter_ratio[mask]):.4f}")
        print(f"    p95:    {np.percentile(jitter_ratio[mask], 95):.4f}")

        # Also measure: for gaussians that barely move, how much does VQ make them jitter?
        low_motion = orig_delta < np.percentile(orig_delta, 25)  # Bottom 25% motion
        if np.any(low_motion):
            low_motion_recon_delta = recon_delta[low_motion]
            low_motion_orig_delta = orig_delta[low_motion]
            print(f"\n  Low-motion gaussians (bottom 25% by original delta):")
            print(f"    Original delta mean:       {np.mean(low_motion_orig_delta):.6f}")
            print(f"    Reconstructed delta mean:  {np.mean(low_motion_recon_delta):.6f}")
            print(f"    VQ-induced jitter ratio:   {np.mean(low_motion_recon_delta) / max(np.mean(low_motion_orig_delta), 1e-10):.2f}x")

        # Signal-to-noise: reconstruction error vs actual temporal change
        snr = mean_orig_delta / max(mean_recon_error, 1e-10)
        print(f"\n  Signal-to-noise (temporal_change / recon_error): {snr:.2f}")
        if snr < 2.0:
            print(f"    *** WARNING: VQ error is comparable to temporal signal! ***")
            print(f"    *** This means VQ noise dominates the temporal variation ***")

    # ========== ROTATION: angular error analysis ==========
    print(f"\n--- ROTATION ANGULAR ERROR ---")
    rot_orig = np.array(attrs["rotation"]["original"])
    rot_recon = np.array(attrs["rotation"]["reconstructed"])

    # Quaternion angular error
    dots = np.abs(np.sum(rot_orig * rot_recon, axis=2))
    dots = np.clip(dots, 0, 1)
    angles = 2 * np.degrees(np.arccos(dots))
    print(f"  Per-frame angular error from VQ:")
    print(f"    Mean:   {np.mean(angles):.3f} degrees")
    print(f"    Median: {np.median(angles):.3f} degrees")
    print(f"    p95:    {np.percentile(angles, 95):.3f} degrees")
    print(f"    Max:    {np.max(angles):.3f} degrees")

    # Temporal angular jitter
    for i in range(NUM_FRAMES - 1):
        dots_orig = np.abs(np.sum(rot_orig[i] * rot_orig[i+1], axis=1))
        dots_orig = np.clip(dots_orig, 0, 1)
        angles_orig = 2 * np.degrees(np.arccos(dots_orig))

        dots_recon = np.abs(np.sum(rot_recon[i] * rot_recon[i+1], axis=1))
        dots_recon = np.clip(dots_recon, 0, 1)
        angles_recon = 2 * np.degrees(np.arccos(dots_recon))

        jitter = np.abs(angles_recon - angles_orig)
        if i == 0:
            print(f"\n  Frame-to-frame angular jitter (|recon_angle_change - orig_angle_change|):")
        print(f"    Frame {i}->{i+1}: mean={np.mean(jitter):.3f} deg, "
              f"p95={np.percentile(jitter, 95):.3f} deg, "
              f"max={np.max(jitter):.3f} deg")

    # ========== OPACITY uint8 analysis ==========
    print(f"\n--- OPACITY uint8 QUANTIZATION ---")
    for frame_i, g in enumerate(frames[:3]):
        opacity_raw = g["opacity"].astype(np.float32)
        opacity_act = 1.0 / (1.0 + np.exp(-opacity_raw))
        op_uint8 = (np.clip(opacity_act, 0, 1) * 255).astype(np.uint8)
        op_recon = op_uint8.astype(np.float32) / 255.0
        op_err = np.abs(opacity_act - op_recon)
        print(f"  Frame {frame_i}: mean_err={np.mean(op_err):.6f}, max_err={np.max(op_err):.6f}")

    print(f"\n{'='*70}")
    print("SUMMARY OF FINDINGS")
    print("="*70)
    print("""
1. CRITICAL BUG - Rotation coordinate mismatch:
   V1 stores rotation in (Z, X, -Y, W) coordinate order in the RGBA texture.
   V2 stores rotation as (rot_0, rot_1, rot_2, rot_3) = (W, X, Y, Z) directly.
   The shader expects V1's order, so V2's W and Z channels are SWAPPED,
   and Y sign is wrong. This causes MASSIVE rotation errors.

2. CRITICAL BUG - Scale axis order mismatch:
   V1: ScaleOpacity R=exp(scale_2), G=exp(scale_0), B=exp(scale_1) (Z,X,Y swizzle)
   V2 decoder: R=exp(S[0]), G=exp(S[1]), B=exp(S[2]) (0,1,2 order - NO swizzle)
   Scale axes 0 and 2 are swapped, axis 1 goes from G to B.

3. Position fp16 vs fp32:
   V1 uses fp32 by default. V2 uses fp16.
   For typical SHARP scene ranges, fp16 error is small but not zero.

4. Temporal VQ jitter:
   Per-frame independent codebooks cause temporal flickering.
   See analysis above for signal-to-noise ratios.
""")


if __name__ == "__main__":
    main()
