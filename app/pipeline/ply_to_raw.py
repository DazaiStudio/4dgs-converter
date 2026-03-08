"""PLY to RAW converter for UE GaussianStreamer plugin.

Precisely replicates the conversion logic from:
  - GSRawFrameConverter.cpp (binary output, metadata)
  - ThreeDGaussiansLibrary.cpp (WriteGaussianDataToTexture, Sort3dMortonOrder)
"""

import json
import math
import os
from typing import Callable, Optional

import numpy as np

from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order

# Precision enum values matching EGSRawPrecision
PRECISION_FULL = 0   # 32-bit float, 16 bytes/pixel
PRECISION_HALF = 1   # 16-bit float, 8 bytes/pixel

NUM_SH_TEXTURES = 12

# SH degree -> number of SH textures needed
SH_DEGREE_TO_TEXTURES = {0: 1, 1: 3, 2: 7, 3: 12}


def _bytes_per_pixel(precision: int) -> int:
    return 16 if precision == PRECISION_FULL else 8


def _write_texture_binary(
    texture_data: np.ndarray,
    file_path: str,
    precision: int,
) -> int:
    """Write a texture's RGBA data to a binary file.

    Args:
        texture_data: (pixel_count, 4) float32 array.
        file_path: Output path.
        precision: PRECISION_FULL or PRECISION_HALF.

    Returns:
        File size in bytes.
    """
    if precision == PRECISION_FULL:
        # Direct float32 write
        raw = texture_data.astype(np.float32).tobytes()
    else:
        # Convert to float16
        raw = texture_data.astype(np.float16).tobytes()

    with open(file_path, "wb") as f:
        f.write(raw)

    return len(raw)


def _pack_textures(
    gaussians: dict[str, np.ndarray],
    sorted_indices: np.ndarray,
    texture_size: int,
) -> list[np.ndarray]:
    """Pack Gaussian data into 15 RGBA textures.

    Exactly replicates WriteGaussianDataToTexture from ThreeDGaussiansLibrary.cpp.

    Returns:
        List of 15 arrays, each (texture_size*texture_size, 4) float32.
    """
    n_gaussians = len(sorted_indices)
    n_pixels = texture_size * texture_size

    # Initialize 15 textures with zeros
    textures = [np.zeros((n_pixels, 4), dtype=np.float32) for _ in range(15)]

    # Fill excess pixels to match UE's behavior:
    # Position: FLinearColor(0, 0, -0) → alpha defaults to 1.0
    # Rotation: FQuat4f::Identity = (0,0,0,1) → FLinearColor(0, 0, 0, 1)
    # ScaleOpacity & SH: all zeros (already correct)
    if n_pixels > n_gaussians:
        textures[0][n_gaussians:, 3] = 1.0   # position alpha = 1.0
        textures[1][n_gaussians:, 3] = 1.0   # rotation W = 1.0

    # Reorder all data by sorted indices
    idx = sorted_indices

    pos = gaussians["position"][idx]        # (N, 3) COLMAP coords
    sh_dc = gaussians["sh_dc"][idx]          # (N, 3)
    sh_r = gaussians["sh_rest_r"][idx]       # (N, 15)
    sh_g = gaussians["sh_rest_g"][idx]       # (N, 15)
    sh_b = gaussians["sh_rest_b"][idx]       # (N, 15)
    opacity = gaussians["opacity"][idx]      # (N,)
    scale = gaussians["scale"][idx]          # (N, 3)
    rot = gaussians["rotation"][idx]         # (N, 4) rot_0, rot_1, rot_2, rot_3

    # --- Quaternion normalization ---
    # UE: FQuat4f(rotation[1], rotation[2], rotation[3], rotation[0]).GetNormalized()
    # So quat = (X=rot_1, Y=rot_2, Z=rot_3, W=rot_0)
    qx = rot[:, 1]
    qy = rot[:, 2]
    qz = rot[:, 3]
    qw = rot[:, 0]
    qlen = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qlen = np.where(qlen == 0, 1.0, qlen)
    qx /= qlen
    qy /= qlen
    qz /= qlen
    qw /= qlen

    # --- Scale activation ---
    scale_activated = np.exp(scale)  # (N, 3) - exp(scale_0), exp(scale_1), exp(scale_2)

    # --- Opacity activation (sigmoid) ---
    opacity_activated = 1.0 / (1.0 + np.exp(-opacity))

    # --- Texture 0: Position ---
    # FLinearColor(pos.Z, pos.X, -pos.Y, 0)  (COLMAP coords, no *100)
    textures[0][:n_gaussians, 0] = pos[:, 2]      # R = pos.Z
    textures[0][:n_gaussians, 1] = pos[:, 0]      # G = pos.X
    textures[0][:n_gaussians, 2] = -pos[:, 1]     # B = -pos.Y
    # A = 0 (already zero)

    # --- Texture 1: Rotation ---
    # FLinearColor(rotNorm.Z, rotNorm.X, -rotNorm.Y, rotNorm.W)
    textures[1][:n_gaussians, 0] = qz             # R = quat.Z
    textures[1][:n_gaussians, 1] = qx             # G = quat.X
    textures[1][:n_gaussians, 2] = -qy            # B = -quat.Y
    textures[1][:n_gaussians, 3] = qw             # A = quat.W

    # --- Texture 2: Scale + Opacity ---
    # FLinearColor(scaleAct.Z, scaleAct.X, scaleAct.Y, opacityAct)
    textures[2][:n_gaussians, 0] = scale_activated[:, 2]  # R = exp(scale_2)
    textures[2][:n_gaussians, 1] = scale_activated[:, 0]  # G = exp(scale_0)
    textures[2][:n_gaussians, 2] = scale_activated[:, 1]  # B = exp(scale_1)
    textures[2][:n_gaussians, 3] = opacity_activated       # A = sigmoid(opacity)

    # --- Textures 3-14: SH coefficients ---
    # Texture 3: (f_dc_0, f_dc_1, f_dc_2, sh_R[0])
    textures[3][:n_gaussians, 0] = sh_dc[:, 0]
    textures[3][:n_gaussians, 1] = sh_dc[:, 1]
    textures[3][:n_gaussians, 2] = sh_dc[:, 2]
    textures[3][:n_gaussians, 3] = sh_r[:, 0]

    # Textures 4-14: Interleaved SH pattern
    # The pattern from the C++ code packs 4 values per texture in this order:
    #   tex4:  G[0], B[0], R[1], G[1]
    #   tex5:  B[1], R[2], G[2], B[2]
    #   tex6:  R[3], G[3], B[3], R[4]
    #   tex7:  G[4], B[4], R[5], G[5]
    #   tex8:  B[5], R[6], G[6], B[6]
    #   tex9:  R[7], G[7], B[7], R[8]
    #   tex10: G[8], B[8], R[9], G[9]
    #   tex11: B[9], R[10], G[10], B[10]
    #   tex12: R[11], G[11], B[11], R[12]
    #   tex13: G[12], B[12], R[13], G[13]
    #   tex14: B[13], R[14], G[14], B[14]

    # Build a flat interleave sequence starting from index 1 of the SH pattern
    # After DC and sh_R[0], the pattern continues: G[0], B[0], R[1], G[1], B[1], ...
    sh_interleaved = []
    for i in range(15):
        if i == 0:
            # R[0] already placed in tex3.A, start with G[0]
            sh_interleaved.append(("G", i))
            sh_interleaved.append(("B", i))
        else:
            sh_interleaved.append(("R", i))
            sh_interleaved.append(("G", i))
            sh_interleaved.append(("B", i))

    # sh_interleaved now has 44 entries (G0,B0 + 14*(R,G,B) = 2+42=44)
    # Pack into textures 4-14 (11 textures * 4 channels = 44 slots) - perfect!
    channel_idx = 0
    for tex_i in range(4, 15):
        for ch in range(4):
            if channel_idx < len(sh_interleaved):
                color, coeff_idx = sh_interleaved[channel_idx]
                if color == "R":
                    textures[tex_i][:n_gaussians, ch] = sh_r[:, coeff_idx]
                elif color == "G":
                    textures[tex_i][:n_gaussians, ch] = sh_g[:, coeff_idx]
                else:  # B
                    textures[tex_i][:n_gaussians, ch] = sh_b[:, coeff_idx]
                channel_idx += 1

    # Excess pixels remain zero (already initialized)

    return textures


def _prune_by_contribution(
    gaussians: dict[str, np.ndarray],
    keep_ratio: float,
) -> dict[str, np.ndarray]:
    """Keep top N% of gaussians ranked by visual contribution.

    Contribution = sigmoid(opacity) * volume, where volume = product of exp(scale_i).
    Higher contribution means the gaussian occupies more visible space.

    Args:
        gaussians: Dict of gaussian arrays.
        keep_ratio: Fraction to keep (e.g. 0.5 for top 50%).

    Returns:
        New dict with pruned arrays.
    """
    n = len(gaussians["opacity"])
    keep_count = int(n * keep_ratio)
    if keep_count >= n:
        return gaussians

    opacity_activated = 1.0 / (1.0 + np.exp(-gaussians["opacity"]))
    scale_activated = np.exp(gaussians["scale"])
    volume = scale_activated[:, 0] * scale_activated[:, 1] * scale_activated[:, 2]
    contribution = opacity_activated * volume

    top_indices = np.argpartition(contribution, -keep_count)[-keep_count:]
    return {key: arr[top_indices] for key, arr in gaussians.items()}


def convert_ply_to_raw(
    ply_path: str,
    output_folder: str,
    frame_index: int,
    position_precision: int = PRECISION_FULL,
    rotation_precision: int = PRECISION_HALF,
    scale_opacity_precision: int = PRECISION_HALF,
    sh_precision: int = PRECISION_HALF,
    sh_degree: int = 3,
    prune_keep_ratio: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Convert a single PLY file to RAW binary format.

    Args:
        ply_path: Path to the input PLY file.
        output_folder: Base output folder (frame subfolder will be created).
        frame_index: Frame index for naming.
        position_precision: PRECISION_FULL (0) or PRECISION_HALF (1).
        rotation_precision: PRECISION_FULL (0) or PRECISION_HALF (1).
        scale_opacity_precision: PRECISION_FULL (0) or PRECISION_HALF (1).
        sh_precision: PRECISION_FULL (0) or PRECISION_HALF (1).
        prune_keep_ratio: If set, keep top N% by contribution (e.g. 0.5 for 50%).
        progress_callback: Optional callback for log messages.

    Returns:
        Frame metadata dict matching the JSON format.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # Create frame folder
    frame_folder_name = f"frame_{frame_index:04d}"
    frame_folder = os.path.join(output_folder, frame_folder_name)
    os.makedirs(frame_folder, exist_ok=True)

    # Load PLY
    log(f"Loading PLY: {os.path.basename(ply_path)}")
    gaussians = load_gaussian_ply(ply_path)
    original_count = len(gaussians["position"])
    log(f"  Loaded {original_count} gaussians")

    # Prune by contribution
    if prune_keep_ratio is not None and prune_keep_ratio < 1.0:
        gaussians = _prune_by_contribution(gaussians, prune_keep_ratio)
        gaussian_count = len(gaussians["position"])
        removed = original_count - gaussian_count
        log(f"  Pruned {removed} gaussians (keep top {prune_keep_ratio*100:.0f}%) -> {gaussian_count} remaining")
    else:
        gaussian_count = original_count

    # Morton sort
    log("  Sorting by 3D Morton order...")
    sorted_indices, min_pos, max_pos = sort_3d_morton_order(gaussians["position"])

    # Calculate texture size
    texture_size = math.ceil(math.sqrt(gaussian_count))
    log(f"  Texture size: {texture_size}x{texture_size}")

    # Pack textures
    log("  Packing texture data...")
    textures = _pack_textures(gaussians, sorted_indices, texture_size)

    # Write binary files
    num_sh = SH_DEGREE_TO_TEXTURES.get(sh_degree, NUM_SH_TEXTURES)
    texture_names = [
        "position", "rotation", "scaleOpacity",
    ] + [f"sh_{i}" for i in range(num_sh)]
    precisions = [
        position_precision,
        rotation_precision,
        scale_opacity_precision,
    ] + [sh_precision] * num_sh

    file_sizes = []
    for i, (name, prec) in enumerate(zip(texture_names, precisions)):
        file_path = os.path.join(frame_folder, f"{name}.bin")
        size = _write_texture_binary(textures[i], file_path, prec)
        file_sizes.append(size)
        log(f"  Wrote {name}.bin ({size:,} bytes)")

    # Build metadata
    metadata = {
        "frameIndex": frame_index,
        "textureWidth": texture_size,
        "textureHeight": texture_size,
        "gaussianCount": gaussian_count,
        "minPosition": {
            "x": float(min_pos[0]),
            "y": float(min_pos[1]),
            "z": float(min_pos[2]),
        },
        "maxPosition": {
            "x": float(max_pos[0]),
            "y": float(max_pos[1]),
            "z": float(max_pos[2]),
        },
        "positionPrecision": position_precision,
        "rotationPrecision": rotation_precision,
        "scaleOpacityPrecision": scale_opacity_precision,
        "shPrecision": sh_precision,
        "positionFileSize": file_sizes[0],
        "rotationFileSize": file_sizes[1],
        "scaleOpacityFileSize": file_sizes[2],
        "shFileSizes": file_sizes[3:],
    }

    # Save metadata.json
    metadata_path = os.path.join(frame_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent="\t")

    log(f"  Saved metadata.json")
    return metadata


def convert_ply_sequence(
    ply_folder: str,
    output_folder: str,
    sequence_name: str,
    target_fps: float = 24.0,
    sh_degree: int = 0,
    position_precision: int = PRECISION_FULL,
    rotation_precision: int = PRECISION_HALF,
    scale_opacity_precision: int = PRECISION_HALF,
    sh_precision: int = PRECISION_HALF,
    prune_keep_ratio: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Convert a folder of PLY files to RAW sequence.

    Args:
        ply_folder: Folder containing .ply files.
        output_folder: Output folder for RAW files.
        sequence_name: Name for the sequence.
        target_fps: Target playback FPS.
        sh_degree: SH degree (0, 1, 2, or 3).
        position_precision: PRECISION_FULL or PRECISION_HALF.
        rotation_precision: PRECISION_FULL or PRECISION_HALF.
        scale_opacity_precision: PRECISION_FULL or PRECISION_HALF.
        sh_precision: PRECISION_FULL or PRECISION_HALF.
        prune_keep_ratio: If set, keep top N% by contribution (e.g. 0.5 for 50%).
        progress_callback: Callback for log messages.
        frame_progress_callback: Callback (current_frame, total_frames).

    Returns:
        Sequence metadata dict.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # Find PLY files
    ply_files = sorted([
        f for f in os.listdir(ply_folder)
        if f.lower().endswith(".ply")
    ])

    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {ply_folder}")

    os.makedirs(output_folder, exist_ok=True)

    log(f"Found {len(ply_files)} PLY files")
    log(f"Output: {output_folder}")

    frame_metadatas = []
    skipped = 0
    for i, ply_file in enumerate(ply_files):
        # Resume: skip frames that already have metadata.json
        frame_folder = os.path.join(output_folder, f"frame_{i:04d}")
        metadata_path = os.path.join(frame_folder, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            frame_metadatas.append(metadata)
            skipped += 1
            if frame_progress_callback:
                frame_progress_callback(i + 1, len(ply_files))
            continue

        if skipped > 0 and i == skipped:
            log(f"Skipped {skipped} already converted frames (resume)")

        log(f"\n--- Frame {i}/{len(ply_files)} ---")
        ply_path = os.path.join(ply_folder, ply_file)

        metadata = convert_ply_to_raw(
            ply_path=ply_path,
            output_folder=output_folder,
            frame_index=i,
            position_precision=position_precision,
            rotation_precision=rotation_precision,
            scale_opacity_precision=scale_opacity_precision,
            sh_precision=sh_precision,
            sh_degree=sh_degree,
            prune_keep_ratio=prune_keep_ratio,
            progress_callback=progress_callback,
        )
        frame_metadatas.append(metadata)

        if frame_progress_callback:
            frame_progress_callback(i + 1, len(ply_files))

    # Build sequence metadata
    sequence_metadata = {
        "sequenceName": sequence_name,
        "frameCount": len(ply_files),
        "targetFPS": target_fps,
        "shDegree": sh_degree,
        "frameFolders": [f"frame_{i:04d}" for i in range(len(ply_files))],
    }

    # Save sequence.json
    seq_path = os.path.join(output_folder, "sequence.json")
    with open(seq_path, "w") as f:
        json.dump(sequence_metadata, f, indent="\t")

    log(f"\nSaved sequence.json ({len(ply_files)} frames)")
    return sequence_metadata
