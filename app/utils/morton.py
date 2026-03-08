"""3D Morton code computation and sorting.

Replicates the exact logic from ThreeDGaussiansLibrary::Sort3dMortonOrder.
"""

import numpy as np


def colmap_to_ue(positions: np.ndarray) -> np.ndarray:
    """Convert COLMAP (x,y,z) positions to UE convention.

    UE = (z, x, -y) * 100.0

    Args:
        positions: (N, 3) array in COLMAP coordinates.

    Returns:
        (N, 3) array in UE coordinates.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    return np.column_stack([z * 100.0, x * 100.0, -y * 100.0]).astype(np.float32)


def morton_code_3(v: np.ndarray) -> np.ndarray:
    """Compute MortonCode3 for an array of uint32 values.

    Spreads the bits of a 10-bit integer so that each bit is separated
    by 2 zero bits. Matches FMath::MortonCode3 in UE.

    Input must be in range [0, 1023].
    """
    v = v.astype(np.uint32)
    v = (v | (v << 16)) & np.uint32(0x030000FF)
    v = (v | (v << 8)) & np.uint32(0x0300F00F)
    v = (v | (v << 4)) & np.uint32(0x030C30C3)
    v = (v | (v << 2)) & np.uint32(0x09249249)
    return v


def sort_3d_morton_order(
    positions_colmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort gaussians by 3D Morton order.

    Exactly replicates Sort3dMortonOrder from the UE plugin.

    Args:
        positions_colmap: (N, 3) float32 array of positions in COLMAP coords.

    Returns:
        sorted_indices: (N,) int array - indices into the original array, sorted by Morton code.
        min_pos: (3,) float32 - minimum UE-coordinate position (with *100).
        max_pos: (3,) float32 - maximum UE-coordinate position (with *100).
    """
    # Convert to UE coordinates (with *100 scaling)
    ue_positions = colmap_to_ue(positions_colmap)

    # Find min and max
    min_pos = ue_positions.min(axis=0).astype(np.float32)
    max_pos = ue_positions.max(axis=0).astype(np.float32)

    # Normalize to 0..1
    extent = max_pos - min_pos
    # Avoid division by zero for degenerate axes
    extent = np.where(extent == 0, 1.0, extent)
    normalized = (ue_positions - min_pos) / extent

    # Compute Morton codes
    # Quantize to 0..1023
    qx = (normalized[:, 0] * 1023).astype(np.uint32)
    qy = (normalized[:, 1] * 1023).astype(np.uint32)
    qz = (normalized[:, 2] * 1023).astype(np.uint32)

    morton = morton_code_3(qx) | (morton_code_3(qy) << np.uint32(1)) | (morton_code_3(qz) << np.uint32(2))

    # Stable sort by Morton code (matches RadixSort32 behavior)
    sorted_indices = np.argsort(morton, kind="stable").astype(np.int32)

    return sorted_indices, min_pos, max_pos
