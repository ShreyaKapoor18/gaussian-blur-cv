"""
stereo.py — stereo vision and disparity estimation.

Assumes rectified stereo pairs (epipolar lines are horizontal rows).

Block matching:
  For each pixel (i, j) in the left image, search the same row in the
  right image over [j - max_disp, j] and find the shift d that minimises
  the Sum of Absolute Differences (SAD) over a block_size × block_size patch.

  disparity[i, j] = d   (≥ 0 because objects are shifted right→left)

Depth from disparity:
  Z = (f · B) / d
  where f = focal length (pixels), B = baseline (same units as Z).
"""

import numpy as np


def _sad(patch_l: np.ndarray, patch_r: np.ndarray) -> float:
    return float(np.abs(patch_l.astype(float) - patch_r.astype(float)).sum())


def disparity_map(left: np.ndarray, right: np.ndarray,
                  block_size: int = 7,
                  max_disp: int = 32) -> np.ndarray:
    """
    Compute a dense disparity map using block matching (SAD cost).

    Args:
        left, right: Rectified grayscale images (H, W), same shape.
        block_size:  Side of the matching block (odd integer).
        max_disp:    Maximum disparity to search (pixels).

    Returns:
        disp: (H, W) float array of disparity values.
              Pixels where no valid match exists are 0.
    """
    if left.ndim == 3:
        left  = 0.299*left[:,:,0]  + 0.587*left[:,:,1]  + 0.114*left[:,:,2]
        right = 0.299*right[:,:,0] + 0.587*right[:,:,1] + 0.114*right[:,:,2]

    H, W = left.shape
    half = block_size // 2
    disp = np.zeros((H, W), dtype=float)

    for i in range(half, H - half):
        for j in range(half, W - half):
            patch_l = left[i-half:i+half+1, j-half:j+half+1]
            best_sad = np.inf
            best_d   = 0
            for d in range(0, max_disp + 1):
                jd = j - d
                if jd - half < 0:
                    break
                patch_r = right[i-half:i+half+1, jd-half:jd+half+1]
                s = _sad(patch_l, patch_r)
                if s < best_sad:
                    best_sad = s
                    best_d   = d
            disp[i, j] = best_d

    return disp


def depth_from_disparity(disparity: np.ndarray,
                          focal_length: float,
                          baseline: float) -> np.ndarray:
    """
    Convert a disparity map to a depth map using the thin-lens formula.

    Z = (f · B) / d,  with d=0 mapped to 0 (unknown depth).

    Args:
        disparity:    (H, W) disparity array (pixels).
        focal_length: Camera focal length (pixels).
        baseline:     Distance between the two cameras (meters or any unit).

    Returns:
        depth: (H, W) array in the same units as baseline.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disparity > 0, focal_length * baseline / disparity, 0)
    return depth


def make_stereo_pair(H: int = 48, W: int = 64,
                     true_disp: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic rectified stereo pair with known disparity."""
    rng = np.random.default_rng(7)
    left = rng.integers(30, 200, (H, W)).astype(np.uint8)
    left[12:36, 20:44] = 200   # bright rectangle = foreground object

    # Right image is the left image shifted by true_disp columns
    right = np.zeros_like(left)
    right[:, true_disp:] = left[:, :W-true_disp]
    return left, right


if __name__ == "__main__":
    left, right = make_stereo_pair(true_disp=6)
    disp = disparity_map(left, right, block_size=7, max_disp=16)
    # Measure accuracy inside the foreground rectangle
    roi = disp[12:36, 20:44]
    print(f"True disparity: 6  —  mean inside rectangle: {roi.mean():.1f}")
    depth = depth_from_disparity(disp, focal_length=500, baseline=0.1)
    print(f"Depth at foreground (true ≈ 8.33 m): {depth[24,32]:.2f} m")
