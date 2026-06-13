"""
harris.py — Harris corner detector.

Algorithm:
  1. Compute image gradients Ix, Iy (Sobel)
  2. Compute second-moment matrix M per pixel (Gaussian-weighted window):
       M = [[sum(Ix²), sum(IxIy)],
            [sum(IxIy), sum(Iy²)]]
  3. Corner response R = det(M) - k·trace(M)²
     R >> 0  → corner
     R << 0  → edge
     |R| ≈ 0 → flat
  4. Threshold R and suppress non-maxima
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from convolve import convolve2d_naive, gaussian_blur

SOBEL_GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_GY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def harris_response(image: np.ndarray, sigma: float = 1.0,
                    k: float = 0.04) -> np.ndarray:
    """
    Compute the Harris corner response map R.

    Args:
        image: Grayscale float array (H, W).
        sigma: Gaussian window σ for weighting the structure tensor.
        k:     Harris sensitivity (typically 0.04–0.06).

    Returns:
        R: float array (H, W).  Positive peaks = corners.
    """
    if image.ndim == 3:
        gray = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    else:
        gray = image.astype(float)

    ix = convolve2d_naive(gray, SOBEL_GX)
    iy = convolve2d_naive(gray, SOBEL_GY)

    # Structure tensor elements, smoothed by Gaussian window
    ix2 = gaussian_blur(ix * ix, sigma)
    iy2 = gaussian_blur(iy * iy, sigma)
    ixy = gaussian_blur(ix * iy, sigma)

    det   = ix2 * iy2 - ixy ** 2
    trace = ix2 + iy2
    return det - k * trace ** 2


def harris_corners(image: np.ndarray, sigma: float = 1.0, k: float = 0.04,
                   threshold: float = 1e6,
                   min_distance: int = 5) -> tuple[np.ndarray, list[tuple]]:
    """
    Detect corners using the Harris detector.

    Args:
        image:        Input image (grayscale or RGB).
        sigma:        Gaussian window σ.
        k:            Harris sensitivity.
        threshold:    Minimum R value to accept as a corner.
        min_distance: Minimum pixel distance between returned corners.

    Returns:
        R:       Corner response map (H, W float).
        corners: List of (row, col) corner locations, sorted by strength.
    """
    R = harris_response(image, sigma, k)
    candidates = np.argwhere(R > threshold)
    # Sort by response strength descending
    candidates = sorted(candidates, key=lambda p: -R[p[0], p[1]])

    corners = []
    taken = np.zeros(R.shape, dtype=bool)
    for r, c in candidates:
        if not taken[r, c]:
            corners.append((int(r), int(c)))
            r0, r1 = max(0, r-min_distance), min(R.shape[0], r+min_distance+1)
            c0, c1 = max(0, c-min_distance), min(R.shape[1], c+min_distance+1)
            taken[r0:r1, c0:c1] = True

    return R, corners


if __name__ == "__main__":
    img = np.zeros((64, 64), dtype=float)
    img[16:48, 16:48] = 200
    R, corners = harris_corners(img, sigma=1.5, threshold=1e4)
    print(f"Detected {len(corners)} corners")
    for r, c in corners[:8]:
        print(f"  ({r}, {c})  R={R[r,c]:.2e}")
