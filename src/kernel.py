"""
kernel.py — build a normalized discrete Gaussian kernel.

The kernel is constructed by sampling G(x, y) = exp(-(x²+y²) / 2σ²)
at integer coordinates, then normalizing so all weights sum to 1.

Kernel size and σ are independent parameters:
  - σ controls the shape (how fast weights fall off with distance)
  - size controls how far out you sample

Rule of thumb: size = 2 * ceil(3σ) + 1  (covers ±3σ, 99.7% of mass)
"""

import math
import numpy as np


def gaussian_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    """
    Build a normalized 2D Gaussian kernel.

    Args:
        sigma: Standard deviation of the Gaussian.
        size:  Kernel side length (must be odd). Defaults to 2*ceil(3σ)+1.

    Returns:
        2D numpy array of shape (size, size) with weights summing to 1.
    """
    if size is None:
        size = 2 * math.ceil(3 * sigma) + 1
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    half = size // 2
    coords = np.arange(-half, half + 1)          # e.g. [-1, 0, 1] for size=3
    x, y = np.meshgrid(coords, coords)

    raw = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return raw / raw.sum()                        # normalize


def print_kernel(k: np.ndarray, decimals: int = 4) -> None:
    """Pretty-print a kernel matrix."""
    for row in k:
        print("  " + "  ".join(f"{v:.{decimals}f}" for v in row))


if __name__ == "__main__":
    for sigma in [0.5, 1.0, 2.0]:
        size = 2 * math.ceil(3 * sigma) + 1
        k = gaussian_kernel(sigma, size)
        print(f"\nσ={sigma}, size={size}×{size}")
        print_kernel(k)
        print(f"  centre={k[size//2, size//2]:.4f}  corner={k[0,0]:.4f}  sum={k.sum():.6f}")
