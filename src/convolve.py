"""
convolve.py — 2D convolution implementations.

Two versions:
  - convolve2d_naive:     O(H·W·k²)  straightforward nested loops (via numpy)
  - convolve2d_separable: O(H·W·2k)  two 1D passes, exploiting Gaussian separability

Both operate on a 2D grayscale float array and return the same result.
The separable version is significantly faster for large kernels.
"""

import numpy as np
from kernel import gaussian_kernel


def convolve2d_naive(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve image with kernel using zero-padding (same output size).

    For each output pixel (i, j):
        out[i, j] = sum over (kr, kc) of image[i+kr-half, j+kc-half] * kernel[kr, kc]

    Args:
        image:  2D float array, shape (H, W)
        kernel: 2D float array, shape (k, k), must be square and odd-sized

    Returns:
        2D float array, same shape as image (zero-padded borders)
    """
    H, W = image.shape
    k = kernel.shape[0]
    half = k // 2

    # Pad with zeros so output is the same size as input
    padded = np.pad(image, half, mode='constant', constant_values=0)
    out = np.zeros_like(image, dtype=float)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+k, j:j+k]
            out[i, j] = np.sum(region * kernel)

    return out


def convolve2d_separable(image: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """
    Separable 2D convolution: two 1D passes (rows then columns).

    Because G(x,y) = G(x)·G(y), a 2D Gaussian kernel is rank-1 separable.
    This reduces cost from O(k²) to O(2k) multiplications per pixel.

    Args:
        image:     2D float array, shape (H, W)
        kernel_1d: 1D float array of length k (the Gaussian 1D kernel)

    Returns:
        2D float array, same shape as image
    """
    H, W = image.shape
    k = len(kernel_1d)
    half = k // 2

    # Pass 1: convolve along rows
    padded_h = np.pad(image, [(0, 0), (half, half)], mode='constant')
    tmp = np.zeros_like(image, dtype=float)
    for j in range(W):
        tmp[:, j] = padded_h[:, j:j+k] @ kernel_1d

    # Pass 2: convolve along columns
    padded_v = np.pad(tmp, [(half, half), (0, 0)], mode='constant')
    out = np.zeros_like(image, dtype=float)
    for i in range(H):
        out[i, :] = padded_v[i:i+k, :].T @ kernel_1d

    return out


def gaussian_blur(image: np.ndarray, sigma: float, size: int | None = None) -> np.ndarray:
    """
    Apply Gaussian blur to a grayscale image using the separable method.

    Args:
        image: 2D float array
        sigma: Gaussian standard deviation
        size:  Kernel size (odd integer). Defaults to 2*ceil(3σ)+1.

    Returns:
        Blurred image, same shape as input, values in [0, 255]
    """
    import math
    if size is None:
        size = 2 * math.ceil(3 * sigma) + 1

    k2d = gaussian_kernel(sigma, size)
    k1d = k2d[size // 2]          # centre row == the 1D Gaussian (after normalization)
    k1d = k1d / k1d.sum()         # re-normalize the 1D slice

    blurred = convolve2d_separable(image.astype(float), k1d)
    return np.clip(blurred, 0, 255)


if __name__ == "__main__":
    # Verify naive and separable give the same result
    import math
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (32, 32)).astype(float)

    sigma = 1.0
    size = 2 * math.ceil(3 * sigma) + 1
    k2d = gaussian_kernel(sigma, size)
    k1d = k2d[size // 2]; k1d = k1d / k1d.sum()

    out_naive = convolve2d_naive(img, k2d)
    out_sep   = convolve2d_separable(img, k1d)

    diff = np.abs(out_naive - out_sep).max()
    print(f"Max difference between naive and separable: {diff:.2e}")
    assert diff < 1e-9, "Mismatch!"
    print("Both methods agree.")
