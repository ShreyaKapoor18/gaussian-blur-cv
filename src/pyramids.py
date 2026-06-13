"""
pyramids.py — Gaussian pyramid, Laplacian pyramid, Difference of Gaussians (DoG).

Gaussian pyramid:  each level = blur + downsample (÷2)
Laplacian pyramid: each level = G[l] - upsample(G[l+1])  (band-pass)
DoG:               G(sigma) - G(k*sigma)  ≈ LoG, used by SIFT
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from convolve import gaussian_blur


def _downsample(image: np.ndarray) -> np.ndarray:
    return image[::2, ::2]


def _upsample(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    H, W = target_shape
    out = np.zeros((H, W), dtype=float)
    sh, sw = image.shape
    out[:sh*2:2, :sw*2:2] = image
    # Simple nearest-neighbour fill for odd rows/cols
    out[1::2, :] = out[::2, :]
    out[:, 1::2] = out[:, ::2]
    return out[:H, :W]


def gaussian_pyramid(image: np.ndarray, levels: int = 4,
                     sigma: float = 1.0) -> list[np.ndarray]:
    """
    Build a Gaussian image pyramid.

    Each level blurs then halves spatial resolution.

    Args:
        image:  2D float array (H, W).
        levels: Number of levels (including the original).
        sigma:  σ for the blur at each level.

    Returns:
        List of length `levels`, each a 2D float array.
    """
    pyramid = [image.astype(float)]
    for _ in range(levels - 1):
        blurred = gaussian_blur(pyramid[-1], sigma)
        pyramid.append(_downsample(blurred))
    return pyramid


def laplacian_pyramid(image: np.ndarray, levels: int = 4,
                      sigma: float = 1.0) -> list[np.ndarray]:
    """
    Build a Laplacian pyramid (band-pass decomposition).

    L[l] = G[l] - upsample(G[l+1])

    The top level is the residual low-frequency component.
    Summing all levels reconstructs the original image exactly.

    Returns:
        List of length `levels`. Last entry is the coarsest Gaussian level.
    """
    gpyr = gaussian_pyramid(image, levels, sigma)
    lpyr = []
    for l in range(levels - 1):
        up = _upsample(gpyr[l + 1], gpyr[l].shape)
        lpyr.append(gpyr[l] - up)
    lpyr.append(gpyr[-1])   # residual
    return lpyr


def reconstruct_from_laplacian(lpyr: list[np.ndarray]) -> np.ndarray:
    """Reconstruct image from its Laplacian pyramid."""
    out = lpyr[-1].copy()
    for band in reversed(lpyr[:-1]):
        out = _upsample(out, band.shape) + band
    return out


def dog(image: np.ndarray, sigma: float = 1.0,
        k: float = 1.6) -> np.ndarray:
    """
    Difference of Gaussians — approximates the Laplacian of Gaussian (LoG).

    DoG(x,y) = G(x,y;σ) - G(x,y;kσ)

    Used in SIFT to detect scale-space extrema (blob-like keypoints).

    Args:
        image: 2D float array.
        sigma: Base scale.
        k:     Scale ratio between the two Gaussians (SIFT uses 1.6).

    Returns:
        DoG response map, same shape as input.
    """
    g1 = gaussian_blur(image.astype(float), sigma)
    g2 = gaussian_blur(image.astype(float), k * sigma)
    return g1 - g2


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (128, 128)).astype(float)
    gpyr = gaussian_pyramid(img, levels=4)
    lpyr = laplacian_pyramid(img, levels=4)
    recon = reconstruct_from_laplacian(lpyr)

    print("Gaussian pyramid shapes:", [g.shape for g in gpyr])
    print("Laplacian pyramid shapes:", [l.shape for l in lpyr])
    print(f"Reconstruction error: {np.abs(recon - img).max():.2e}")

    d = dog(img, sigma=1.0)
    print(f"DoG  min={d.min():.1f}  max={d.max():.1f}")
