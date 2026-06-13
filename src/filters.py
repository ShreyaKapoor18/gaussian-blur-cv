"""
filters.py — classic CV preprocessing filters and edge detectors.

All functions operate on numpy arrays (float or uint8).
"""

import numpy as np
from convolve import gaussian_blur, convolve2d_naive


# ── Colour / intensity ────────────────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using BT.601 luminance weights:
        Y = 0.299·R + 0.587·G + 0.114·B
    """
    if image.ndim == 2:
        return image.astype(float)
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    return (0.299*r + 0.587*g + 0.114*b)


def threshold(image: np.ndarray, value: int = 128) -> np.ndarray:
    """Binary threshold: pixels above value → 255, below → 0."""
    gray = to_grayscale(image)
    return np.where(gray > value, 255, 0).astype(np.uint8)


def normalize(image: np.ndarray) -> np.ndarray:
    """Stretch intensity range to [0, 255]."""
    mn, mx = image.min(), image.max()
    if mx == mn:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - mn) / (mx - mn) * 255).astype(np.uint8)


# ── Blur ─────────────────────────────────────────────────────────────────────

def blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur. Works on grayscale or RGB (processes each channel)."""
    if image.ndim == 2:
        return np.clip(gaussian_blur(image, sigma), 0, 255).astype(np.uint8)
    channels = [gaussian_blur(image[:,:,c], sigma) for c in range(image.shape[2])]
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


# ── Edge detection ────────────────────────────────────────────────────────────

# Sobel kernels — detect horizontal and vertical gradients
SOBEL_GX = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=float)

SOBEL_GY = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=float)

# Prewitt kernels — similar to Sobel but uniform weights
PREWITT_GX = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=float)

PREWITT_GY = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]], dtype=float)

# Laplacian — second derivative, detects regions of rapid intensity change
LAPLACIAN = np.array([[ 0,  1,  0],
                       [ 1, -4,  1],
                       [ 0,  1,  0]], dtype=float)


def sobel(image: np.ndarray, pre_blur_sigma: float = 1.0,
          threshold: int = 60) -> np.ndarray:
    """
    Sobel edge detector.

    Steps:
      1. Convert to grayscale
      2. Apply Gaussian blur (reduces noise before differentiation)
      3. Compute Gx and Gy with Sobel kernels
      4. Magnitude = sqrt(Gx² + Gy²)
      5. Threshold to binary

    Args:
        image:           Input image (grayscale or RGB)
        pre_blur_sigma:  σ for the Gaussian pre-blur (0 = skip)
        threshold:       Magnitude threshold for binary output

    Returns:
        Binary edge map (uint8, values 0 or 255)
    """
    gray = to_grayscale(image)
    if pre_blur_sigma > 0:
        gray = gaussian_blur(gray, pre_blur_sigma)

    gx = convolve2d_naive(gray, SOBEL_GX)
    gy = convolve2d_naive(gray, SOBEL_GY)
    magnitude = np.sqrt(gx**2 + gy**2)

    return np.where(magnitude > threshold, 255, 0).astype(np.uint8)


def prewitt(image: np.ndarray, pre_blur_sigma: float = 1.0,
            threshold: int = 60) -> np.ndarray:
    """Prewitt edge detector. Same pipeline as Sobel, different kernels."""
    gray = to_grayscale(image)
    if pre_blur_sigma > 0:
        gray = gaussian_blur(gray, pre_blur_sigma)

    gx = convolve2d_naive(gray, PREWITT_GX)
    gy = convolve2d_naive(gray, PREWITT_GY)
    magnitude = np.sqrt(gx**2 + gy**2)

    return np.where(magnitude > threshold, 255, 0).astype(np.uint8)


def laplacian_edges(image: np.ndarray, pre_blur_sigma: float = 1.0,
                    threshold: int = 20) -> np.ndarray:
    """
    Laplacian edge detector (second derivative).

    More sensitive to noise than Sobel — pre-blur is important.
    The LoG (Laplacian of Gaussian) fuses both steps into one kernel.
    """
    gray = to_grayscale(image)
    if pre_blur_sigma > 0:
        gray = gaussian_blur(gray, pre_blur_sigma)

    lap = convolve2d_naive(gray, LAPLACIAN)
    return np.where(np.abs(lap) > threshold, 255, 0).astype(np.uint8)


# ── Histogram equalization ────────────────────────────────────────────────────

def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization to improve contrast.

    Steps:
      1. Compute intensity histogram (256 bins)
      2. Compute cumulative distribution function (CDF)
      3. Map each intensity via: new = round((CDF(v) - CDF_min) / (N - CDF_min) * 255)

    Works on grayscale images.
    """
    gray = to_grayscale(image).astype(np.uint8)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))

    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    N = gray.size

    lut = np.round((cdf - cdf_min) / (N - cdf_min) * 255).astype(np.uint8)
    lut[cdf == 0] = 0

    return lut[gray]
