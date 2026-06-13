"""
canny.py — Canny edge detector from scratch.

Pipeline:
  1. Gaussian blur (noise suppression)
  2. Sobel gradient magnitude + direction
  3. Non-maximum suppression (thin edges to 1 px)
  4. Double threshold (strong / weak edges)
  5. Hysteresis (keep weak edges connected to strong ones)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from convolve import gaussian_blur, convolve2d_naive

SOBEL_GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_GY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def _gradients(gray: np.ndarray):
    gx = convolve2d_naive(gray, SOBEL_GX)
    gy = convolve2d_naive(gray, SOBEL_GY)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) * 180 / np.pi % 180   # 0–180°
    return mag, ang


def _nms(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    H, W = mag.shape
    out = np.zeros_like(mag)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = ang[i, j]
            m = mag[i, j]
            # Quantise angle to 4 directions
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                q, r = mag[i, j+1], mag[i, j-1]
            elif 22.5 <= a < 67.5:
                q, r = mag[i+1, j-1], mag[i-1, j+1]
            elif 67.5 <= a < 112.5:
                q, r = mag[i+1, j], mag[i-1, j]
            else:
                q, r = mag[i-1, j-1], mag[i+1, j+1]
            if m >= q and m >= r:
                out[i, j] = m
    return out


def _hysteresis(thin: np.ndarray, low: float, high: float) -> np.ndarray:
    strong = (thin >= high).astype(np.uint8) * 255
    weak   = ((thin >= low) & (thin < high)).astype(np.uint8) * 75

    out = strong.copy()
    H, W = out.shape
    # BFS from strong pixels — keep weak neighbours connected to strong
    from collections import deque
    q = deque(zip(*np.where(strong > 0)))
    visited = strong > 0
    while q:
        i, j = q.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj] and weak[ni, nj]:
                    visited[ni, nj] = True
                    out[ni, nj] = 255
                    q.append((ni, nj))
    return out


def canny(image: np.ndarray, sigma: float = 1.0,
          low_thresh: float = 0.05, high_thresh: float = 0.15) -> np.ndarray:
    """
    Canny edge detector.

    Args:
        image:        Grayscale or RGB uint8/float array.
        sigma:        Gaussian blur σ before gradient computation.
        low_thresh:   Hysteresis low threshold  (fraction of max gradient).
        high_thresh:  Hysteresis high threshold (fraction of max gradient).

    Returns:
        Binary edge map (uint8, 0 or 255), same spatial size as input.
    """
    if image.ndim == 3:
        gray = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    else:
        gray = image.astype(float)

    blurred = gaussian_blur(gray, sigma)
    mag, ang = _gradients(blurred)
    thin = _nms(mag, ang)

    mx = thin.max() if thin.max() > 0 else 1
    return _hysteresis(thin, low_thresh * mx, high_thresh * mx)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    img[20:44, 20:44] = 200
    edges = canny(img, sigma=1.0)
    pct = 100 * (edges > 0).sum() / edges.size
    print(f"Canny edge pixels: {pct:.1f}%")
