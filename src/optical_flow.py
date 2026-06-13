"""
optical_flow.py — dense optical flow via Lucas-Kanade (local) and
                  Horn-Schunck (global).

Both estimate (u, v) displacement at each pixel between two frames.

Lucas-Kanade (1981):
  Assumes constant flow within a small window W.
  Minimises: sum_W (Ix·u + Iy·v + It)²
  Closed-form solution via 2×2 linear system per window.

Horn-Schunck (1981):
  Global smoothness constraint + brightness constancy.
  Minimises: sum[ (Ix·u + Iy·v + It)² + α²(|∇u|² + |∇v|²) ]
  Solved via iterative averaging update.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from convolve import convolve2d_naive

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float) / 8
BOX3    = np.ones((3, 3), dtype=float) / 9


def _gradients(f1: np.ndarray, f2: np.ndarray):
    """Spatial and temporal image gradients."""
    ix = convolve2d_naive(f1, SOBEL_X)
    iy = convolve2d_naive(f1, SOBEL_Y)
    it = (f2 - f1).astype(float)
    return ix, iy, it


def lucas_kanade(frame1: np.ndarray, frame2: np.ndarray,
                 window_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Lucas-Kanade optical flow.

    For each pixel, accumulates the 2×2 structure tensor A and
    vector b over a local window, then solves A·[u,v]ᵀ = b.

    Args:
        frame1, frame2: Consecutive grayscale frames (H, W float).
        window_size:    Side of the square integration window (odd).

    Returns:
        u, v: Horizontal and vertical flow fields (H, W).
        Unreliable pixels (small |det A|) are set to 0.
    """
    f1 = frame1.astype(float)
    f2 = frame2.astype(float)
    ix, iy, it = _gradients(f1, f2)

    half = window_size // 2
    H, W = f1.shape
    u = np.zeros((H, W))
    v = np.zeros((H, W))

    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy
    ixt = ix * it
    iyt = iy * it

    for i in range(half, H - half):
        for j in range(half, W - half):
            r0, r1 = i - half, i + half + 1
            c0, c1 = j - half, j + half + 1
            A = np.array([[ix2[r0:r1, c0:c1].sum(), ixy[r0:r1, c0:c1].sum()],
                          [ixy[r0:r1, c0:c1].sum(), iy2[r0:r1, c0:c1].sum()]])
            b = -np.array([ixt[r0:r1, c0:c1].sum(),
                           iyt[r0:r1, c0:c1].sum()])
            det = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
            if abs(det) > 1e-6:
                flow = np.linalg.solve(A, b)
                u[i, j], v[i, j] = flow
    return u, v


def horn_schunck(frame1: np.ndarray, frame2: np.ndarray,
                 alpha: float = 1.0,
                 n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Horn-Schunck optical flow.

    Iterative update:
        u ← ū - Ix · (Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)
        v ← v̄ - Iy · (Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)

    where ū, v̄ are local averages (3×3 box filter).

    Args:
        frame1, frame2: Consecutive grayscale frames (H, W float).
        alpha:          Smoothness weight (larger = smoother flow).
        n_iter:         Number of iterations.

    Returns:
        u, v: Flow fields (H, W).
    """
    f1 = frame1.astype(float)
    f2 = frame2.astype(float)
    ix, iy, it = _gradients(f1, f2)

    denom = alpha**2 + ix**2 + iy**2
    u = np.zeros_like(f1)
    v = np.zeros_like(f1)

    for _ in range(n_iter):
        u_avg = convolve2d_naive(u, BOX3)
        v_avg = convolve2d_naive(v, BOX3)
        p = (ix * u_avg + iy * v_avg + it) / denom
        u = u_avg - ix * p
        v = v_avg - iy * p

    return u, v


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    frame1 = rng.integers(0, 256, (32, 32)).astype(float)
    # Shift right by 2 pixels to create known flow
    frame2 = np.roll(frame1, 2, axis=1)

    u_lk, v_lk = lucas_kanade(frame1, frame2, window_size=5)
    u_hs, v_hs = horn_schunck(frame1, frame2, alpha=1.0, n_iter=50)

    print(f"LK  — mean u={u_lk.mean():.2f}  mean v={v_lk.mean():.2f}")
    print(f"HS  — mean u={u_hs.mean():.2f}  mean v={v_hs.mean():.2f}")
