"""
demo.py — end-to-end demo: build kernel, blur, detect edges, equalize histogram.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from kernel import gaussian_kernel, print_kernel
from convolve import gaussian_blur
from filters import to_grayscale, sobel, prewitt, laplacian_edges, histogram_equalize

import math

def make_test_image(H=64, W=64):
    img = np.zeros((H, W), dtype=float)
    for r in range(H):
        for c in range(W):
            img[r, c] = 40 + (c / W) * 80
    img[20:44, 20:44] = 220
    img[28:36, 28:36] = 80
    rng = np.random.default_rng(0)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

if __name__ == "__main__":
    section("1. Kernel construction")
    for sigma in [0.5, 1.0, 2.0]:
        size = 2 * math.ceil(3 * sigma) + 1
        k = gaussian_kernel(sigma, size)
        print(f"\n  σ={sigma}, size={size}×{size}")
        print_kernel(k, decimals=4)
        print(f"  centre={k[size//2,size//2]:.4f}  corner={k[0,0]:.4f}  sum={k.sum():.6f}")

    section("2. Gaussian blur")
    img = make_test_image()
    print(f"  Input  — mean={img.mean():.1f}  std={img.std():.1f}  max={img.max():.0f}")
    for sigma in [0.5, 1.0, 2.0, 4.0]:
        blurred = gaussian_blur(img, sigma)
        print(f"  σ={sigma:<4} — mean={blurred.mean():.1f}  std={blurred.std():.1f}  max={blurred.max():.0f}")

    section("3. Edge detection")
    for name, fn in [("Sobel", sobel), ("Prewitt", prewitt), ("Laplacian", laplacian_edges)]:
        edges = fn(img, pre_blur_sigma=1.0)
        pct = 100 * (edges > 0).sum() / edges.size
        print(f"  {name:<12} edge pixels: {pct:.1f}%")

    section("4. Histogram equalization")
    eq = histogram_equalize(img)
    print(f"  Before — mean={img.mean():.1f}  std={img.std():.1f}")
    print(f"  After  — mean={eq.mean():.1f}  std={eq.std():.1f}")

    section("5. Separable vs naive agreement")
    from convolve import convolve2d_naive, convolve2d_separable
    k2d = gaussian_kernel(1.0, 7)
    k1d = k2d[3]; k1d = k1d / k1d.sum()
    out_naive = convolve2d_naive(img, k2d)
    out_sep   = convolve2d_separable(img, k1d)
    diff = np.abs(out_naive - out_sep).max()
    print(f"  Max pixel difference: {diff:.2e}  {'OK' if diff < 1e-6 else 'MISMATCH'}")

    print("\nDone.\n")
