# Gaussian Blur

The foundational smoothing filter used in almost every CV pipeline.

---

## The Gaussian Function

In 2D, centered at the origin:

```
G(x, y) = exp(-(x² + y²) / 2σ²)
```

**σ (sigma)** controls how fast weights fall off with distance from centre.
The normalization constant (1/2πσ²) is dropped because you renormalize
the discrete kernel anyway.

---

## Building the Discrete Kernel

### 1 — Choose size and σ

They are independent. Rule of thumb:

```
size = 2 · ceil(3σ) + 1
```

Covers ±3σ (99.7% of distribution mass).

### 2 — Sample G(x,y) at integer coordinates

For a 3×3 kernel, x, y ∈ {-1, 0, 1}:

```
0.3679  0.6065  0.3679
0.6065  1.0000  0.6065
0.3679  0.6065  0.3679
```

### 3 — Normalize

Divide every entry by the sum so weights sum to 1. Ensures a flat image passes
through unchanged.

```
0.0751  0.1238  0.0751
0.1238  0.2042  0.1238
0.0751  0.1238  0.0751
```

### σ vs weights (3×3 grid)

| σ   | centre | corner | character |
|-----|--------|--------|-----------|
| 0.5 | 0.619  | 0.019  | nearly a point filter |
| 1.0 | 0.204  | 0.075  | natural blur |
| 2.0 | 0.120  | 0.102  | near-uniform (≈ box blur) |

---

## Convolution

Slide the kernel to every pixel position, multiply element-wise with the
overlapping patch, sum:

```
output[i,j] = Σ_{kr,kc} image[i+kr-half, j+kc-half] · kernel[kr,kc]
```

### Separability trick

G(x,y) = G(x)·G(y), so the 2D convolution decomposes into two 1D passes:

1. Convolve each **row** with the 1D Gaussian
2. Convolve each **column** of the result

Cost: O(k²) → O(2k) per pixel. For a 15×15 kernel: 225 ops → 30 ops.

---

## Why every CV pipeline uses it

| Use case | Reason |
|---|---|
| Pre-blur before edge detection | Suppresses noise that Sobel would amplify |
| Scale-space / SIFT | Repeated blurring builds a Gaussian pyramid |
| Anti-aliasing before downsampling | Removes aliasing frequencies |
| Stable gradient computation | Fused as Derivative of Gaussian |

---

## Visualization

![Convolution animation](../assets/convolution.gif)

---

## Code

```python
from src.kernel  import gaussian_kernel
from src.convolve import gaussian_blur

kernel  = gaussian_kernel(sigma=1.0, size=7)
blurred = gaussian_blur(image, sigma=1.0)
```

See [`src/kernel.py`](../src/kernel.py) and [`src/convolve.py`](../src/convolve.py).
