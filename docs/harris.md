# Harris Corner Detection

Detects corners (two-direction intensity change) and edges (one-direction change)
by analysing the local structure tensor.

---

## Intuition

Slide a small window over the image and measure how much intensity changes in
each direction.

| Shift response | Region type |
|---|---|
| Small in all directions | Flat |
| Large in one direction only | Edge |
| Large in all directions | **Corner** |

---

## Algorithm

### 1. Compute gradients

```
Ix = image ∗ Sobel_x
Iy = image ∗ Sobel_y
```

### 2. Structure tensor M (per pixel)

Sum second-order products over a Gaussian-weighted window W:

```
M = [[Σ Ix²,  Σ IxIy],
     [Σ IxIy, Σ Iy² ]]
```

Eigenvalues λ₁, λ₂ of M tell us the curvature of the auto-correlation surface.
Both large → corner. One large → edge. Both small → flat.

### 3. Corner response R

To avoid computing eigenvalues directly, Harris uses:

```
R = det(M) - k · trace(M)²
  = λ₁λ₂ - k(λ₁+λ₂)²
```

- R >> 0 → corner
- R << 0 → edge (one dominant eigenvalue)
- |R| ≈ 0 → flat

Typical k = 0.04 – 0.06.

### 4. Threshold + non-maximum suppression

Retain pixels where R > threshold and R is a local maximum within a
minimum-distance neighbourhood.

---

## Properties

- **Rotation-invariant**: M is symmetric; eigenvalues don't change with rotation.
- **Not scale-invariant**: eigenvalues scale with window size and gradient magnitude.
  (Scale invariance → SIFT/SURF which uses Harris at multiple scales.)

---

## Visualization

![Harris corners](../assets/harris.gif)

---

## Code

```python
from src.harris import harris_corners
R, corners = harris_corners(image, sigma=1.5, k=0.04, threshold=1e6)
```

See [`src/harris.py`](../src/harris.py).
