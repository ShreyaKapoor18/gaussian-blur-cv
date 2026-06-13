# Stereo Vision and Disparity

Recover depth from two cameras (a stereo pair) by finding corresponding pixels
and measuring their horizontal shift (disparity).

---

## Geometry

With **rectified** stereo images (epipolar lines are horizontal), a point P at
depth Z projects to column xL in the left image and xR in the right image.

```
disparity d = xL − xR = f · B / Z
```

| Symbol | Meaning |
|---|---|
| d | Disparity (pixels) |
| f | Focal length (pixels) |
| B | Baseline — distance between cameras |
| Z | Depth (same units as B) |

Depth from disparity:

```
Z = f · B / d
```

Large disparity → close object. Disparity = 0 → point at infinity.

---

## Rectification

Before matching, images must be rectified so corresponding points lie on the same
row. This involves:
1. Estimate the fundamental matrix F from point correspondences
2. Warp both images so epipolar lines become horizontal

---

## Block Matching

For each pixel (i, j) in the left image:

1. Extract a block_size × block_size patch around it.
2. Search the same row in the right image over d ∈ [0, max_disp].
3. Find d that minimises the **SAD** (Sum of Absolute Differences):

```
d*(i,j) = argmin_d  Σ |I_L(i+r, j+c) − I_R(i+r, j+c−d)|
```

### Cost functions

| Cost | Formula | Sensitivity |
|---|---|---|
| SAD | Σ |ΔI| | Illumination-sensitive |
| SSD | Σ ΔI² | Outlier-sensitive |
| NCC | correlation / (σL·σR) | Robust to linear illumination change |
| Census | Hamming distance of bit-string | Very robust |

---

## Disparity Post-processing

Raw block-matching outputs are noisy. Common fixes:
- **Left-right consistency check**: compute disparity both ways; flag where they disagree
- **Sub-pixel refinement**: fit a parabola around the cost minimum
- **Median filter**: removes isolated outliers

---

## Visualization

![Stereo disparity](../assets/stereo.gif)

---

## Code

```python
from src.stereo import disparity_map, depth_from_disparity

disp  = disparity_map(left, right, block_size=7, max_disp=32)
depth = depth_from_disparity(disp, focal_length=500, baseline=0.1)
```

See [`src/stereo.py`](../src/stereo.py).
