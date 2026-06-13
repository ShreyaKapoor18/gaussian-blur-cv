# Gaussian Blur in Computer Vision

A ground-up explanation of Gaussian blur — the math, the kernel construction, convolution mechanics, and why every CV pipeline uses it.

---

## 1. The Gaussian function

In 2D, the Gaussian centered at the origin is:

```
G(x, y) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)
```

The normalization constant is dropped when building a discrete kernel since you renormalize anyway:

```
G(x, y) = exp(-(x² + y²) / 2σ²)
```

**σ (sigma)** controls the shape of the bell curve — how quickly weights fall off with distance from centre. It does **not** determine the kernel size.

---

## 2. Building the discrete kernel

### Step 1 — choose kernel size and σ

Kernel size and σ are independent. The rule of thumb is:

```
kernel_size = 2 * ceil(3σ) + 1
```

This covers ±3σ, capturing 99.7% of the distribution. But for pedagogical or performance reasons you can use a smaller kernel (e.g. 3×3 with σ=1).

### Step 2 — sample G(x, y) at integer coordinates

For a 3×3 kernel, coordinates run over x, y ∈ {-1, 0, 1}:

```
position     x    y    x²+y²    raw value
top-left    -1   -1      2      exp(-1)  = 0.3679
top-centre   0   -1      1      exp(-0.5)= 0.6065
centre       0    0      0      exp(0)   = 1.0000
...
```

Full 3×3 raw matrix (σ=1):

```
0.3679  0.6065  0.3679
0.6065  1.0000  0.6065
0.3679  0.6065  0.3679
```

### Step 3 — normalize

Sum all values: 4×0.3679 + 4×0.6065 + 1.0 = **4.8976**

Divide every entry by the sum so weights sum to 1:

```
0.0751  0.1238  0.0751
0.1238  0.2042  0.1238
0.0751  0.1238  0.0751
```

This ensures a flat image passes through unchanged (no brightening/darkening).

### How σ affects weights within a fixed 3×3 grid

| σ    | centre weight | corner weight | character          |
|------|--------------|---------------|--------------------|
| 0.5  | 0.619        | 0.019         | centre-heavy, mild blur |
| 1.0  | 0.204        | 0.075         | moderate, natural  |
| 2.0  | 0.120        | 0.102         | near-uniform       |
| ∞    | 0.111        | 0.111         | box blur (1/9 each)|

Large σ → weights flatten → approaches a uniform box blur.

---

## 3. Convolution

The kernel is **convolved** over the image: slide it to every valid position, multiply each kernel weight against the overlapping pixel, sum the products.

For a 5×5 image with a 3×3 kernel (no padding), valid positions = (5-3+1)² = **9** output pixels.

```
output[oy, ox] = Σ_{kr=0}^{2} Σ_{kc=0}^{2}  image[oy+kr, ox+kc] · kernel[kr, kc]
```

### The separability trick

Because G(x,y) = G(x)·G(y), the 2D convolution decomposes into two 1D passes:

1. Convolve each **row** with the 1D Gaussian
2. Convolve each **column** of the result

Cost drops from O(k²) to O(2k) per pixel. A 15×15 kernel: 225 ops → 30 ops.

---

## 4. Why CV pipelines use Gaussian blur

| Use case | Why Gaussian |
|---|---|
| Noise suppression before edge detection | Attenuates high-frequency noise; Sobel/Prewitt gradients on raw images pick up noise as false edges |
| Scale-space construction | Repeated blurring at increasing σ builds a Gaussian pyramid; SIFT uses Difference of Gaussians (DoG) to find scale-invariant keypoints |
| Anti-aliasing before downsampling | Removes frequencies above the new Nyquist limit; prevents Moiré artefacts in image pyramids |
| Stable gradient computation | Differentiation amplifies noise; Gaussian smoothing first (or fused as Derivative of Gaussian) gives noise-robust ∂I/∂x and ∂I/∂y |

---

## 5. The tradeoff

Gaussian blur trades **detail for stability**:

- Too little blur → noisy gradients, false edges
- Too much blur → thin structures disappear, edge localisation degrades

Canny edge detection formalises this by treating σ as an explicit parameter, automatically selecting the right level of smoothing.

---

## 6. Code

See [`src/`](src/) for Python implementations:

- `kernel.py` — build a Gaussian kernel for any σ and size
- `convolve.py` — naive 2D convolution and separable fast version
- `filters.py` — grayscale, threshold, blur, edge detection (Sobel, Prewitt, Laplacian)
- `demo.py` — end-to-end demo on a test image

```bash
pip install numpy pillow matplotlib
python src/demo.py
```
