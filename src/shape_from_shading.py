"""
shape_from_shading.py — recover surface normals from a single shaded image.

Assumes:
  - Lambertian (diffuse) surface: I(x,y) = ρ · max(0, n̂ · l̂)
  - Orthographic projection
  - Single distant point light source l̂ (unit vector)
  - Uniform albedo ρ (defaults to 1.0)

Surface normal n̂ is parameterised by gradients p = ∂z/∂x, q = ∂z/∂y:
    n̂ = (-p, -q, 1) / sqrt(p² + q² + 1)

Horn's iterative method (1970):
    Update p, q by minimising photometric error + integrability constraint.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from convolve import convolve2d_naive


def _lambertian_intensity(p: np.ndarray, q: np.ndarray,
                           light: np.ndarray) -> np.ndarray:
    lx, ly, lz = light
    denom = np.sqrt(p**2 + q**2 + 1)
    return np.clip((-p*lx - q*ly + lz) / denom, 0, 1)


def shape_from_shading(image: np.ndarray,
                        light_dir: tuple = (0.0, 0.0, 1.0),
                        albedo: float = 1.0,
                        n_iter: int = 200,
                        mu: float = 1.0) -> dict:
    """
    Recover surface gradients (p, q) and normals from a shaded image.

    Args:
        image:     2D float array, intensities in [0, 1].
        light_dir: Unit vector (lx, ly, lz) pointing toward the light.
        albedo:    Surface albedo ρ (assumed uniform).
        n_iter:    Number of Horn iterations.
        mu:        Step size (learning rate).

    Returns:
        dict with keys:
          'p':       ∂z/∂x  (H, W)
          'q':       ∂z/∂y  (H, W)
          'normals': Unit surface normals (H, W, 3)
          'depth':   Integrated depth map (H, W), up to an additive constant
          'reconstructed': Re-rendered Lambertian image (H, W)
    """
    light = np.array(light_dir, dtype=float)
    light /= np.linalg.norm(light)
    lx, ly, lz = light

    I = image.astype(float) / albedo
    I = np.clip(I, 0, 1)

    H, W = I.shape
    p = np.zeros((H, W))
    q = np.zeros((H, W))

    # Averaging kernel (4-connected neighbourhood mean)
    avg_k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) / 4

    for _ in range(n_iter):
        p_avg = convolve2d_naive(p, avg_k)
        q_avg = convolve2d_naive(q, avg_k)

        denom = np.sqrt(p_avg**2 + q_avg**2 + 1)
        I_hat = np.clip((-p_avg*lx - q_avg*ly + lz) / denom, 0, 1)
        err = I_hat - I

        # Gradient of photometric error w.r.t. p and q
        d_denom = denom**3
        dp = mu * err * (lx * denom**2 - (-p_avg*lx - q_avg*ly + lz) * p_avg) / d_denom
        dq = mu * err * (ly * denom**2 - (-p_avg*lx - q_avg*ly + lz) * q_avg) / d_denom

        p = p_avg - dp
        q = q_avg - dq

    # Build unit normals
    denom = np.sqrt(p**2 + q**2 + 1)
    nx = -p / denom
    ny = -q / denom
    nz =  1 / denom
    normals = np.stack([nx, ny, nz], axis=-1)

    # Integrate depth by summing gradients along rows then columns
    depth = np.cumsum(p, axis=1) + np.cumsum(q, axis=0)

    reconstructed = _lambertian_intensity(p, q, light) * albedo

    return {
        'p': p, 'q': q,
        'normals': normals,
        'depth': depth,
        'reconstructed': reconstructed,
    }


def make_sphere_image(H: int = 64, W: int = 64,
                      light_dir: tuple = (0.3, 0.3, 1.0)) -> np.ndarray:
    """Render a synthetic Lambertian sphere for testing."""
    cx, cy, r = W / 2, H / 2, min(H, W) * 0.4
    light = np.array(light_dir, dtype=float)
    light /= np.linalg.norm(light)

    image = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            dx, dy = (j - cx) / r, (i - cy) / r
            r2 = dx**2 + dy**2
            if r2 <= 1:
                dz = np.sqrt(max(0, 1 - r2))
                normal = np.array([dx, dy, dz])
                normal /= np.linalg.norm(normal)
                image[i, j] = max(0, np.dot(normal, light))
    return image


if __name__ == "__main__":
    img = make_sphere_image(64, 64, light_dir=(0.3, 0.3, 1.0))
    result = shape_from_shading(img, light_dir=(0.3, 0.3, 1.0), n_iter=100)
    err = np.abs(result['reconstructed'] - img).mean()
    print(f"Mean photometric error after 100 iters: {err:.4f}")
    print(f"Normal z-channel mean (should be ~1 at centre): "
          f"{result['normals'][32,32,2]:.3f}")
