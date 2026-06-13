# Shape from Shading

Recover the 3D surface shape from the brightness variation in a single 2D image,
under a known lighting model.

---

## Reflectance Model

Assumes **Lambertian** (perfectly diffuse) reflectance:

```
I(x, y) = ρ · max(0, n̂(x,y) · l̂)
```

| Symbol | Meaning |
|---|---|
| I(x,y) | Observed image intensity at pixel (x,y) |
| ρ | Albedo (surface reflectivity, assumed uniform) |
| n̂(x,y) | Unit surface normal at (x,y) |
| l̂ | Unit vector pointing toward the light source |

---

## Parameterisation

A surface z(x,y) has gradients:

```
p = ∂z/∂x,    q = ∂z/∂y
```

The outward normal is:

```
n̂ = (−p, −q, 1) / sqrt(p² + q² + 1)
```

Substituting into the reflectance equation gives the **image irradiance equation**:

```
I(x,y) = ρ · (−p·lx − q·ly + lz) / sqrt(p² + q² + 1)
```

One equation, two unknowns (p, q) per pixel — the problem is **underdetermined**.

---

## Horn's Iterative Method (1970)

Resolves ambiguity by imposing an **integrability constraint**: p and q must
correspond to a consistent surface (∂p/∂y = ∂q/∂x).

Minimise:

```
E = Σ(I − Î(p,q))² + λ·(p_y − q_x)²
```

Gradient descent with local averaging of p, q provides the update:

```
p ← p̄ − μ · ∂E/∂p
q ← q̄ − μ · ∂E/∂q
```

where p̄, q̄ are 4-connected neighbourhood means.

---

## Depth recovery

Integrate p = ∂z/∂x and q = ∂z/∂y to get depth z(x,y), up to an additive constant:

```
z(x, y) ≈ cumsum(p, axis=x) + cumsum(q, axis=y)
```

---

## Ambiguities and extensions

| Ambiguity | Solution |
|---|---|
| Unknown albedo ρ | Segment by colour; normalise |
| Unknown lighting l̂ | Photometric stereo (≥3 images, different lights) |
| Non-Lambertian surfaces | Add specular term (Phong model) or use deep SfS |
| Global concave/convex | User interaction or shape prior |

---

## Visualization

![Shape from shading](../assets/shape_from_shading.gif)

---

## Code

```python
from src.shape_from_shading import shape_from_shading, make_sphere_image

img    = make_sphere_image(64, 64, light_dir=(0.3, 0.3, 1.0))
result = shape_from_shading(img, light_dir=(0.3, 0.3, 1.0), n_iter=200)

normals = result['normals']   # (H, W, 3)
depth   = result['depth']     # (H, W)
```

See [`src/shape_from_shading.py`](../src/shape_from_shading.py).
