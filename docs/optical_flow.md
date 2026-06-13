# Optical Flow

Estimate the apparent motion of pixels between two consecutive frames.

---

## Brightness Constancy Assumption

Both methods assume a pixel's intensity doesn't change as it moves:

```
I(x, y, t) = I(x+u, y+v, t+1)
```

Taylor-expanding and dropping higher-order terms:

```
Ix·u + Iy·v + It = 0
```

This is the **optical flow constraint equation** — one equation, two unknowns (u, v).
The *aperture problem*: along a single edge you can only recover flow perpendicular to it.
Both LK and HS resolve this with an additional constraint.

---

## Lucas-Kanade (Local)

**Additional assumption**: flow is constant inside a small window W.

Stack the constraint equation for all pixels in W:

```
A · [u, v]ᵀ = b

A = [[ΣIx²,  ΣIxIy],    b = [−ΣIxIt]
     [ΣIxIy, ΣIy²  ]]       [−ΣIyIt]
```

Solve the 2×2 system with least squares.  
The solution is reliable only when A is well-conditioned — i.e., the window
contains two independent gradient directions (a **corner**, not an edge).

---

## Horn-Schunck (Global)

**Additional assumption**: flow field (u, v) is spatially smooth everywhere.

Minimise the global energy:

```
E = Σ (Ix·u + Iy·v + It)² + α²(|∇u|² + |∇v|²)
```

Euler-Lagrange equations give iterative updates using local averages ū, v̄:

```
u ← ū − Ix·(Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)
v ← v̄ − Iy·(Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)
```

Larger α → smoother flow; better for rigid scenes. Smaller α → allows discontinuities.

---

## Comparison

| Property | Lucas-Kanade | Horn-Schunck |
|---|---|---|
| Scope | Local (per window) | Global (whole image) |
| Flow at edges | Aperture problem | Propagated from unambiguous regions |
| Handles large motions | Poorly (need pyramids) | Poorly |
| Computational cost | O(H·W·W²) | O(H·W·iter) |

Both are classical methods. Modern approaches (FlowNet, RAFT) use CNNs but are built on the same brightness-constancy intuition.

---

## Visualization

![Optical flow](../assets/optical_flow.gif)

---

## Code

```python
from src.optical_flow import lucas_kanade, horn_schunck

u, v = lucas_kanade(frame1, frame2, window_size=5)
u, v = horn_schunck(frame1, frame2, alpha=1.0, n_iter=100)
```

See [`src/optical_flow.py`](../src/optical_flow.py).
