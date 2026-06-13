# Morphological Operations

Shape-based image processing using a **structuring element (SE)** — a small
binary mask that defines the neighbourhood shape.

---

## Core Operations

### Erosion

```
(A ⊖ B)[i,j] = min of A under B centred at (i,j)
```

- Shrinks bright regions
- Removes protrusions and small objects smaller than SE

### Dilation

```
(A ⊕ B)[i,j] = max of A under B centred at (i,j)
```

- Expands bright regions
- Fills gaps and holes smaller than SE

---

## Derived Operations

### Opening  =  Erosion → Dilation

```
A ∘ B = (A ⊖ B) ⊕ B
```

- Removes small bright blobs (smaller than SE)
- Preserves shape of larger structures

### Closing  =  Dilation → Erosion

```
A • B = (A ⊕ B) ⊖ B
```

- Fills small dark holes (smaller than SE)
- Preserves overall shape

### Morphological Gradient

```
(A ⊕ B) − (A ⊖ B)
```

Highlights object boundaries — similar to an edge detector.

### Top-Hat

```
A − (A ∘ B)
```

Extracts bright peaks smaller than the SE (useful for uneven illumination correction).

### Black-Hat

```
(A • B) − A
```

Extracts dark valleys smaller than the SE.

---

## Structuring Elements

| Shape | Use case |
|---|---|
| 3×3 cross | Default, fast |
| 3×3 square | Isotropic, fills diagonals |
| Disk radius r | Truly circular, rotation-invariant |
| Line at angle θ | Directional filtering |

---

## Visualization

![Morphology](../assets/morphology.gif)

---

## Code

```python
from src.morphology import erosion, dilation, opening, closing, morph_gradient

se = np.ones((5, 5), dtype=np.uint8)   # 5×5 square SE
eroded  = erosion(image, se)
dilated = dilation(image, se)
opened  = opening(image, se)
closed  = closing(image, se)
```

See [`src/morphology.py`](../src/morphology.py).
