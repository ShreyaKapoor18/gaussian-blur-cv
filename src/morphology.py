"""
morphology.py — binary and grayscale morphological operations.

Core operations:
  erosion:  output[i,j] = min of image under structuring element (SE)
  dilation: output[i,j] = max of image under structuring element (SE)

Derived:
  opening  = erosion  then dilation  (removes small bright blobs)
  closing  = dilation then erosion   (fills small dark holes)
  gradient = dilation - erosion      (highlights edges)
  top-hat  = image - opening         (isolates bright peaks)
  black-hat= closing - image         (isolates dark valleys)
"""

import numpy as np


def _apply(image: np.ndarray, se: np.ndarray, func) -> np.ndarray:
    H, W = image.shape
    kH, kW = se.shape
    pH, pW = kH // 2, kW // 2
    padded = np.pad(image, ((pH, pH), (pW, pW)), mode='edge')
    out = np.zeros_like(image, dtype=float)
    rows, cols = np.where(se > 0)
    patches = np.stack([padded[r:r+H, c:c+W] for r, c in zip(rows, cols)], axis=0)
    return func(patches, axis=0)


def erosion(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """
    Morphological erosion: each output pixel = min of neighbours under SE.

    Shrinks bright regions; removes thin protrusions.

    Args:
        image: 2D float or uint8 array.
        se:    Structuring element (2D bool/int array). Default: 3×3 cross.
    """
    if se is None:
        se = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    return _apply(image.astype(float), se, np.min)


def dilation(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """
    Morphological dilation: each output pixel = max of neighbours under SE.

    Expands bright regions; fills thin gaps.
    """
    if se is None:
        se = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    return _apply(image.astype(float), se, np.max)


def opening(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Erosion followed by dilation — removes small bright objects."""
    return dilation(erosion(image, se), se)


def closing(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Dilation followed by erosion — closes small dark holes."""
    return erosion(dilation(image, se), se)


def morph_gradient(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Dilation minus erosion — highlights object boundaries."""
    return dilation(image, se) - erosion(image, se)


def top_hat(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Image minus opening — extracts bright peaks smaller than SE."""
    return image.astype(float) - opening(image, se)


def black_hat(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Closing minus image — extracts dark valleys smaller than SE."""
    return closing(image, se) - image.astype(float)


if __name__ == "__main__":
    img = np.zeros((16, 16), dtype=float)
    img[4:12, 4:12] = 255
    img[7, 7] = 0       # small hole
    img[1, 1] = 255     # isolated dot

    se = np.ones((3, 3), dtype=np.uint8)

    print(f"Original  — non-zero pixels: {(img > 0).sum()}")
    print(f"Eroded    — non-zero pixels: {(erosion(img, se) > 0).sum()}")
    print(f"Dilated   — non-zero pixels: {(dilation(img, se) > 0).sum()}")
    print(f"Opened    — non-zero pixels: {(opening(img, se) > 0).sum()}")
    print(f"Closed    — non-zero pixels: {(closing(img, se) > 0).sum()}")
