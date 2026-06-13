"""
animate.py — generate a GIF showing a convolution kernel sliding over an image.

Produces assets/convolution.gif
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from kernel import gaussian_kernel

# ── Setup ─────────────────────────────────────────────────────────────────────

# Small image so each step is visible
IMG_H, IMG_W = 8, 10
rng = np.random.default_rng(7)
image = rng.integers(30, 220, (IMG_H, IMG_W)).astype(float)
# Put a bright square so there's obvious structure
image[2:6, 3:7] = 180

SIGMA = 1.0
kernel = gaussian_kernel(SIGMA, size=3)   # 3×3
K = 3
HALF = K // 2

# Zero-pad the image
padded = np.pad(image, HALF, mode='constant', constant_values=0)

# Pre-compute all output values
output = np.zeros_like(image)
positions = []
for i in range(IMG_H):
    for j in range(IMG_W):
        region = padded[i:i+K, j:j+K]
        output[i, j] = np.sum(region * kernel)
        positions.append((i, j))

# ── Figure layout ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")

ax_img, ax_kern, ax_out = axes

norm_img = Normalize(vmin=0, vmax=255)
norm_kern = Normalize(vmin=0, vmax=kernel.max())

# Left: input image
im_input = ax_img.imshow(image, cmap="gray", norm=norm_img, interpolation="nearest")
ax_img.set_title("Input image", color="white", fontsize=11, pad=8)
ax_img.tick_params(colors="gray", labelsize=7)
for spine in ax_img.spines.values():
    spine.set_edgecolor("#444")

# Highlight rect on input
rect_input = patches.Rectangle(
    (-HALF - 0.5, -HALF - 0.5), K, K,
    linewidth=2, edgecolor="#e94560", facecolor="none"
)
ax_img.add_patch(rect_input)

# Middle: kernel weights (static)
ax_kern.imshow(kernel, cmap="hot", norm=norm_kern, interpolation="nearest")
ax_kern.set_title(f"3×3 Gaussian kernel  σ={SIGMA}", color="white", fontsize=11, pad=8)
ax_kern.tick_params(colors="gray", labelsize=7)
ax_kern.set_xticks(range(K))
ax_kern.set_yticks(range(K))
ax_kern.set_xticklabels(range(-HALF, HALF+1), color="gray", fontsize=7)
ax_kern.set_yticklabels(range(-HALF, HALF+1), color="gray", fontsize=7)
for spine in ax_kern.spines.values():
    spine.set_edgecolor("#444")
for r in range(K):
    for c in range(K):
        ax_kern.text(c, r, f"{kernel[r,c]:.3f}", ha="center", va="center",
                     color="white", fontsize=7.5, fontweight="bold")

# Right: output being built
out_display = np.full_like(image, np.nan)
im_out = ax_out.imshow(out_display, cmap="inferno",
                        vmin=0, vmax=255, interpolation="nearest")
ax_out.set_title("Output (convolved)", color="white", fontsize=11, pad=8)
ax_out.tick_params(colors="gray", labelsize=7)
rect_out = patches.Rectangle(
    (-0.5, -0.5), 1, 1,
    linewidth=2, edgecolor="#0f3460", facecolor="none"
)
ax_out.add_patch(rect_out)
for spine in ax_out.spines.values():
    spine.set_edgecolor("#444")

# Step counter text
step_text = fig.text(0.5, 0.01, "", ha="center", color="#aaa", fontsize=9)

plt.tight_layout(pad=1.8)

# ── Animation ─────────────────────────────────────────────────────────────────

def init():
    out_display[:] = np.nan
    im_out.set_data(out_display.copy())
    rect_input.set_xy((-HALF - 0.5, -HALF - 0.5))
    rect_out.set_xy((-0.5, -0.5))
    step_text.set_text("")
    return im_input, im_out, rect_input, rect_out, step_text


def update(frame):
    i, j = positions[frame]

    # Move highlight on input
    rect_input.set_xy((j - HALF - 0.5, i - HALF - 0.5))

    # Move highlight on output
    rect_out.set_xy((j - 0.5, i - 0.5))

    # Reveal output pixel
    out_display[i, j] = output[i, j]
    im_out.set_data(out_display.copy())

    step_text.set_text(f"step {frame+1}/{len(positions)}  →  pixel ({i},{j})  =  {output[i,j]:.1f}")
    return im_input, im_out, rect_input, rect_out, step_text


n_frames = len(positions)
ani = FuncAnimation(
    fig, update, frames=n_frames,
    init_func=init, blit=False, interval=120
)

out_path = os.path.join(os.path.dirname(__file__), "..", "assets", "convolution.gif")
out_path = os.path.normpath(out_path)
ani.save(out_path, writer=PillowWriter(fps=10))
print(f"Saved → {out_path}")
