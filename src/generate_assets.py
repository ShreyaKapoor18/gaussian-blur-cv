"""
generate_assets.py — generate all GIF visualizations for the docs.

Run from the repo root:
    python src/generate_assets.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

FPS = 10


def save(ani, name):
    path = os.path.join(ASSETS, name)
    ani.save(path, writer=PillowWriter(fps=FPS))
    print(f"  saved {path}")


# ── Canny ─────────────────────────────────────────────────────────────────────

def make_canny_gif():
    from canny import _gradients, _nms, _hysteresis
    from convolve import gaussian_blur

    rng = np.random.default_rng(3)
    img = rng.integers(20, 180, (48, 64)).astype(float)
    img[10:38, 15:49] = 200
    img[18:30, 22:42] = 80

    stages = []
    labels = []

    stages.append(img / 255)
    labels.append("Input")

    blurred = gaussian_blur(img, sigma=1.0)
    stages.append(blurred / 255)
    labels.append("1. Gaussian blur (σ=1)")

    mag, ang = _gradients(blurred)
    stages.append(mag / mag.max())
    labels.append("2. Gradient magnitude")

    thin = _nms(mag, ang)
    stages.append(thin / (thin.max() + 1e-9))
    labels.append("3. Non-maximum suppression")

    mx = thin.max()
    edges = _hysteresis(thin, 0.05 * mx, 0.15 * mx)
    stages.append(edges / 255)
    labels.append("4. Hysteresis → edges")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.axis("off")
    im = ax.imshow(stages[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    title = ax.set_title(labels[0], color="white", fontsize=12, pad=8)
    plt.tight_layout()

    def update(f):
        im.set_data(stages[f])
        title.set_text(labels[f])
        return im, title

    ani = FuncAnimation(fig, update, frames=len(stages) * 10,
                        init_func=lambda: (im, title), blit=False, interval=600)
    # Repeat each stage for a few frames
    def update2(f):
        idx = min(f // 10, len(stages) - 1)
        im.set_data(stages[idx])
        title.set_text(labels[idx])
        return im, title

    ani2 = FuncAnimation(fig, update2, frames=len(stages) * 10,
                         blit=False, interval=150)
    save(ani2, "canny.gif")
    plt.close()


# ── Harris ────────────────────────────────────────────────────────────────────

def make_harris_gif():
    from harris import harris_response

    img = np.zeros((64, 64), dtype=float)
    img[10:54, 10:54] = 180
    img[20:44, 20:44] = 0
    img[30:34, 30:34] = 255

    R = harris_response(img, sigma=1.5, k=0.04)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.axis("off")

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("Input", color="white", fontsize=11)

    im_r = axes[1].imshow(np.zeros_like(R), cmap="hot",
                           vmin=R.min(), vmax=R.max(), interpolation="nearest")
    axes[1].set_title("Corner response R", color="white", fontsize=11)

    overlay = np.stack([img/255]*3, axis=-1)
    im_o = axes[2].imshow(overlay, interpolation="nearest")
    axes[2].set_title("Detected corners", color="white", fontsize=11)
    dots, = axes[2].plot([], [], 'r+', markersize=10, markeredgewidth=2)

    plt.tight_layout(pad=1.5)

    n_frames = 30

    def update(f):
        frac = f / (n_frames - 1)
        thresh_frac = frac * R.max()
        partial = np.where(R > thresh_frac, R, 0)
        im_r.set_data(partial)

        corners = np.argwhere(R > max(thresh_frac, 0.7 * R.max()))
        if len(corners):
            dots.set_data(corners[:, 1], corners[:, 0])
        else:
            dots.set_data([], [])
        return im_r, dots

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=100)
    save(ani, "harris.gif")
    plt.close()


# ── Pyramids ──────────────────────────────────────────────────────────────────

def make_pyramids_gif():
    from pyramids import gaussian_pyramid, laplacian_pyramid

    rng = np.random.default_rng(7)
    img = rng.integers(30, 200, (128, 128)).astype(float)
    img[30:90, 30:90] = 200
    img[50:70, 50:70] = 80

    gpyr = gaussian_pyramid(img, levels=5)
    lpyr = laplacian_pyramid(img, levels=5)

    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    fig.patch.set_facecolor("#1a1a2e")
    plt.suptitle("Gaussian (top) and Laplacian (bottom) Pyramids",
                 color="white", fontsize=12)

    ims_g, ims_l = [], []
    for i in range(5):
        for row, pyr, ims in [(0, gpyr, ims_g), (1, lpyr, ims_l)]:
            ax = axes[row][i]
            ax.set_facecolor("#16213e")
            ax.axis("off")
            data = pyr[i]
            lp = lpyr[i]
            vmin = lp.min() if row == 1 else 0
            vmax = lp.max() if row == 1 else 255
            im = ax.imshow(np.zeros((4, 4)), cmap="gray" if row == 0 else "RdBu",
                           vmin=vmin, vmax=vmax, interpolation="nearest",
                           aspect="auto")
            ax.set_title(f"L{i}  {data.shape[0]}×{data.shape[1]}",
                         color="#aaa", fontsize=8)
            ims.append((im, pyr[i], vmin, vmax))

    plt.tight_layout(pad=1.5)

    def update(f):
        alpha = min(f / 15, 1.0)
        artists = []
        for im, data, vmin, vmax in ims_g + ims_l:
            show = data * alpha
            im.set_data(show)
            artists.append(im)
        return artists

    ani = FuncAnimation(fig, update, frames=25, blit=False, interval=80)
    save(ani, "pyramids.gif")
    plt.close()


# ── Optical flow ─────────────────────────────────────────────────────────────

def make_optical_flow_gif():
    rng = np.random.default_rng(42)
    H, W = 40, 56
    base = rng.integers(30, 200, (H, W)).astype(float)
    base[10:30, 15:40] = 200

    n_frames = 20
    shift_per_frame = 1   # shift right by 1 px per frame
    frames = []
    for t in range(n_frames):
        f = np.roll(base, t * shift_per_frame, axis=1)
        frames.append(f)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.axis("off")

    im1 = axes[0].imshow(frames[0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("Frame t", color="white", fontsize=11)

    im2 = axes[1].imshow(frames[1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("Frame t+1", color="white", fontsize=11)

    # Flow quiver placeholder
    step = 6
    ys = np.arange(step, H - step, step)
    xs = np.arange(step, W - step, step)
    XX, YY = np.meshgrid(xs, ys)
    U = np.zeros_like(XX, dtype=float)
    V = np.zeros_like(YY, dtype=float)
    q = axes[2].quiver(XX, YY, U, V, color="#e94560", scale=30, width=0.005)
    axes[2].imshow(frames[0], cmap="gray", vmin=0, vmax=255, interpolation="nearest", alpha=0.5)
    axes[2].set_xlim(0, W)
    axes[2].set_ylim(H, 0)
    axes[2].set_title("Lucas-Kanade flow", color="white", fontsize=11)

    plt.tight_layout(pad=1.5)

    from optical_flow import lucas_kanade

    def update(f):
        t = f % (n_frames - 1)
        im1.set_data(frames[t])
        im2.set_data(frames[t + 1])
        u, v = lucas_kanade(frames[t], frames[t + 1], window_size=5)
        U_q = u[ys[:, None], xs[None, :]]
        V_q = v[ys[:, None], xs[None, :]]
        q.set_UVC(U_q, V_q)
        return im1, im2, q

    ani = FuncAnimation(fig, update, frames=n_frames - 1, blit=False, interval=150)
    save(ani, "optical_flow.gif")
    plt.close()


# ── Morphology ────────────────────────────────────────────────────────────────

def make_morphology_gif():
    from morphology import erosion, dilation, opening, closing, morph_gradient

    img = np.zeros((48, 64), dtype=float)
    img[8:40, 8:56] = 255
    img[18:30, 22:42] = 0    # hole
    img[2, 2] = 255           # noise dot
    img[45, 60] = 255

    se = np.ones((5, 5), dtype=np.uint8)

    ops = [
        (img,                        "Original"),
        (erosion(img, se),           "Erosion"),
        (dilation(img, se),          "Dilation"),
        (opening(img, se),           "Opening (erosion→dilation)"),
        (closing(img, se),           "Closing (dilation→erosion)"),
        (morph_gradient(img, se),    "Morphological gradient"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.axis("off")
    im = ax.imshow(ops[0][0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    title = ax.set_title(ops[0][1], color="white", fontsize=12, pad=8)
    plt.tight_layout()

    def update(f):
        idx = f // 12
        data, label = ops[idx]
        im.set_data(data)
        title.set_text(label)
        return im, title

    ani = FuncAnimation(fig, update, frames=len(ops) * 12, blit=False, interval=120)
    save(ani, "morphology.gif")
    plt.close()


# ── Shape from shading ────────────────────────────────────────────────────────

def make_sfs_gif():
    from shape_from_shading import make_sphere_image, shape_from_shading

    img = make_sphere_image(64, 64, light_dir=(0.4, 0.2, 1.0))
    light = (0.4, 0.2, 1.0)

    checkpoints = [5, 10, 20, 50, 100, 200]
    results = []
    for n in checkpoints:
        r = shape_from_shading(img, light_dir=light, n_iter=n)
        results.append(r)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.axis("off")

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Input image", color="white", fontsize=11)

    im_depth = axes[1].imshow(np.zeros((64, 64)), cmap="viridis", interpolation="nearest")
    axes[1].set_title("Recovered depth", color="white", fontsize=11)

    im_norm = axes[2].imshow(np.zeros((64, 64, 3)), interpolation="nearest")
    axes[2].set_title("Surface normals", color="white", fontsize=11)

    im_recon = axes[3].imshow(np.zeros((64, 64)), cmap="gray", vmin=0, vmax=1,
                               interpolation="nearest")
    t_recon = axes[3].set_title("Reconstructed (iter 0)", color="white", fontsize=11)

    plt.tight_layout(pad=1.5)

    n_frames = len(checkpoints) * 8

    def update(f):
        idx = min(f // 8, len(results) - 1)
        r = results[idx]
        depth = r['depth']
        d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        im_depth.set_data(d_norm)

        normals = (r['normals'] + 1) / 2
        im_norm.set_data(np.clip(normals, 0, 1))

        im_recon.set_data(r['reconstructed'])
        t_recon.set_text(f"Reconstructed (iter {checkpoints[idx]})")
        return im_depth, im_norm, im_recon, t_recon

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=120)
    save(ani, "shape_from_shading.gif")
    plt.close()


# ── Stereo ────────────────────────────────────────────────────────────────────

def make_stereo_gif():
    from stereo import make_stereo_pair, disparity_map, depth_from_disparity

    left, right = make_stereo_pair(H=48, W=64, true_disp=6)
    disp = disparity_map(left, right, block_size=7, max_disp=16)
    depth = depth_from_disparity(disp, focal_length=500, baseline=0.1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.axis("off")

    im_l = axes[0].imshow(left, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("Left image", color="white", fontsize=11)

    im_r = axes[1].imshow(right, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("Right image", color="white", fontsize=11)

    im_d = axes[2].imshow(np.zeros_like(disp), cmap="plasma",
                           vmin=0, vmax=disp.max(), interpolation="nearest")
    axes[2].set_title("Disparity map", color="white", fontsize=11)

    # Draw epipolar line
    line_l, = axes[0].plot([0, left.shape[1]], [24, 24], 'r--', linewidth=1.2, alpha=0.8)
    line_r, = axes[1].plot([0, right.shape[1]], [24, 24], 'r--', linewidth=1.2, alpha=0.8)

    plt.tight_layout(pad=1.5)

    n_frames = 40

    def update(f):
        frac = f / (n_frames - 1)
        # Animate disparity filling in row by row
        rows_shown = int(frac * disp.shape[0])
        partial = np.zeros_like(disp)
        partial[:rows_shown] = disp[:rows_shown]
        im_d.set_data(partial)

        row = int(frac * left.shape[0])
        line_l.set_ydata([row, row])
        line_r.set_ydata([row, row])
        return im_d, line_l, line_r

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=100)
    save(ani, "stereo.gif")
    plt.close()


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks = [
        ("Canny",             make_canny_gif),
        ("Harris",            make_harris_gif),
        ("Pyramids",          make_pyramids_gif),
        ("Optical flow",      make_optical_flow_gif),
        ("Morphology",        make_morphology_gif),
        ("Shape from shading",make_sfs_gif),
        ("Stereo",            make_stereo_gif),
    ]
    for name, fn in tasks:
        print(f"Generating {name}...")
        fn()
    print("Done.")
