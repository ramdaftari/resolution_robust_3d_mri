"""
Save presentation figures (image-only, no axes/text), in the (Y, Z) plane
at a fixed mid-X (readout) index, where the 2D Poisson undersampling pattern
is actually visible:
  - <kspace_name> : undersampled k-space, dc-centered, log-magnitude
  - <recon_name>  : reconstructed slice (magnitude)

SCALING_FACTOR is parsed from <pt_dir>/run.log ("scaling_factor=...").

Run from resolution_robust_3d_mri/ with .venv activated.
"""
import argparse
import json
import re
import numpy as np
np.complex = complex; np.float = float; np.int = int
import lmdb
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQRT2_FIX = np.complex64(0.5 - 0.5j)
VOL_IDX = 0

p = argparse.ArgumentParser()
p.add_argument("pt_dir", help="dir containing final_rec_0.pt, ground_truth_0.pt, run.log")
p.add_argument("png_dir", nargs="?", default=None, help="dir to save PNGs (default: pt_dir)")
p.add_argument("--lmdb", required=True, help="LMDB path used for this reconstruction")
p.add_argument("--recon_name", default="recon_yz.png")
p.add_argument("--kspace_name", default="kspace_yz.png")
p.add_argument("--skip_kspace", action="store_true")
args = p.parse_args()

PT_DIR = args.pt_dir
PNG_DIR = args.png_dir or PT_DIR
LMDB_PATH = args.lmdb


def to_mag(t):
    t = t.float()
    mag = torch.hypot(t[..., 0], t[..., 1])
    while mag.dim() > 3:
        mag = mag.squeeze(0)
    return mag.detach().numpy()


def save_image(arr, path, vmax=None):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.rot90(arr), cmap="gray", vmin=0, vmax=vmax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_lmdb_array(name, idx):
    key = str(idx).encode()
    opts = dict(readonly=True, lock=False, readahead=False)
    with lmdb.open(f"{LMDB_PATH}/shapes", **opts).begin() as t:
        shapes = json.loads(t.get(key).decode())
    actual_key = "mask" if name == "masks" else name
    with lmdb.open(f"{LMDB_PATH}/{name}", **opts).begin() as t:
        return np.frombuffer(t.get(key), dtype=np.complex64).reshape(tuple(shapes[actual_key])).copy()


# ---- scaling factor (data-dependent: resolution + acceleration) ------------
with open(f"{PT_DIR}/run.log") as f:
    log_text = f.read()
m = re.search(r"scaling_factor=([\d.]+)", log_text)
SCALING_FACTOR = float(m.group(1))
print(f"SCALING_FACTOR={SCALING_FACTOR}")

# ---- Reconstruction slice (Y, Z) at mid-X -----------------------------------
rec = torch.load(f"{PT_DIR}/final_rec_0.pt", map_location="cpu")
gt = torch.load(f"{PT_DIR}/ground_truth_0.pt", map_location="cpu")

rec_mag = to_mag(rec) / SCALING_FACTOR  # (X, Y, Z), back to GT scale
gt_mag = to_mag(gt)

X, Y, Z = rec_mag.shape
x_mid = X // 2

rec_slice = rec_mag[x_mid, :, :]
gt_slice = gt_mag[x_mid, :, :]
vmax = float(np.percentile(gt_slice, 99.5))

save_image(rec_slice, f"{PNG_DIR}/{args.recon_name}", vmax=vmax)
print(f"Saved {PNG_DIR}/{args.recon_name}")

# ---- undersampled k-space (Y, Z) at mid-X, dc-centered ----------------------
if not args.skip_kspace:
    kspace = load_lmdb_array("kspace", VOL_IDX) * SQRT2_FIX  # (C, X, Y, Z)
    mask = load_lmdb_array("masks", VOL_IDX)                  # (1, X, Y, Z) m*(1+j)

    ks_masked = kspace * (mask.real * 2)  # (1+j)*m -> real part = m
    ks_centered = np.fft.fftshift(ks_masked, axes=(-3, -2, -1))

    ks_slice = ks_centered[:, x_mid, :, :]                  # (C, Y, Z)
    ks_mag = np.sqrt((np.abs(ks_slice) ** 2).sum(axis=0))   # combine coils
    ks_log = np.log1p(ks_mag)
    vmax_ks = float(np.percentile(ks_log, 99.9))

    save_image(ks_log, f"{PNG_DIR}/{args.kspace_name}", vmax=vmax_ks)
    print(f"Saved {PNG_DIR}/{args.kspace_name}")
