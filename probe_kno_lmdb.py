#!/usr/bin/env python3
"""
Probe KNO val LMDB: confirm shapes, dtypes, mask format, normalisation.

Run on Vista login node:
    cd /work/10471/peterwg/vista/r/3dmri_new/mri3d/src/baselines/resolution_robust_3d_mri
    export UV_CACHE_DIR=$WORK/.uv_cache
    uv pip install lmdb            # only if not already in .venv
    uv run python probe_kno_lmdb.py

No GPU needed, ~5 seconds total. Paste the full stdout back to continue.
"""
import json
from pathlib import Path

import lmdb
import numpy as np

VAL_LMDB = Path("/scratch/10846/armeet/datasets/skmtea_val_0.5first2_4x_lmdb")
NORM_CONSTANT = 3e7  # LMDBVolumeDataset applies this at load time (not baked into bytes)


def open_ro(subdir):
    return lmdb.open(str(VAL_LMDB / subdir), readonly=True, lock=False, readahead=False, meminit=False)


def get_bytes(env, key):
    with env.begin() as txn:
        b = txn.get(key)
    assert b is not None, f"missing key {key!r}"
    return b


def row(name, arr):
    mag = np.abs(arr)
    print(
        f"  {name:6s} shape={arr.shape}  dtype={arr.dtype}  "
        f"|a|∈[{mag.min():.3e}, {mag.max():.3e}]  ‖a‖₂={np.linalg.norm(arr):.3e}"
    )


def main():
    # 1. enumerate volumes
    with open_ro("shapes") as env, env.begin() as txn:
        keys = sorted((k.decode() for k, _ in txn.cursor()), key=int)
    print(f"{VAL_LMDB}")
    print(f"  {len(keys)} volumes  (first={keys[0]!r}, last={keys[-1]!r})\n")

    # 2. shape metadata for the first sample
    k0 = keys[0].encode()
    with open_ro("shapes") as env:
        shapes_info = json.loads(get_bytes(env, k0).decode("utf-8"))
    print(f"sample {keys[0]} shape metadata:")
    for k, s in shapes_info.items():
        print(f"  {k}: {tuple(s)}")
    print()

    # 3. load raw tensors (bytes are complex64; norm_constant division is at load
    #    time in LMDBVolumeDataset.__getitem__, not baked into LMDB bytes)
    def load(subdir, shape_key):
        with open_ro(subdir) as env:
            raw = get_bytes(env, k0)
        return np.frombuffer(raw, dtype=np.complex64).reshape(tuple(shapes_info[shape_key]))

    k_raw  = load("kspace", "kspace")
    m_raw  = load("maps",   "maps")
    mk_raw = load("masks",  "mask")
    t_raw  = load("target", "target")

    # post-normalisation (what LMDBVolumeDataset returns at runtime):
    k   = k_raw / NORM_CONSTANT
    tgt = t_raw / NORM_CONSTANT
    maps = m_raw   # NOT divided
    mask = mk_raw  # NOT divided

    print("post-load tensor stats (kspace,target divided by 3e7; maps,mask as-is):")
    row("kspace", k)
    row("maps",   maps)
    row("mask",   mask)
    row("target", tgt)
    print()

    # 4. mask format interrogation — the main unknown
    r_vals = np.unique(np.round(mask.real, 6))
    i_vals = np.unique(np.round(mask.imag, 6))
    re_eq_im = np.allclose(mask.real, mask.imag)
    active = np.abs(mask) > 1e-9
    print("mask interrogation:")
    print(f"  unique real values (first 8): {r_vals[:8]}")
    print(f"  unique imag values (first 8): {i_vals[:8]}")
    print(f"  real == imag everywhere?      {re_eq_im}")
    print(f"  active fraction:              {active.sum()/active.size:.4f}"
          f"  (undersampling ≈ {active.size/max(active.sum(),1):.2f}×)")
    if mask.ndim == 4 and mask.shape[0] == 1:
        tiled_along_x = np.allclose(mask - mask[:, :1], 0)
        print(f"  identical across axis 1 (X, readout)? {tiled_along_x}")
    print()

    # 5. kspace masking self-consistency: is stored kspace actually zero where mask is zero?
    # reduce mask to 3D (drop coil / batch), take active positions
    mask_3d = np.abs(mask).squeeze()  # expected (X, Y, Z)
    if mask_3d.ndim == 3 and mask_3d.shape == k.shape[-3:]:
        active_3d = mask_3d > 1e-9
        # aggregate |k| over coils
        k_mag_per_voxel = np.abs(k).sum(axis=0)  # (X, Y, Z)
        inact = k_mag_per_voxel[~active_3d]
        act   = k_mag_per_voxel[active_3d]
        print("kspace-vs-mask self-consistency:")
        print(f"  mean |k| at mask=0:  {inact.mean():.3e}  (should be ~0)")
        print(f"  max  |k| at mask=0:  {inact.max():.3e}")
        print(f"  mean |k| at mask=1:  {act.mean():.3e}")
        print()
    else:
        print(f"kspace-vs-mask: mask shape {mask.shape} couldn't be reduced to match k shape {k.shape}\n")

    # 6. prior-scale sanity — res_rob prior was trained on |slice| ~ O(1) via scale_target_by_kspacenorm
    print("prior-scale sanity:")
    print(f"  max  |target| after /3e7 = {np.abs(tgt).max():.4f}")
    print(f"  mean |target| after /3e7 = {np.abs(tgt).mean():.4f}")
    print("  (these will be rescaled to ~O(1) at recon time via rescale_observation)")


if __name__ == "__main__":
    main()