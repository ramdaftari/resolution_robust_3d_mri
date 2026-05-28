"""
probe_brats_norm.py
-------------------
Measure the post-`scale_target_by_kspacenorm` per-voxel std of MVUE slices
produced by BratsSliceDataset + FastMRI2DDataTransform, so we can pick
`target_scaling_factor` for BraTS training.

The paper [Mar+23] trains the diffusion model assuming inputs have variance
near 1. SKM-TEA's training config uses target_scaling_factor=2.0, which was
empirically calibrated for that dataset's signal level. BraTS goes through
Biot-Savart-simulated coils with NORM_CONSTANT=3e7 baked in at LMDB-build
time, so its post-scaling distribution may differ.

Procedure: load a few volumes, run them through the trafo with
target_scaling_factor=1.0 (so we observe the std before the empirical
fudge), record target_torch.std() per slice, and suggest the calibrated
value as 1 / median_std.

Run:
    python probe_brats_norm.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets.brats_slice_dataset import BratsSliceDataset
from src.problem_trafos.dataset_trafo.fastmri_2d_trafo import FastMRI2DDataTransform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", default="/scratch/10471/peterwg/brats2021_lmdb/brats_train_120x120x78_4x_lmdb")
    p.add_argument("--n_volumes", type=int, default=5)
    p.add_argument("--slices_per_volume", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Trafo matches mri2d_train_calc_mvue.yaml exactly EXCEPT for two probe-only
    # overrides:
    #   * target_scaling_factor=1.0  -- we want the std BEFORE the fudge factor
    #   * target_interpolate_by_factor=1.0 -- skip resolution augmentation so
    #     the measurement reflects the native scale; the per-slice std should
    #     be approximately scale-invariant under bilinear downsampling
    #     anyway, but isolating it removes a confound.
    trafo = FastMRI2DDataTransform(
        which_challenge="multicoil",
        mask_enabled=False,
        mask_type="random",
        mask_accelerations=4.0,
        mask_center_fractions=[0.08],
        mask_seed=1234,
        use_seed=True,
        use_real_synth_data=False,
        return_magnitude_image=False,
        return_cropped_pseudoinverse=False,
        scale_target_by_kspacenorm=True,
        target_scaling_factor=1.0,          # probe override
        normalize_target=False,
        target_type="fullysampled_rec",
        multicoil_reduction_op="norm_sum_sensmaps",
        target_interpolate_by_factor=1.0,    # probe override
        target_interpolate_factor_is_interval=True,
        target_interpolate_method="bilinear",
    )

    ds = BratsSliceDataset(root_dir=args.lmdb, readout_axis=0,
                           return_sensmaps=True, transform=trafo,
                           volume_cache_size=1)

    # Sample roughly args.n_volumes * args.slices_per_volume slices uniformly
    # across the LMDB so we cover multiple volumes (the dataset's flat sample
    # list interleaves slices from each volume in order).
    vol_keys = ds._volume_keys
    n_vols = min(args.n_volumes, len(vol_keys))
    sampled_vols = random.sample(vol_keys, n_vols)

    stds = []
    print(f"Probing {n_vols} volumes x ~{args.slices_per_volume} slices each "
          f"from {args.lmdb}\n")
    for vk in sampled_vols:
        # find the contiguous range of flat-sample indices for this volume
        vol_indices = [i for i, (k, _, _) in enumerate(ds.raw_samples) if k == vk]
        picks = random.sample(vol_indices, min(args.slices_per_volume, len(vol_indices)))
        for i in picks:
            _, target_torch, _, _ = ds[i]
            stds.append(float(target_torch.std()))

    arr = np.array(stds)
    print(f"  samples: {len(arr)}")
    print(f"  mean std:   {arr.mean():.4f}")
    print(f"  median std: {np.median(arr):.4f}")
    print(f"  min/max:    {arr.min():.4f} / {arr.max():.4f}\n")

    suggested = 1.0 / float(np.median(arr))
    print(f"Suggested target_scaling_factor (to bring median std to ~1.0):")
    print(f"    {suggested:.3f}")
    print(f"(SKM-TEA uses 2.0 — if BraTS suggestion is within ~20% of 2.0,")
    print(f" keep the SKM-TEA value for consistency; otherwise override")
    print(f" problem_trafos.dataset_trafo.target_scaling_factor in")
    print(f" hydra/exps/brats/base_training.yaml.)")


if __name__ == "__main__":
    main()
