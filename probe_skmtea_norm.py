"""
probe_skmtea_norm.py
--------------------
SKM-TEA-side counterpart of probe_brats_norm.py — measures the
post-`scale_target_by_kspacenorm` per-voxel std of MVUE slices produced
by SkmteaSliceDataset + FastMRI2DDataTransform, so we can verify that the
SKM-TEA training-time target_scaling_factor of 2.0 lands targets at
unit variance (and that our 3506.266 for BraTS plays the same role).
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

from src.datasets.skmtea_slice_dataset import SkmteaSliceDataset
from src.problem_trafos.dataset_trafo.fastmri_2d_trafo import FastMRI2DDataTransform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/scratch/10846/armeet/datasets/skmtea/files_recon_calib-24")
    p.add_argument("--csv", default="/work/10471/peterwg/vista/r/3dmri_new/mri3d/src/datasets/train.csv")
    p.add_argument("--n_volumes", type=int, default=5)
    p.add_argument("--slices_per_volume", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Same trafo as mri2d_train_calc_mvue.yaml EXCEPT target_scaling_factor=1.0
    # so we observe std BEFORE the global scalar.
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
        target_scaling_factor=1.0,                 # probe override
        normalize_target=False,
        target_type="fullysampled_rec",
        multicoil_reduction_op="norm_sum_sensmaps",
        target_interpolate_by_factor=1.0,          # probe override
        target_interpolate_factor_is_interval=True,
        target_interpolate_method="bilinear",
    )

    ds = SkmteaSliceDataset(
        data_root=args.data_root,
        csv_path=args.csv,
        echo=1,
        transform=trafo,
        return_sensmaps=True,
    )

    fpaths = ds.fpaths
    n_vols = min(args.n_volumes, len(fpaths))
    sampled_vols = random.sample(fpaths, n_vols)

    stds = []
    print(f"Probing {n_vols} volumes x ~{args.slices_per_volume} slices each "
          f"from {args.data_root}\n")
    for fp in sampled_vols:
        vol_indices = [i for i, (f, _, _) in enumerate(ds.raw_samples) if f == fp]
        picks = random.sample(vol_indices, min(args.slices_per_volume, len(vol_indices)))
        for i in picks:
            sample = ds[i]
            target_torch = sample[1] if not hasattr(sample, "target") else sample.target
            stds.append(float(target_torch.std()))

    arr = np.array(stds)
    median = float(np.median(arr))
    print(f"  samples: {len(arr)}")
    print(f"  mean std:   {arr.mean():.6f}")
    print(f"  median std: {median:.6f}")
    print(f"  min/max:    {arr.min():.6f} / {arr.max():.6f}\n")

    print(f"With training-time target_scaling_factor=2.0 the post-stage-2 std would be:")
    print(f"  median: {median * 2.0:.4f}   (target: ~1.0)")
    print(f"  mean:   {arr.mean() * 2.0:.4f}")
    print(f"  range:  [{arr.min() * 2.0:.4f}, {arr.max() * 2.0:.4f}]\n")
    print(f"Suggested factor 1/median_std = {1.0/median:.4f}")


if __name__ == "__main__":
    main()
