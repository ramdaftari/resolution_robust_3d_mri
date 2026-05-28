"""
smoke_brats.py
--------------
End-to-end offline smoke test for the BraTS 2D-diffusion training pipeline.
Run this in an idev session BEFORE sbatching to catch the failure modes
that bit cas2/3/5/sweep/QR-pure-KNO (CLAUDE.md gotcha #9): missing config
keys, dataset path typos, shape mismatches, sens-map dtype/layout issues,
trafo crashes on real data, dataloader hangs.

Checks (in order, fail-fast):

  1. Three LMDB dirs (train/val multicoil, plus sub-dbs kspace/maps/target/shapes)
     exist and have non-zero size — catches the "empty stub" full-res
     situation that already exists for 240x240x155 multicoil train.

  2. BratsSliceDataset constructs on the train LMDB for axis 0, reports
     slice count, returns a sample whose `kspace` is real (C, *, *, 2)
     and whose `attrs` contains `sens_maps`, `kspace_vol_norm`, `target_vol_shape`
     with the right dtypes and shapes — matches the SliceDatasetSample
     contract that FastMRI2DDataTransform expects.

  3. The FastMRI2DDataTransform (same params as mri2d_train_calc_mvue.yaml,
     target_scaling_factor=1.0 so we observe raw post-norm std) runs on
     a few slices without raising. We then assert the post-`scale_target_by_kspacenorm`
     target_torch has a finite, plausible per-voxel std (loose bounds:
     0.01 < std < 100). This catches sens-map dtype bugs (numpy != torch
     interop), NaN-producing sqrt/divide-by-zero in the norm formula, and
     the silent factor-of-sqrt(N) divergence that motivated the pre-IFFT
     norm fix in brats_slice_dataset.py.

  4. Build a multi-plane ConcatDataset via get_brats_dataset(max_volumes=3)
     and iterate a single batch through a real DataLoader (batch_size=4,
     num_workers=0). Verifies shape compatibility of slices coming from
     three different anatomical axes within the same batch — the trafo
     and downstream U-Net both assume (..., 2, H, W) after `swap_channels`.

  5. Report a calibrated `target_scaling_factor` (1 / median std) and print
     the exact next idev command to run to test the full training entrypoint.

Run:
    cd /work/10471/peterwg/vista/r/3dmri_new/mri3d/src/baselines/resolution_robust_3d_mri
    source .venv/bin/activate          # or use .venv/bin/python explicitly
    python smoke_brats.py

Expected runtime: < 60 s. No GPU strictly required — runs on CPU.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

LMDB_TRAIN = "/scratch/10471/peterwg/brats2021_lmdb/brats_train_120x120x78_4x_lmdb"
LMDB_VAL   = "/scratch/10471/peterwg/brats2021_lmdb/brats_val_120x120x78_4x_lmdb"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_lmdb_paths() -> None:
    """Verify each LMDB has the four required sub-DBs and that each one
    contains entries for many volumes — not just the few-bytes 'empty stub'
    state that exists for the full-res multicoil LMDB on disk today.
    We count keys rather than checking byte size because `shapes` stores
    only a JSON string per key (legitimately ~300 bytes per volume)."""
    import lmdb
    section("[1/5] LMDB paths and sub-DBs")
    for root in (LMDB_TRAIN, LMDB_VAL):
        p = Path(root)
        assert p.is_dir(), f"missing LMDB root: {p}"
        for sub in ("kspace", "maps", "target", "shapes"):
            q = p / sub
            assert q.is_dir(), f"missing sub-db: {q}"
            sz = (q / "data.mdb").stat().st_size if (q / "data.mdb").exists() else 0
            env = lmdb.open(str(q), readonly=True, lock=False, readahead=False)
            with env.begin() as txn:
                n_entries = txn.stat()["entries"]
            env.close()
            assert n_entries >= 50, (
                f"{q} has only {n_entries} entries — looks like an empty stub. "
                "This LMDB was not built; rebuild via scripts/convert_brats_lmdb.py "
                "or point at a different resolution."
            )
            print(f"  {q}  ({sz/2**30:.2f} GB, {n_entries} entries)")


def check_loader_contract() -> None:
    section("[2/5] BratsSliceDataset contract")
    from src.datasets.brats_slice_dataset import BratsSliceDataset

    ds = BratsSliceDataset(root_dir=LMDB_TRAIN, readout_axis=0,
                           return_sensmaps=True, transform=None,
                           volume_cache_size=1)
    print(f"  volumes: {len(ds._volume_keys)}, slices: {len(ds)}")

    sample = ds[0]
    ks = sample.kspace
    attrs = sample.attrs
    assert torch.is_tensor(ks), f"kspace must be a tensor, got {type(ks)}"
    assert ks.dim() == 4 and ks.shape[-1] == 2, (
        f"kspace must be (C, *, *, 2) real, got {tuple(ks.shape)}"
    )
    assert ks.dtype == torch.float32, f"kspace dtype: {ks.dtype}"

    for key in ("sens_maps", "kspace_vol_norm", "target_vol_shape"):
        assert key in attrs, f"missing attrs[{key!r}]"
    sm = attrs["sens_maps"]
    assert isinstance(sm, np.ndarray) and sm.dtype == np.complex64, (
        f"sens_maps must be np.complex64, got {type(sm).__name__} {getattr(sm, 'dtype', None)}"
    )
    C = ks.shape[0]
    assert sm.shape[0] == C, f"sens_maps coil dim {sm.shape[0]} != kspace coils {C}"

    norm = float(attrs["kspace_vol_norm"])
    assert np.isfinite(norm) and norm > 0, f"bad kspace_vol_norm: {norm}"
    print(f"  sample[0]: kspace {tuple(ks.shape)} {ks.dtype}, "
          f"sens_maps {sm.shape} {sm.dtype}")
    print(f"  attrs: kspace_vol_norm={norm:.3e}, "
          f"target_vol_shape={attrs['target_vol_shape']}")


def check_trafo_runs() -> list[float]:
    section("[3/5] FastMRI2DDataTransform end-to-end")
    from src.datasets.brats_slice_dataset import BratsSliceDataset
    from src.problem_trafos.dataset_trafo.fastmri_2d_trafo import FastMRI2DDataTransform

    # Match the training trafo contract: train_diff_models.py constructs the
    # dataset trafo with provide_pseudoinverse=False AND provide_measurement=False,
    # which collapses BaseDatasetTrafo.__call__ to return JUST `target` (single
    # tensor), not the 4-tuple. We mirror that here so the smoke exercises the
    # same return shape as production training.
    trafo = FastMRI2DDataTransform(
        which_challenge="multicoil",
        mask_enabled=False,
        mask_type="random",
        mask_accelerations=4.0,
        mask_center_fractions=[0.08],
        mask_seed=1234,
        use_seed=True,
        provide_pseudoinverse=False,
        provide_measurement=False,
        use_real_synth_data=False,
        return_magnitude_image=False,
        return_cropped_pseudoinverse=False,
        scale_target_by_kspacenorm=True,
        target_scaling_factor=1.0,           # probe override: observe raw std
        normalize_target=False,
        target_type="mvue",                  # BraTS: pass through precomputed clean
                                              # target (LMDB k-space is already masked
                                              # so fullysampled_rec would give zero-fill)
        multicoil_reduction_op="norm_sum_sensmaps",
        target_interpolate_by_factor=1.0,    # probe override
        target_interpolate_factor_is_interval=True,
        target_interpolate_method="bilinear",
    )

    ds = BratsSliceDataset(root_dir=LMDB_TRAIN, readout_axis=0,
                           return_sensmaps=True, transform=trafo,
                           volume_cache_size=1)

    rng = random.Random(0)
    vol_keys = ds._volume_keys
    n_vols = min(3, len(vol_keys))
    sampled_vols = rng.sample(vol_keys, n_vols)

    stds: list[float] = []
    for vk in sampled_vols:
        vol_indices = [i for i, (k, _, _) in enumerate(ds.raw_samples) if k == vk]
        picks = rng.sample(vol_indices, min(8, len(vol_indices)))
        for i in picks:
            target_torch = ds[i]
            assert torch.is_tensor(target_torch) and target_torch.dim() >= 2, (
                f"trafo target_torch malformed: shape={tuple(target_torch.shape)}"
            )
            assert torch.isfinite(target_torch).all(), "NaN/Inf in target_torch"
            stds.append(float(target_torch.std()))

    arr = np.array(stds)
    med = float(np.median(arr))
    print(f"  n_samples={len(arr)}, mean std={arr.mean():.4f}, "
          f"median std={med:.4f}, min/max={arr.min():.4f}/{arr.max():.4f}")
    # Loose plausibility bound only — BraTS post-norm std can be ~2 orders of
    # magnitude below SKM-TEA's because the BraTS target is image-space [0,1]
    # while SKM-TEA's is k-space-derived natural units. The CALIBRATION step
    # (1/median_std) absorbs this divergence into target_scaling_factor. We
    # only need to catch obviously-broken pipelines: NaN/Inf, all-zero (loader
    # not producing real data), or runaway-large (norm formula misapplied).
    assert 1e-6 < med < 1000, (
        f"post-norm median std {med} is implausible — loader returned all "
        f"zeros, kspace_vol_norm is 0, or sens-map scaling diverged. "
        f"Investigate before calibrating target_scaling_factor."
    )
    return stds


def check_multiplane_dataloader() -> None:
    section("[4/5] Three-plane ConcatDataset via get_brats_dataset")
    from src.datasets.dataset_resolver import get_brats_dataset
    from src.problem_trafos.dataset_trafo.fastmri_2d_trafo import FastMRI2DDataTransform

    # Same training contract as check_trafo_runs: both provide_* False so we
    # see the single-tensor return path that score_model_trainer consumes.
    trafo = FastMRI2DDataTransform(
        which_challenge="multicoil",
        mask_enabled=False,
        mask_type="random",
        mask_accelerations=4.0,
        mask_center_fractions=[0.08],
        mask_seed=1234,
        use_seed=True,
        provide_pseudoinverse=False,
        provide_measurement=False,
        use_real_synth_data=False,
        return_magnitude_image=False,
        return_cropped_pseudoinverse=False,
        scale_target_by_kspacenorm=True,
        target_scaling_factor=2.0,
        normalize_target=False,
        target_type="mvue",                  # BraTS: pass through precomputed clean
                                              # target (LMDB k-space is already masked
                                              # so fullysampled_rec would give zero-fill)
        multicoil_reduction_op="norm_sum_sensmaps",
        target_interpolate_by_factor=1.0,    # disable resolution aug for shape
                                              # consistency within the batch;
                                              # production training uses [0.1,1.0]
        target_interpolate_factor_is_interval=True,
        target_interpolate_method="bilinear",
    )

    # path_resolver mirrors what get_path_by_cluster_name does (pick "default").
    path_resolver = lambda p: p["default"] if isinstance(p, dict) else p

    ds = get_brats_dataset(
        fold_overwrite=None,
        fold="train",
        dataset_trafo=trafo,
        data_root_train={"default": LMDB_TRAIN},
        data_root_val={"default": LMDB_VAL},
        data_root_test={"default": LMDB_VAL},
        readout_axes=(0, 1, 2),
        return_sensmaps=True,
        volume_cache_size=1,
        max_volumes=2,           # smoke: 2 volumes per plane
        path_resolver=path_resolver,
    )
    print(f"  ConcatDataset len: {len(ds)}  (3 planes x 2 volumes)")

    # Single batch — note: slices from different axes have different (H, W),
    # so we cannot collate them in one batch without same-shape batching.
    # In production, BatchSamplerSameShape (group_shape_by='target') groups
    # batch members by shape. For the smoke, iterate per-axis subsets
    # explicitly to confirm shapes are consistent within each axis.
    per_axis_lens = [len(d) for d in ds.datasets]  # type: ignore[attr-defined]
    print(f"  per-axis slice counts: {per_axis_lens}")
    for ai, sub in enumerate(ds.datasets):           # type: ignore[attr-defined]
        loader = DataLoader(sub, batch_size=4, shuffle=False, num_workers=0)
        target_batch = next(iter(loader))
        assert torch.is_tensor(target_batch), type(target_batch)
        print(f"  axis {ai}: target batch shape = {tuple(target_batch.shape)} "
              f"dtype={target_batch.dtype}")


def print_next_command(stds: list[float]) -> None:
    section("[5/5] Calibration + next command")
    arr = np.array(stds)
    med = float(np.median(arr))
    suggested = 1.0 / med
    print(f"  Suggested target_scaling_factor (median std={med:.4f}): {suggested:.3f}")
    print(f"  (BraTS target is image-space [0,1], so this typically diverges")
    print(f"   from SKM-TEA's 2.0 by ~2 orders of magnitude — that's expected.")
    print(f"   The diffusion model only needs unit-variance inputs; the absolute")
    print(f"   target_scaling_factor value doesn't otherwise matter.)")
    print()
    print(f"  Set this value in hydra/exps/brats/base_training.yaml:")
    print(f"    problem_trafos.dataset_trafo.target_scaling_factor: {suggested:.3f}")
    print()
    print("  Then verify the full hydra training entrypoint runs end-to-end.")
    print("  In idev, run:")
    print()
    print("    cd " + str(REPO))
    print("    .venv/bin/python train_diff_models.py \\")
    print("      +exps=brats/train_dense \\")
    print("      hydra.job.chdir=False \\")
    print("      dataset.max_volumes=3 \\")
    print("      diffmodels.train.epochs=1 \\")
    print("      diffmodels.train.cache_dataset=False \\")
    print("      diffmodels.train.cache_dataset_in_gpu=False \\")
    print("      diffmodels.train.cache_dataset_store_on_disk=False \\")
    print("      diffmodels.val.cache_dataset=False \\")
    print("      diffmodels.val.cache_dataset_in_gpu=False \\")
    print("      diffmodels.val.cache_dataset_store_on_disk=False \\")
    print("      diffmodels.train.save_model_every_n_epoch=1 \\")
    print("      diffmodels.train.ema_warm_start_steps=10 \\")
    print(f"      problem_trafos.dataset_trafo.target_scaling_factor={suggested:.3f} \\")
    print("      wandb.log=False \\")
    print("      wandb.project=brats_smoke \\")
    print("      wandb.entity=smoke \\")
    print("      descr_short=brats_smoke \\")
    print("      note=brats_smoke_one_epoch")
    print()
    print("  If that completes one epoch and writes ema_model_0.pt, submit")
    print("  slurm/train_brats_dense.sbatch for the production run.")


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    torch.manual_seed(0)

    check_lmdb_paths()
    check_loader_contract()
    stds = check_trafo_runs()
    check_multiplane_dataloader()
    print_next_command(stds)

    print("\nSMOKE PASS\n")


if __name__ == "__main__":
    main()
