#!/usr/bin/env python3
"""
reconstruct_modified.py

Variational reconstruction with diffusion prior on LMDB val data.

Two and ONLY two differences from reconstruct.py + varrecon_voxelrep_diffprior_4x:
  1. Dataset:  LMDBVolumeDataset with 3D Gaussian mask (instead of H5 + Poisson2D mask)
  2. PSNR:     magnitude-based formula (instead of res_rob's scalar-max)

Everything else is identical to reconstruct.py:
  - Trafo configs loaded from the same YAML files
  - get_fwd_trafo / get_target_trafo / get_prior_trafo resolvers
  - Same score model loader, SDE, slice methods, variational objective, fit()
  - Same rescale_observation logic (constant_scaling_factor=1.0)
  - Same FixedGridRepresentation warm-started from filtbackproj * scaling_factor

Run from baselines/resolution_robust_3d_mri/:
    .venv/bin/python reconstruct_modified.py
    .venv/bin/python reconstruct_modified.py --num_volumes 1 --no_wandb --device cuda:0
"""

import sys
import os
import math
import logging
import argparse
from pathlib import Path
from itertools import islice
from functools import partial

import numpy as np
import numpy as _np_orig

# meddlr (imported transitively) uses deprecated np.complex alias
if not hasattr(np, 'complex'):
    np.complex = complex  # type: ignore[attr-defined]

import torch
import torch.nn.functional as _F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: mri3d venv first (for torchmetrics 0.11.4), mri3d src last
# ---------------------------------------------------------------------------
MRI3D_SRC  = Path(__file__).resolve().parents[2]        # …/mri3d/src/
MRI3D_VENV = Path(__file__).resolve().parents[3] / ".venv/lib/python3.10/site-packages"
sys.path.insert(0, str(MRI3D_VENV))   # torchmetrics 0.11.4 (needed by meddlr)
sys.path.append(str(MRI3D_SRC))       # datasets/, src/fastmri/ (mri3d-local, partial)

import meddlr.metrics.functional as _meddlr_metrics

# skmtea.py imports `from fastmri.subsample import MaskFunc` (mri3d-local path).
# The installed fastmri exposes it at fastmri.data.subsample — add alias.
import fastmri.data.subsample as _fmri_sub
sys.modules.setdefault("fastmri.subsample", _fmri_sub)

# ---------------------------------------------------------------------------
# res_rob imports (script must be run from baselines/resolution_robust_3d_mri/)
# ---------------------------------------------------------------------------
from src.problem_trafos.trafo_resolver import get_fwd_trafo, get_target_trafo, get_prior_trafo
from src.diffmodels.diffmodels_resolver import create_dense_model
from src.diffmodels.ema import ExponentialMovingAverage
from src.diffmodels.sde import DDPM
from src.reconstruction.variational.fit import fit
from src.reconstruction.variational.var_objectives import get_variational_objective, DiffusionVariationanlObjective
from src.reconstruction.utils.pass_through import ScoreWithIdentityGradWrapper
from src.representations.representation_resolver import get_mesh, get_slice_method
from src.representations.fixed_grid_representation import FixedGridRepresentation
from src.sample_logger.SampleLoggerWithTarget import SampleLoggerWithTarget

# ---------------------------------------------------------------------------
# mri3d imports (available after sys.path.append above)
# ---------------------------------------------------------------------------
from datasets.skmtea import LMDBVolumeDataset


# ---------------------------------------------------------------------------
# Magnitude PSNR: 20*log10(max(|gt|) / RMSE(|rec|, |gt|)) per (B,C)
# ---------------------------------------------------------------------------
def _mag_psnr_fn(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    B, C = pred_mag.shape[:2]
    pred_flat   = pred_mag.view(B, C, -1).float()
    target_flat = target_mag.view(B, C, -1).float()
    rmse    = (pred_flat - target_flat).pow(2).mean(dim=-1).sqrt()
    max_val = target_flat.abs().amax(dim=-1)
    return (20.0 * torch.log10(max_val / (rmse + 1e-8))).mean()


# ---------------------------------------------------------------------------
# MagSampleLogger — extends SampleLoggerWithTarget, replaces only PSNR calls
# ---------------------------------------------------------------------------
class MagSampleLogger(SampleLoggerWithTarget):
    """
    Identical to SampleLoggerWithTarget except PSNR is computed on magnitudes:
    take complex magnitude first, then 20*log10(max/RMSE) per (B,C) volume.
    """

    @staticmethod
    def _vol_psnr(rec: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        rec, gt: (X, Y, Z, 2)  real-as-complex tensors in the same scale.
        Returns scalar magnitude PSNR (dB).
        """
        def to_mag(t):
            return torch.view_as_complex(t.contiguous()).abs() if t.shape[-1] == 2 else t.abs()
        rec_mag = to_mag(rec).unsqueeze(0).unsqueeze(0)    # (1,1,X,Y,Z)
        gt_mag  = to_mag(gt).unsqueeze(0).unsqueeze(0)     # (1,1,X,Y,Z)
        return _mag_psnr_fn(rec_mag, gt_mag).squeeze()

    @staticmethod
    def _vol_ssim(rec: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        rec, gt: (X, Y, Z, 2)  real-as-complex tensors in the same scale.
        Returns scalar magnitude SSIM using meddlr.metrics.functional.ssim — identical
        to how SSIM is computed in the mri3d lightning modules.
        """
        def to_mag(t):
            return torch.view_as_complex(t.contiguous()).abs() if t.shape[-1] == 2 else t.abs()
        rec_mag = to_mag(rec).unsqueeze(0).unsqueeze(0)    # (1,1,X,Y,Z)
        gt_mag  = to_mag(gt).unsqueeze(0).unsqueeze(0)     # (1,1,X,Y,Z)
        return _meddlr_metrics.ssim(rec_mag, gt_mag).mean()

    @staticmethod
    def _vol_nmse(rec: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        rec, gt: (X, Y, Z, 2)  real-as-complex tensors in the same scale.
        Returns scalar NMSE = ||rec_mag - gt_mag||^2 / ||gt_mag||^2.
        Identical to eval_ckpts_to_latex.py line:
            ((pm - tm)**2).sum() / (tm**2).sum()
        which matches fastmri/evaluate.py nmse() convention.
        """
        def to_mag(t):
            return torch.view_as_complex(t.contiguous()).abs() if t.shape[-1] == 2 else t.abs()
        rec_mag = to_mag(rec).float()
        gt_mag  = to_mag(gt).float()
        return (rec_mag - gt_mag).pow(2).sum() / gt_mag.pow(2).sum()

    # Steps (0-indexed) at which to log all 3 metrics mid-optimization.
    # Corresponds to after iterations 10, 25, 50, 75, 100, 150.
    # Iteration 200 (step 199) is handled by close_sample_log.
    _CKPT_STEPS = frozenset({9, 24, 49, 74, 99, 149})

    # ---- Override init_run: also create SSIM/NMSE tracking arrays ----------
    def init_run(self, num_samples: int):
        super().init_run(num_samples)
        self.fbp_ssims = torch.zeros(num_samples)
        self.rec_ssims = torch.zeros(num_samples)
        self.fbp_nmses = torch.zeros(num_samples)
        self.rec_nmses = torch.zeros(num_samples)
        # Per-checkpoint tracking: key = display step (10,25,...,200), value = list of per-vol scalars
        ckpt_keys = sorted([s+1 for s in self._CKPT_STEPS]) + [200]
        self.ckpt_psnrs = {k: [] for k in ckpt_keys}
        self.ckpt_ssims = {k: [] for k in ckpt_keys}
        self.ckpt_nmses = {k: [] for k in ckpt_keys}

    # ---- Override init_sample_log: replace FBP PSNR only -------------------
    def init_sample_log(self, observation, filtbackproj, ground_truth,
                        sample_nr, scaling_factor, mesh):
        orig_log_psnr  = self.log_psnr
        self.log_psnr  = False                # suppress parent's PSNR block
        super().init_sample_log(
            observation=observation,
            filtbackproj=filtbackproj,        # correct keyword order (parent: fbp 2nd)
            ground_truth=ground_truth,
            sample_nr=sample_nr,
            scaling_factor=scaling_factor,
            mesh=mesh,
        )
        self.log_psnr  = orig_log_psnr

        with torch.no_grad():
            trafo_adj = self.fwd_trafo.trafo_adjoint(observation.to(self.device))
            tf_fbp    = self.target_trafo(trafo_adj).to(self.eval_device)
            gt        = ground_truth.to(self.eval_device)

            if orig_log_psnr:
                fbp_psnr = self._vol_psnr(tf_fbp, gt)
                self.fbp_psnrs[sample_nr] = fbp_psnr
                logging.info(f"[vol {sample_nr}] FBP PSNR = {fbp_psnr:.2f} dB")
                wandb.log({"fbp_psnr": fbp_psnr.item(),
                           "global_step": sample_nr, "step": sample_nr})

            fbp_ssim = self._vol_ssim(tf_fbp, gt)
            self.fbp_ssims[sample_nr] = fbp_ssim
            logging.info(f"[vol {sample_nr}] FBP SSIM = {fbp_ssim:.4f}")
            wandb.log({"fbp_ssim": fbp_ssim.item(),
                       "global_step": sample_nr, "step": sample_nr})

            fbp_nmse = self._vol_nmse(tf_fbp, gt)
            self.fbp_nmses[sample_nr] = fbp_nmse
            logging.info(f"[vol {sample_nr}] FBP NMSE = {fbp_nmse:.6f}")
            wandb.log({"fbp_nmse": fbp_nmse.item(),
                       "global_step": sample_nr, "step": sample_nr})

    # ---- Override __call__: replace per-step PSNR --------------------------
    def __call__(self, representation, step, pbar, log_dict={}):
        orig_log_psnr  = self.log_psnr
        self.log_psnr  = False
        super().__call__(representation, step, pbar, log_dict)
        self.log_psnr  = orig_log_psnr

        if not orig_log_psnr:
            return

        # Always update pbar PSNR every volume_stats_period steps
        if step % self.volume_stats_period == 0:
            with torch.no_grad():
                sample    = representation.forward_splitted(
                    self.mesh, self.eval_device, self.sample_gen_split)
                tf_sample = self.target_trafo(sample) / self.scaling_factor
                gt        = self.ground_truth.to(self.eval_device)
                p         = self._vol_psnr(tf_sample, gt)
                wandb.log({"rec_psnr": p.item(), "global_step": step})
                pbar.set_postfix_str(f"psnr={p:.1f} dB")

        # Log all 3 metrics at checkpoint steps (10, 25, 50, 75, 100, 150)
        if step not in self._CKPT_STEPS:
            return

        display_step = step + 1  # 1-indexed for display
        with torch.no_grad():
            sample    = representation.forward_splitted(
                self.mesh, self.eval_device, self.sample_gen_split)
            tf_sample = self.target_trafo(sample) / self.scaling_factor
            gt        = self.ground_truth.to(self.eval_device)
            p = self._vol_psnr(tf_sample, gt)
            s = self._vol_ssim(tf_sample, gt)
            n = self._vol_nmse(tf_sample, gt)
        self.ckpt_psnrs[display_step].append(p.item())
        self.ckpt_ssims[display_step].append(s.item())
        self.ckpt_nmses[display_step].append(n.item())
        logging.info(f"[vol {self.sample_nr} step {display_step:3d}] "
                     f"PSNR={p:.2f} dB  SSIM={s:.4f}  NMSE={n:.6f}")
        wandb.log({"rec_psnr": p.item(), "rec_ssim": s.item(),
                   "rec_nmse": n.item(),
                   "ckpt_step": display_step, "global_step": step})

    # ---- Override close_sample_log: replace final PSNR ---------------------
    def close_sample_log(self, representation):
        orig_log_psnr  = self.log_psnr
        self.log_psnr  = False
        super().close_sample_log(representation)
        self.log_psnr  = orig_log_psnr

        if not orig_log_psnr:
            return

        with torch.no_grad():
            sample    = representation.forward_splitted(
                self.mesh, self.eval_device, self.sample_gen_split)
            tf_sample = self.target_trafo(sample) / self.scaling_factor
            gt        = self.ground_truth.to(self.eval_device)
            p         = self._vol_psnr(tf_sample, gt)
            s         = self._vol_ssim(tf_sample, gt)
            n_val     = self._vol_nmse(tf_sample, gt)
            self.rec_psnrs[self.sample_nr] = p
            self.rec_ssims[self.sample_nr] = s
            self.rec_nmses[self.sample_nr] = n_val
            wandb.log({"rec_psnr_final":   p.item(),
                       "rec_psnrs_mean":   self.rec_psnrs[:self.sample_nr+1].mean().item(),
                       "rec_ssim_final":   s.item(),
                       "rec_ssims_mean":   self.rec_ssims[:self.sample_nr+1].mean().item(),
                       "rec_nmse_final":   n_val.item(),
                       "rec_nmses_mean":   self.rec_nmses[:self.sample_nr+1].mean().item(),
                       "global_step": self.sample_nr})
            logging.info(f"[vol {self.sample_nr} step 200] "
                         f"PSNR={p:.2f} dB  SSIM={s:.4f}  NMSE={n_val:.6f}")
            # Store final in checkpoint dicts for summary table
            self.ckpt_psnrs[200].append(p.item())
            self.ckpt_ssims[200].append(s.item())
            self.ckpt_nmses[200].append(n_val.item())

    # ---- Override close_run: print cross-volume summary table ---------------
    def close_run(self):
        super().close_run()
        steps = sorted(self.ckpt_psnrs.keys())
        if not any(self.ckpt_psnrs[s] for s in steps):
            return
        logging.info("=" * 72)
        logging.info("SUMMARY — mean across completed volumes")
        logging.info(f"{'step':>6}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'NMSE':>10}  {'n':>4}")
        logging.info("-" * 72)
        for s in steps:
            ps = self.ckpt_psnrs[s]; ss = self.ckpt_ssims[s]; ns = self.ckpt_nmses[s]
            if not ps:
                continue
            import statistics as _st
            logging.info(f"{s:>6}  {_st.mean(ps):>10.4f}  {_st.mean(ss):>8.4f}"
                         f"  {_st.mean(ns):>10.6f}  {len(ps):>4}")
        logging.info("=" * 72)


# ---------------------------------------------------------------------------
# Weighted data consistency: multiply datafit tensor by DATA_CON_WEIGHT.
# Must be done on the tensor before .backward() — a post-hoc float multiply
# would not change gradients since datafit.item() is already detached.
# ---------------------------------------------------------------------------
DATA_CON_WEIGHT = 1.0  # overridden by --data_con_weight at runtime

class WeightedDiffusionObjective(DiffusionVariationanlObjective):
    """DiffusionVariationanlObjective with datafit scaled by DATA_CON_WEIGHT."""

    def __call__(self, coord_rep, outer_iteration, inner_iteration):
        from functools import partial as _p
        from src.reconstruction.variational.noise_loss import noise_loss as _nl

        criterion = torch.nn.MSELoss()

        if inner_iteration in self.steps_data_con:
            if self.slice_method_data_con is not None:
                slices, slice_inds = self.slice_method_data_con(
                    coord_rep, self.mesh_data_con, self.mesh_data_con,
                    outer_iteration=outer_iteration, inner_iteration=inner_iteration)
                sv = self.slice_method_data_con.volume_indices[0]
                datafit = DATA_CON_WEIGHT * criterion(
                    self.fwd_trafo.trafo(slices[0], slice_inds[0].int(), sv),
                    self.observation.index_select(sv - slices[0].ndim, slice_inds[0].int()),
                ) / len(self.steps_data_con)
            else:
                datafit = DATA_CON_WEIGHT * criterion(
                    self.fwd_trafo(coord_rep.forward(self.mesh_data_con)),
                    self.observation,
                ) / len(self.steps_data_con)
        else:
            datafit = torch.zeros(1, device=coord_rep.device)

        if inner_iteration in self.steps_data_reg:
            nl = _p(_nl, outer_iteration=outer_iteration,
                    outer_iterations_max=self.outer_iterations_max,
                    score=self.score, sde=self.sde, repetition=1,
                    reg_strength=self.reg_strength,
                    adapt_reg_strength=self.adapt_reg_strength,
                    steps_scaler=self.steps_scaler,
                    time_sampling_method=self.time_sampling_method)
            slices = (self.slice_method_prior_reg(
                          coord_rep, self.mesh_data_reg, self.mesh_data_con,
                          outer_iteration=outer_iteration,
                          inner_iteration=inner_iteration)[0]
                      if self.slice_method_prior_reg is not None
                      else [coord_rep.forward(self.mesh_data_reg)])
            regfit = (sum(nl(self.prior_trafo(s)).mean() for s in slices)
                      / len(slices) / len(self.steps_data_reg))
        else:
            regfit = torch.zeros(1, device=coord_rep.device)

        return datafit + regfit, datafit.item(), regfit.item()


# ---------------------------------------------------------------------------
# Score model loader (bypasses hydra path resolver; arch from training config)
# ---------------------------------------------------------------------------
def load_score_model_direct(ckpt_path: Path, train_cfg_path: Path,
                             device: str) -> torch.nn.Module:
    train_cfg = OmegaConf.load(train_cfg_path)
    try:
        arch_cfg = train_cfg.arch
    except Exception:
        arch_cfg = train_cfg.diffmodels.arch

    if "name" not in arch_cfg:
        param_dict = {"name": "dense", "params": dict(arch_cfg)}
    else:
        param_dict = dict(arch_cfg)

    score = create_dense_model(**param_dict.get("params", param_dict)).to(device)
    ema   = ExponentialMovingAverage(score.parameters(), decay=0.999)
    ema.load_state_dict(torch.load(ckpt_path, map_location=device))
    ema.copy_to(score.parameters())
    logging.info(f"Loaded EMA score model from {ckpt_path}")
    return score


# ---------------------------------------------------------------------------
# Gaussian noise calibration
# Finds σ such that PSNR(obs + σ·n, obs) ≈ target_psnr_db in k-space.
# Formula: PSNR_kspace = 20·log10(max|obs_complex| / (σ·rms|noise_complex|))
# Closed-form: σ = max|obs| / (rms_noise · 10^(target/20))
# Returns (sigma, noise_tensor) — apply as: observation + sigma * noise_tensor
# ---------------------------------------------------------------------------
def find_noise_sigma(observation: torch.Tensor, target_psnr_db: float) -> tuple:
    with torch.no_grad():
        noise = torch.randn_like(observation)

        obs_complex   = torch.view_as_complex(observation.contiguous())
        noise_complex = torch.view_as_complex(noise.contiguous())

        max_obs   = obs_complex.abs().amax().item()
        rms_noise = noise_complex.abs().pow(2).mean().sqrt().item()  # ≈ √2 for randn

        sigma = max_obs / (rms_noise * 10 ** (target_psnr_db / 20.0))

        actual_psnr = 20.0 * math.log10(max_obs / (sigma * rms_noise + 1e-12))
        logging.info(f"Noise calibration: σ={sigma:.6f}  →  k-space PSNR={actual_psnr:.2f} dB "
                     f"(target={target_psnr_db} dB)")
        return sigma, noise


# ---------------------------------------------------------------------------
# Z-interpolation helper (mirrors Ruibo's full-res eval pipeline)
# Linearly interpolates any tensor along its last dimension to new_z.
# Handles both real and complex tensors.
# ---------------------------------------------------------------------------
def _interp_z(x: torch.Tensor, new_z: int) -> torch.Tensor:
    s   = x.shape
    x_  = x.reshape(-1, s[-1]).unsqueeze(1)          # (N, 1, Z)
    if torch.is_complex(x_):
        out = torch.complex(
            _F.interpolate(x_.real, size=new_z, mode="linear", align_corners=True),
            _F.interpolate(x_.imag, size=new_z, mode="linear", align_corners=True),
        )
    else:
        out = _F.interpolate(x_, size=new_z, mode="linear", align_corners=True)
    ns      = list(s); ns[-1] = new_z
    return out.squeeze(1).reshape(*ns)


# ---------------------------------------------------------------------------
# Config constants — mirrors varrecon_voxelrep_diffprior_4x + base_recon.yaml
# ---------------------------------------------------------------------------
VAL_LMDB       = Path("/scratch/10846/armeet/datasets/skmtea_val_0.5first2_4x_lmdb")
EMA_CKPT       = Path("recon_workdir/ema_model_110.pt")
TRAIN_CFG      = Path("recon_workdir/.hydra/config.yaml")
HYDRA_ROOT     = Path("hydra")
NORM_CONSTANT  = 3e7
H5_BASE        = Path("/scratch/10846/armeet/datasets/skmtea/files_recon_calib-24")


def _load_trafo_configs():
    """
    Load trafo configs from the same YAML files reconstruct.py uses via hydra.
    Applies the skmtea/varrecon_voxelrep_diffprior_4x + base_recon overrides.
    """
    fwd_cfg = OmegaConf.load(HYDRA_ROOT / "problem_trafos/fwd_trafo/mri3d.yaml")
    # mri3d.yaml already has mask_type='dataset'; apply varrecon_voxelrep override:
    OmegaConf.update(fwd_cfg, "sensitivitymaps_fillouter", True)
    # Remove 'name' key — get_fwd_trafo takes it as a positional arg
    _SKIP = {"name", "defaults"}
    fwd_kwargs = {k: v for k, v in OmegaConf.to_container(fwd_cfg).items() if k not in _SKIP}

    target_cfg = OmegaConf.load(HYDRA_ROOT / "problem_trafos/target_trafo/crop_mag.yaml")
    target_kwargs = {k: v for k, v in OmegaConf.to_container(target_cfg).items() if k not in _SKIP}

    prior_cfg = OmegaConf.load(HYDRA_ROOT / "problem_trafos/prior_trafo/crop_mag.yaml")
    # skmtea/base_recon.yaml overrides for prior_trafo:
    OmegaConf.update(prior_cfg, "scaling_factor", 2.0)
    OmegaConf.update(prior_cfg, "move_axis", [-1, 1])
    prior_kwargs = {k: v for k, v in OmegaConf.to_container(prior_cfg).items() if k not in _SKIP}

    sample_logger_cfg = OmegaConf.load(HYDRA_ROOT / "sample_logger/with_target.yaml")
    sl_kwargs = {k: v for k, v in OmegaConf.to_container(sample_logger_cfg).items()
                 if k not in _SKIP}

    return fwd_cfg.name, fwd_kwargs, target_cfg.name, target_kwargs, prior_cfg.name, prior_kwargs, sl_kwargs


# ---------------------------------------------------------------------------
# Main reconstruction loop
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_volumes",       type=int,   default=1)
    parser.add_argument("--vol_start",         type=int,   default=0)
    parser.add_argument("--device",            type=str,   default="cuda:0")
    parser.add_argument("--no_wandb",          action="store_true")
    parser.add_argument("--lmdb",              type=str,   default=None,
                        help="LMDB path override (QR or half-res). Defaults to VAL_LMDB. "
                             "QR:   /scratch/10846/armeet/datasets/ruibo/skmtea_val_128x128x80_4x_lmdb  "
                             "Half: /scratch/10846/armeet/datasets/ruibo/skmtea_val_256x256x160_4x_random_lmdb")
    parser.add_argument("--full_res",          action="store_true",
                        help="Full-res SR eval: load H5 vols via VolumeDataset + 2x Z interp "
                             "(overrides --lmdb). Matches Ruibo's full-res eval approach.")
    parser.add_argument("--data_con_weight",   type=float, default=None,
                        help="Override DATA_CON_WEIGHT (default 1.0). "
                             "Try 0.1 for full-res SR where observation is synthetic.")
    parser.add_argument("--crop_prior",        action="store_true",
                        help="Center-crop prior slabs to 320×320 before the score model. "
                             "Use with --full_res to keep the score model in its training "
                             "distribution (trained on 320×320 crops from full-res images).")
    parser.add_argument("--noise_target_psnr", type=float, default=None,
                        help="If set, add Gaussian noise to k-space so that "
                             "FBP PSNR ≈ this value (dB). E.g. --noise_target_psnr 10")
    parser.add_argument("--out_dir",           type=str,   default=None,
                        help="Absolute directory where SampleLoggerWithTarget will save "
                             "final_rec_<i>.pt / ground_truth_<i>.pt. If unset, saves go to cwd.")
    parser.add_argument("--save_final_sample", action="store_true",
                        help="Save final reconstruction tensor as final_rec_<i>.pt in --out_dir.")
    parser.add_argument("--save_ground_truth", action="store_true",
                        help="Save ground-truth tensor as ground_truth_<i>.pt in --out_dir.")
    parser.add_argument("--iterations",        type=int,   default=200,
                        help="Number of inference iterations per volume "
                             "(overrides outer_iterations_max and optimizer.iterations).")
    args = parser.parse_args()
    device = args.device
    if args.data_con_weight is not None:
        global DATA_CON_WEIGHT
        DATA_CON_WEIGHT = args.data_con_weight

    # ---- wandb ---------------------------------------------------------------
    # Honor WANDB_PROJECT and WANDB_NAME env vars (set per-job by sbatch).
    # Fall back to a safe default project if neither is set.
    wandb.init(project=os.environ.get("WANDB_PROJECT", "skmtea_recon"),
               name=os.environ.get("WANDB_NAME"),
               mode="disabled" if args.no_wandb else "online")

    # ---- Load trafo configs from YAML ----------------------------------------
    (fwd_name,    fwd_kwargs,
     target_name, target_kwargs,
     prior_name,  prior_kwargs,
     sl_kwargs) = _load_trafo_configs()

    fwd_trafo    = get_fwd_trafo(name=fwd_name,       **fwd_kwargs)
    target_trafo = get_target_trafo(name=target_name, **target_kwargs)
    prior_trafo  = get_prior_trafo(name=prior_name,   **prior_kwargs)
    if args.crop_prior:
        prior_trafo.center_crop_enabled = True
        prior_trafo.crop_size = (320, 320)
        logging.info("prior_trafo: center_crop_enabled=True  crop_size=(320, 320)")

    # ---- Score model + SDE ---------------------------------------------------
    score = load_score_model_direct(EMA_CKPT, TRAIN_CFG, device)
    score = ScoreWithIdentityGradWrapper(module=score)
    sde   = DDPM(beta_min=0.0001, beta_max=0.02, num_steps=1000)

    # ---- Slice methods -------------------------------------------------------
    slice_method_data_con  = get_slice_method(name="None")
    slice_method_prior_reg = get_slice_method(
        name="rnd_slicing",
        slice_budget=50,
        slab_thickness=5,
        volume_indices=[0, 1, 2],
        swapaxis=[False, True, True],
        slices_discrete=True,
        random_downsample_mesh=False,
        random_downsample_mesh_range=[0.1, 1.0],
        grid_aligned=None,
        slice_stride=0,
        slice_enabled=[True, True, True],
        average_slabs=[False, False, False],
        rnd_indices=[True, True, True],
        keep_dims=[False, False, False],
    )

    # ---- Dataset setup -------------------------------------------------------
    if args.full_res:
        # Full-res SR: VolumeDataset from raw H5 + 2× Z interpolation.
        # Mirrors Ruibo's full-res eval pipeline exactly.
        import pandas as _pd
        import importlib.util as _ilu
        from datasets.skmtea import VolumeDataset as _VDS, SenseModel as _SM
        # Load Gaussian3DMaskFunc from mri3d-local fastmri (avoids sys.modules conflict
        # with the installed fastmri package imported earlier in this file).
        _spec = _ilu.spec_from_file_location(
            "mri3d_sub", str(MRI3D_SRC / "fastmri" / "subsample.py"))
        _ksub = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_ksub)

        VAL_CSV  = MRI3D_SRC / "datasets" / "val.csv"
        fps      = [str(H5_BASE / r) for r in _pd.read_csv(VAL_CSV)["file_name"]
                    if (H5_BASE / r).exists()]
        mask_func = _ksub.Gaussian3DMaskFunc(accelerations=[4], stds_x=[128], stds_y=[40])
        vol_ds    = _VDS(fps, scale=1.0, mask_func=mask_func, echo=1)
        num_volumes = min(args.num_volumes, len(vol_ds) - args.vol_start)
        logging.info(f"Full-res H5: {len(vol_ds)} vols  ({num_volumes} to process)")
    else:
        # LMDB: QR (128×128×80) or half-res (256×256×160)
        lmdb_path = Path(args.lmdb) if args.lmdb else VAL_LMDB
        vol_ds    = LMDBVolumeDataset(root_dir=lmdb_path, norm_constant=3e7)
        num_volumes = min(args.num_volumes, len(vol_ds) - args.vol_start)
        logging.info(f"LMDB: {len(vol_ds)} vols at {lmdb_path}  ({num_volumes} to process)")

    # ---- Override sample-logger save flags from CLI --------------------------
    if args.save_final_sample:
        sl_kwargs["save_final_sample"] = True
    if args.save_ground_truth:
        sl_kwargs["save_ground_truth"] = True

    # ---- Chdir into out_dir so SampleLoggerWithTarget's bare-filename --------
    # torch.save() calls land in the per-vol folder, not the script root.
    # Done AFTER all relative-path loads (recon_workdir/, hydra/) so they don't break.
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(out_dir)
        logging.info(f"chdir to out_dir: {out_dir}")

    # ---- Sample logger -------------------------------------------------------
    sample_logger = MagSampleLogger(
        device=device,
        devices=[device],
        fwd_trafo=fwd_trafo,
        target_trafo=target_trafo,
        **sl_kwargs,
    )
    sample_logger.init_run(num_samples=num_volumes)

    if not args.full_res:
        loader    = DataLoader(vol_ds, batch_size=1, shuffle=False)
        lmdb_iter = iter(islice(loader, args.vol_start, args.vol_start + num_volumes))

    for i in tqdm(range(num_volumes), total=num_volumes, desc="volumes"):

        # ------------------------------------------------------------------
        # Data extraction — two paths converge to the same four variables:
        #   observation  : (C, X, Y, Z, 2)  real-as-complex  [on device]
        #   ground_truth : (X, Y, Z, 2)     real-as-complex  [on device]
        #   sens_maps_np : (X, Y, Z, C)     numpy complex64
        #   mask_binary  : (Y, Z, 1)        float32 tensor   [cpu]
        # ------------------------------------------------------------------
        if args.full_res:
            # ---- Full-res SR: H5 + 2× Z interpolation --------------------
            # Identical to Ruibo's full-res eval data prep.
            sample = vol_ds[args.vol_start + i]
            new_z  = sample.kspace.shape[-1] * 2          # 160 → 320

            maps_up   = _interp_z(sample.maps,         new_z)   # (C, X, Y, 2Z)
            target_up = _interp_z(sample.target,        new_z)   # (1, X, Y, 2Z)
            mask_up   = _interp_z(sample.mask.real.float(), new_z)  # (1, X, Y, 2Z)

            # SENSE recon of masked half-Z kspace → interp image → fwd SENSE
            sense     = _SM(sample.maps.unsqueeze(0))
            image_up  = _interp_z(sense.backward(sample.kspace.unsqueeze(0))[0, 0], new_z)
            sense_up  = _SM(maps_up.unsqueeze(0))
            kspace_up = sense_up.forward(image_up.unsqueeze(0).unsqueeze(0))[0]  # (C, X, Y, 2Z)

            # Re-apply mask before passing to fwd_trafo (kspace_up is non-zero
            # everywhere from SENSE forward; fwd_trafo expects zero at unsampled
            # positions for correct data-consistency loss).
            kspace_clean = (kspace_up * mask_up) / NORM_CONSTANT   # (C, X, Y, 2Z) complex
            target_norm  = target_up / NORM_CONSTANT

            dtype        = torch.get_default_dtype()
            observation  = torch.view_as_real(kspace_clean).to(dtype=dtype, device=device)
            ground_truth = torch.view_as_real(target_norm.squeeze(0)).to(dtype=dtype, device=device)

            sens_maps_np = maps_up.permute(1, 2, 3, 0).numpy()           # (X, Y, 2Z, C)
            mask_binary  = mask_up[0, 0, :, :].float().unsqueeze(-1)     # (Y, 2Z, 1)

            if i == 0:
                logging.info(f"[full-res] kspace_up {tuple(kspace_up.shape)}  "
                             f"obs |max|={kspace_clean.abs().max():.4f}  "
                             f"target |max|={target_norm.abs().max():.4f}")

        else:
            # ---- LMDB: QR (128×128×80) or half-res (256×256×160) ---------
            # kspace stored as m*(1+j)*k in LMDB — correct the mask bug.
            batch  = next(lmdb_iter)
            kspace = batch.kspace.squeeze(0)   # (C, X, Y, Z) complex
            maps   = batch.maps.squeeze(0)     # (C, X, Y, Z) complex
            target = batch.target.squeeze(0)   # (1, X, Y, Z) complex
            mask   = batch.mask.squeeze(0)     # (1, X, Y, Z) complex  m*(1+j)

            # Correct mask bug: m*(1+j)*k → m*k   [(1+j)*(1-j)/2 = 1]
            kspace_clean = kspace * torch.tensor((1 - 1j) / 2, dtype=torch.complex64)

            dtype        = torch.get_default_dtype()
            observation  = torch.view_as_real(kspace_clean).to(dtype=dtype, device=device)
            ground_truth = torch.view_as_real(target.squeeze(0)).to(dtype=dtype, device=device)

            sens_maps_np = maps.permute(1, 2, 3, 0).numpy()          # (X, Y, Z, C)
            mask_binary  = mask.real[0, 0, :, :].unsqueeze(-1)       # (Y, Z, 1)

            if i == 0:
                logging.info(f"kspace {tuple(kspace.shape)}  "
                             f"stored |max|={kspace.abs().max():.4f}  "
                             f"cleaned |max|={kspace_clean.abs().max():.4f}  "
                             f"target |max|={target.abs().max():.4f}")

        # ------------------------------------------------------------------
        # Common pipeline (identical for all three resolutions)
        # ------------------------------------------------------------------
        calib_params = {
            "sens_maps": sens_maps_np,
            "mask":      mask_binary.to(device),
        }

        fwd_trafo.calibrate(observation, calib_params)

        if args.noise_target_psnr is not None:
            sigma, noise = find_noise_sigma(observation, args.noise_target_psnr)
            observation  = observation + sigma * noise
            logging.info(f"[vol {i}] noise σ={sigma:.6f} added to k-space")

        with torch.no_grad():
            filtbackproj = fwd_trafo.trafo_adjoint(observation)

        rep_shape       = filtbackproj.shape    # (X, Y, Z, 2)
        base_mesh_shape = rep_shape[:-1]        # (X, Y, Z)

        mesh_cfg = OmegaConf.create({
            "matrix_size":        list(base_mesh_shape),
            "field_of_view":      [128.0, 160.0, 160.0],
            "max_coord":          1.0,
            "requires_coords":    False,
            "mesh_jitter_enable": False,
            "mesh_jitter_is_int": False,
            "mesh_jitter_bounds": None,
        })
        mesh_data_con = get_mesh(mesh_cfg, device=device)

        scaling_factor = (
            math.sqrt(float(np.prod(rep_shape)))
            / observation.detach().cpu().norm().item()
            * 1.0
        )
        if i == 0:
            logging.info(f"scaling_factor={scaling_factor:.6f}  "
                         f"obs_norm={observation.detach().cpu().norm():.2f}")

        sample_logger.init_sample_log(
            observation=observation,
            filtbackproj=filtbackproj,
            ground_truth=ground_truth,
            sample_nr=i,
            scaling_factor=scaling_factor,
            mesh=mesh_data_con,
        )

        initialise_with = filtbackproj.clone().to(device) * scaling_factor
        filtbackproj = None; torch.cuda.empty_cache()

        representation = FixedGridRepresentation(
            in_shape=tuple(base_mesh_shape),
            out_features=2,
            warm_start=initialise_with,
        )

        var_objective = WeightedDiffusionObjective(
            name="diffusion",
            reg_strength=0.01,
            adapt_reg_strength=True,
            steps_scaler=0.4,
            time_sampling_method="random",
            observation=observation * scaling_factor,
            mesh_data_con=mesh_data_con,
            mesh_data_reg=mesh_data_con,
            fwd_trafo=fwd_trafo,
            prior_trafo=prior_trafo,
            steps_data_con=[0],
            steps_data_reg=[0],
            slice_method_data_con=slice_method_data_con,
            slice_method_prior_reg=slice_method_prior_reg,
            outer_iterations_max=args.iterations,
            score=score,
            sde=sde,
        )

        cfg_fitting = OmegaConf.create({
            "use_filterbackproj_as_init": True,
            "use_l1wavelet_as_init":      False,
            "optimizer": {
                "lr":                           2.0,
                "iterations":                   args.iterations,
                "clip_grad_max_norm":           None,
                "gradient_acc_steps_data_con":  [0],
                "gradient_acc_steps_prior_reg": [0],
                "skip_iterations":              0,
            },
            "lr_scheduler": {
                "name":     "ReduceLROnPlateau",
                "mode":     "min",
                "factor":   0.5,
                "patience": 20,
                "verbose":  True,
            },
            "warmstart_fit": None,
        })

        final_representation = fit(
            representation=representation,
            var_objective=var_objective,
            cfg_fitting=cfg_fitting,
            sample_logger=sample_logger,
        )

        sample_logger.close_sample_log(representation=final_representation)

    sample_logger.close_run()
    wandb.finish()


if __name__ == "__main__":
    main()
