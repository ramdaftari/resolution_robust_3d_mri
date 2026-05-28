"""Unconditional sampling from the trained EMA score model.

The score model is the 2D DDPM UNet trained for the resolution-robust 3D MRI
baseline. It operates on complex slices as 2-channel real tensors
([N, 2, H, W]) with epsilon-prediction.

Outputs (under --out_dir, default ./uncon_gen/runs/<sampler>_<num_steps>/):
  - evolution.gif   animation of all N samples (in a grid) across reverse steps
  - grid.png        one slice at 4 equally-spaced timesteps, each labelled with t

Run from the baseline root (so `src.*` resolves) inside the uv venv:

  cd src/baselines/resolution_robust_3d_mri
  source .venv/bin/activate
  python uncon_gen/generate_unconditional.py \
      --ckpt recon_workdir/ema_model_110.pt \
      --num_samples 4 --size 320 --sampler ddim --num_steps 200 \
      --save_every 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Make `src.*` resolve regardless of cwd. Baseline root is the parent of this
# script's directory (uncon_gen/).
_BASELINE_ROOT = Path(__file__).resolve().parent.parent
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.diffmodels.diffmodels_resolver import create_model  # noqa: E402
from src.diffmodels.ema import ExponentialMovingAverage  # noqa: E402
from src.diffmodels.sde import DDPM as DDPMSDE  # noqa: E402
# Sampling + visualization helpers now live in a shared module so the
# trainer can do inline generation without re-implementing them.
from src.diffmodels.inline_generation import (  # noqa: E402
    sample_ddim as _sample_ddim_shared,
    save_outputs as _save_outputs_shared,
)


def _load_arch_cfg(ckpt_path: Path) -> dict:
    hydra_cfg = ckpt_path.parent / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        raise FileNotFoundError(
            f"Could not find hydra config next to checkpoint at {hydra_cfg}."
        )
    cfg = OmegaConf.load(hydra_cfg)
    arch = cfg.diffmodels.arch if "diffmodels" in cfg else cfg.arch
    arch = dict(arch)
    if "name" not in arch:
        arch = {"name": "dense", "params": arch}
    return arch


def load_score_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    arch_cfg = _load_arch_cfg(ckpt_path)
    score = create_model(**arch_cfg, arch_cfg=None).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "shadow_params" in state:
        ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
        ema.load_state_dict(state)
        ema.copy_to(score.parameters())
        logging.info("Loaded EMA shadow params into score model.")
    else:
        clean = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        score.load_state_dict(clean)
        logging.info("Loaded raw model state dict into score model.")

    score.eval()
    return score


def sample_ddim(score, sde, shape, num_steps, eta, device, save_every):
    """Thin wrapper preserving the original (eta-before-device) arg order.

    Implementation lives in `src.diffmodels.inline_generation.sample_ddim`,
    which the trainer reuses for inline checkpoints.
    """
    return _sample_ddim_shared(
        score=score, sde=sde, shape=shape,
        num_steps=num_steps, device=device,
        save_every=save_every, eta=eta,
    )


@torch.no_grad()
def sample_ddpm(score, sde, shape, device, save_every):
    x = sde.prior_sampling(shape).to(device)
    ones = torch.ones(1, device=device)

    timesteps = list(range(sde.num_steps - 1, -1, -1))
    pairs = list(zip(timesteps, timesteps[1:] + [-1]))
    traj, traj_idx = [], []
    for step_i, (cur, prev) in enumerate(tqdm(pairs, desc="DDPM")):
        t_cur = ones * cur
        t_prev = ones * prev

        eps = score(x, t_cur)
        if eps.size(1) == 2 * x.size(1):
            eps = eps[:, : x.size(1)]

        mean_cur = sde.marginal_prob_mean(t=t_cur)[:, None, None, None]
        mean_prev = sde.marginal_prob_mean(t=t_prev)[:, None, None, None]
        alpha_t = mean_cur.pow(2) / mean_prev.pow(2)
        alpha_bar_t = mean_cur.pow(2)

        sqrt_beta = (
            (1 - mean_prev.pow(2)) / (1 - mean_cur.pow(2))
        ).sqrt() * (1 - alpha_t).sqrt()
        xmean = (x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_t)
        if prev >= 0:
            x = xmean + sqrt_beta * torch.randn_like(x)
        else:
            x = xmean

        if save_every > 0 and (step_i % save_every == 0 or prev == -1):
            xhat0 = sde.tweedy(x=x, t=t_cur, score_xt=eps)
            traj.append(xhat0.detach().cpu())
            traj_idx.append(cur)

    return x, traj, traj_idx


def save_outputs(final, traj, traj_idx, out_dir: Path,
                 grid_sample_idx: int = 0, grid_cols: int = 4) -> None:
    """Thin wrapper -- delegates to src.diffmodels.inline_generation.save_outputs.

    The `final` arg is kept for backward compatibility with the original
    CLI signature but is unused: the trajectory's last frame already carries
    the final state.
    """
    return _save_outputs_shared(
        traj=traj, traj_idx=traj_idx, out_dir=out_dir,
        grid_cols=grid_cols, grid_sample_idx=grid_sample_idx,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path,
                        default=Path("recon_workdir/ema_model_110.pt"))
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Reverse steps for DDIM (ignored for DDPM).")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--beta_min", type=float, default=1e-4)
    parser.add_argument("--beta_max", type=float, default=0.02)
    parser.add_argument("--sde_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Capture x0-estimate every K reverse steps for the video. 0 disables.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grid_sample_idx", type=int, default=0,
                        help="Which of the N samples to display in grid.png.")
    parser.add_argument("--grid_cols", type=int, default=4,
                        help="Number of equally-spaced timesteps in grid.png.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Device: {device}")

    out_dir = args.out_dir or (
        Path(__file__).resolve().parent
        / "runs"
        / f"{args.sampler}_steps{args.num_steps}_n{args.num_samples}_sz{args.size}_seed{args.seed}"
    )
    logging.info(f"Output dir: {out_dir}")

    score = load_score_model(args.ckpt, device=device)
    sde = DDPMSDE(beta_min=args.beta_min, beta_max=args.beta_max, num_steps=args.sde_steps)

    shape = (args.num_samples, args.channels, args.size, args.size)
    logging.info(f"Sampling shape={shape} sampler={args.sampler}")

    if args.sampler == "ddim":
        final, traj, traj_idx = sample_ddim(
            score, sde, shape, args.num_steps, args.eta, device, args.save_every,
        )
    else:
        final, traj, traj_idx = sample_ddpm(
            score, sde, shape, device, args.save_every,
        )

    save_outputs(final, traj, traj_idx, out_dir,
                 grid_sample_idx=args.grid_sample_idx,
                 grid_cols=args.grid_cols)
    logging.info(f"Wrote evolution.gif + grid.png to {out_dir}/")


if __name__ == "__main__":
    main()
