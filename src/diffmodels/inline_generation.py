"""Inline unconditional sampling used during training.

After every EMA checkpoint, the trainer generates a single 2-channel image
from pure noise using the EMA score model and saves two artifacts:

    evolution.gif  -- animation of the single sample across reverse steps
    grid.png       -- the same sample at 4 equally-spaced timesteps

Functions here are the runtime-friendly subset of
`uncon_gen/generate_unconditional.py` (same algorithm, no argparse/file-loading
wrapping). The standalone uncon_gen script re-exports these so its behavior
is identical.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.diffmodels.sde import SDE


@torch.no_grad()
def sample_ddim(
    score: torch.nn.Module,
    sde: SDE,
    shape: Tuple[int, ...],
    num_steps: int,
    device: torch.device,
    save_every: int,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """Returns (final_x [N,C,H,W], trajectory list, step_indices list).

    Trajectory entries are Tweedie x0-estimates (cleaner video than raw x_t).
    """
    assert sde.num_steps >= num_steps, (
        f"sde.num_steps={sde.num_steps} < num_steps={num_steps}"
    )
    skip = sde.num_steps // num_steps
    boundaries = list(range(sde.num_steps - 1, -1, -skip)) + [-1]
    time_pairs = list(zip(boundaries[:-1], boundaries[1:]))

    x = sde.prior_sampling(shape).to(device)
    ones = torch.ones(1, device=device)

    traj: List[torch.Tensor] = []
    traj_idx: List[int] = []
    for step_i, (cur, prev) in enumerate(time_pairs):
        t_cur = ones * cur
        t_prev = ones * prev

        eps = score(x, t_cur)
        if eps.size(1) == 2 * x.size(1):
            # learn_sigma paths emit eps and sigma stacked on channels.
            eps = eps[:, : x.size(1)]

        xhat0 = sde.tweedy(x=x, t=t_cur, score_xt=eps)
        mean_prev = sde.marginal_prob_mean(t=t_prev)[:, None, None, None]
        mean_cur = sde.marginal_prob_mean(t=t_cur)[:, None, None, None]

        sqrt_beta = (
            (1 - mean_prev.pow(2)) / (1 - mean_cur.pow(2))
        ).sqrt() * (1 - mean_cur.pow(2) / mean_prev.pow(2)).sqrt()
        sqrt_beta = torch.nan_to_num(sqrt_beta, nan=0.0)

        deterministic = torch.sqrt(
            torch.clamp(1 - mean_prev.pow(2) - (sqrt_beta * eta).pow(2), min=0.0)
        ) * eps
        noise = eta * sqrt_beta * torch.randn_like(x) if prev >= 0 else 0.0
        x = xhat0 * mean_prev + deterministic + noise

        if save_every > 0 and (step_i % save_every == 0 or prev == -1):
            traj.append(xhat0.detach().cpu())
            traj_idx.append(cur)

    return x, traj, traj_idx


def _magnitude(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr[..., 0, :, :] ** 2 + arr[..., 1, :, :] ** 2)


def save_outputs(
    traj: List[torch.Tensor],
    traj_idx: List[int],
    out_dir: Path,
    grid_cols: int = 4,
    grid_sample_idx: int = 0,
) -> None:
    """Write evolution.gif and grid.png from a (possibly multi-sample) trajectory.

    Each entry of `traj` has shape [N, C, H, W]. The animation tiles all N
    samples in a grid; the still grid (`grid.png`) shows sample
    `grid_sample_idx` at `grid_cols` equally-spaced reverse-step indices.
    With N=1 the animation degenerates to a single tile.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not traj:
        logging.warning("inline_generation.save_outputs: empty trajectory, nothing to save.")
        return

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import animation

    traj_np = torch.stack(traj, dim=0).numpy()      # [T, N, C, H, W]
    if traj_np.shape[2] >= 2:
        traj_mag = _magnitude(traj_np)              # [T, N, H, W]
    else:
        traj_mag = np.abs(traj_np[:, :, 0])         # [T, N, H, W]
    T, n = traj_mag.shape[0], traj_mag.shape[1]

    # Per-sample windowing keyed to that sample's final frame -- stable
    # contrast across the animation.
    per_sample_norm = []
    for i in range(n):
        lo = float(np.percentile(traj_mag[-1, i], 1.0))
        hi = float(np.percentile(traj_mag[-1, i], 99.5))
        per_sample_norm.append((lo, max(hi - lo, 1e-8)))

    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 0.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    ims = []
    for i in range(n):
        lo, rng = per_sample_norm[i]
        frame0 = np.clip((traj_mag[0, i] - lo) / rng, 0.0, 1.0)
        ims.append(axes[i // cols, i % cols].imshow(frame0, cmap="gray", vmin=0, vmax=1))
    suptitle = fig.suptitle(f"reverse step  t = {traj_idx[0]}")
    fig.tight_layout()

    def update(k):
        for i in range(n):
            lo, rng = per_sample_norm[i]
            ims[i].set_data(np.clip((traj_mag[k, i] - lo) / rng, 0.0, 1.0))
        suptitle.set_text(f"reverse step  t = {traj_idx[k]}")
        return (*ims, suptitle)

    anim = animation.FuncAnimation(fig, update, frames=T, interval=80, blit=False)
    try:
        anim.save(out_dir / "evolution.gif", writer=animation.PillowWriter(fps=12))
    except Exception as e:
        logging.warning(f"inline_generation: GIF save failed: {e}")
    plt.close(fig)

    # grid.png: one sample at `grid_cols` equally-spaced timesteps.
    s = max(0, min(grid_sample_idx, n - 1))
    col_idx = np.linspace(0, T - 1, grid_cols).round().astype(int)
    lo, rng = per_sample_norm[s]
    fig, axes = plt.subplots(1, grid_cols, figsize=(3.0 * grid_cols, 3.2), squeeze=False)
    for c, k in enumerate(col_idx):
        a = axes[0, c]
        a.imshow(np.clip((traj_mag[k, s] - lo) / rng, 0.0, 1.0), cmap="gray", vmin=0, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"t = {traj_idx[k]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_inline_snapshot(
    score: torch.nn.Module,
    sde: SDE,
    out_dir: Path,
    shape: Tuple[int, ...],
    num_steps: int,
    device: torch.device,
    save_every: int = 5,
) -> None:
    """End-to-end: sample once with DDIM, write evolution.gif + grid.png.

    Caller swaps EMA params into `score` before this and restores after.
    """
    score.eval()
    _, traj, traj_idx = sample_ddim(
        score=score,
        sde=sde,
        shape=shape,
        num_steps=num_steps,
        device=device,
        save_every=save_every,
        eta=0.0,
    )
    save_outputs(traj=traj, traj_idx=traj_idx, out_dir=out_dir, grid_cols=4)
    logging.info(f"inline_generation: wrote evolution.gif + grid.png to {out_dir}/")
