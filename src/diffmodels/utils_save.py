"""Save-side utilities for the diffusion trainer.

Two responsibilities, both invoked once per `save_model_every_n_epoch` epoch:

  1. Per-epoch artifacts (existing) -- model_{epoch}.pt and ema_model_{epoch}.pt.
     Downstream eval / recon scripts depend on this naming, so it is preserved.

  2. NEW -- resume_latest.pt + inline-gen snapshots, both gated by the
     `optim_kwargs` config:
       * resume_latest.pt is a single overwritten file carrying every piece
         of state needed to seamlessly continue training across SLURM 2-day
         boundaries (model, EMA, optimizer, epoch, grad_step, RNG).
       * inline_gen produces evolution.gif + grid.png from a single noise
         sample drawn through the EMA model -- a sanity check the diffusion
         prior is learning something coherent without waiting until the run
         finishes.

Both new behaviors are opt-in via the optim_kwargs config (see
hydra/diffmodels/base.yaml). With defaults absent, save_model behaves
exactly as before -- safe for SKM-TEA training and the reconstruction
scripts that import this module.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import torch

from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels.ema import ExponentialMovingAverage


def save_model(
    score: UNetModel,
    epoch: int,
    optim_kwargs: Dict,
    ema: Optional[ExponentialMovingAverage] = None,
    sde: Optional[Any] = None,
    device: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_step: int = 0,
) -> None:

    model_filename = (
        'model.pt' if epoch == optim_kwargs['epochs'] - 1
        else f'model_{epoch}.pt'
    )
    torch.save(score.state_dict(), model_filename)
    if ema is not None:
        ema_filename = (
            'ema_model.pt' if epoch == optim_kwargs['epochs'] - 1
            else f'ema_model_{epoch}.pt'
        )
        torch.save(ema.state_dict(), ema_filename)

    # ------------------------------------------------------------------
    # New: resume artifact (single file, overwritten each save)
    # ------------------------------------------------------------------
    save_resume = True
    try:
        save_resume = bool(optim_kwargs.get("save_resume_state", True))
    except Exception:
        save_resume = True
    if save_resume:
        try:
            save_resume_state(
                score=score, ema=ema, optimizer=optimizer,
                epoch=epoch, grad_step=grad_step,
                path="resume_latest.pt",
            )
        except Exception as e:
            logging.warning(f"save_resume_state failed at epoch {epoch}: {e}")

    # ------------------------------------------------------------------
    # New: inline unconditional sample (evolution.gif + grid.png)
    # ------------------------------------------------------------------
    try:
        inline_cfg_raw = optim_kwargs.get("inline_gen", None)
    except Exception:
        inline_cfg_raw = None
    inline_cfg = inline_cfg_raw or {}

    def _icfg(key, default):
        # Support both dict and OmegaConf DictConfig access patterns.
        try:
            v = inline_cfg.get(key, default)
        except Exception:
            v = getattr(inline_cfg, key, default)
        return v

    if (
        _icfg("enabled", False)
        and ema is not None
        and sde is not None
        and device is not None
    ):
        try:
            from src.diffmodels.inline_generation import generate_inline_snapshot

            in_channels = getattr(score, "in_channels", None)
            if in_channels is None:
                # UNetModel exposes the conv layer width directly; fall back
                # to the input conv if `in_channels` attr is missing.
                in_channels = score.input_blocks[0][0].in_channels

            size = int(_icfg("size", 240))
            num_steps = int(_icfg("num_steps", 50))
            save_every = int(_icfg("save_every", 5))
            shape = (1, in_channels, size, size)
            out_dir = Path("inline_gen") / f"epoch_{epoch}"

            ema.store(score.parameters())
            ema.copy_to(score.parameters())
            try:
                generate_inline_snapshot(
                    score=score, sde=sde, out_dir=out_dir, shape=shape,
                    num_steps=num_steps, device=device, save_every=save_every,
                )
            finally:
                ema.restore(score.parameters())
                # Sampler ran in eval() mode; the trainer's epoch loop
                # re-enters train() at the top of the next epoch, so no
                # explicit train() restore is needed here.

            # Best-effort wandb logging.
            try:
                import wandb
                if wandb.run is not None:
                    artifacts = {}
                    gif_path = out_dir / "evolution.gif"
                    grid_path = out_dir / "grid.png"
                    if grid_path.exists():
                        artifacts["inline_gen/grid"] = wandb.Image(str(grid_path))
                    if gif_path.exists():
                        artifacts["inline_gen/evolution"] = wandb.Video(
                            str(gif_path), fps=12, format="gif"
                        )
                    if artifacts:
                        artifacts["epoch"] = epoch + 1
                        wandb.log(artifacts)
            except Exception as e:
                logging.warning(f"inline_gen wandb upload failed at epoch {epoch}: {e}")
        except Exception as e:
            logging.warning(f"inline_gen failed at epoch {epoch}: {e}")


# ----------------------------------------------------------------------
# Resume helpers (used by trainer at startup, and by save_model above)
# ----------------------------------------------------------------------


def save_resume_state(
    score: UNetModel,
    ema: Optional[ExponentialMovingAverage],
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    grad_step: int,
    path: str,
) -> None:
    """Persist every piece of state needed to continue training.

    Layout::

        {
            "model":     score.state_dict(),
            "ema":       ema.state_dict() | None,
            "optimizer": optimizer.state_dict() | None,
            "epoch":     int,           # epoch JUST finished
            "grad_step": int,           # cumulative optimizer.step() count
            "torch_rng": torch.ByteTensor,
        }

    `load_resume_state` returns `epoch + 1` as the next start epoch.
    """
    state = {
        "model":     score.state_dict(),
        "ema":       ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch":     int(epoch),
        "grad_step": int(grad_step),
        "torch_rng": torch.get_rng_state(),
    }
    tmp_path = str(path) + ".tmp"
    torch.save(state, tmp_path)
    # Atomic-ish swap so a partial write never corrupts the live file.
    import os
    os.replace(tmp_path, str(path))


def load_resume_state(
    path: str,
    score: UNetModel,
    ema: Optional[ExponentialMovingAverage],
    optimizer: Optional[torch.optim.Optimizer],
    device: Any,
) -> tuple:
    """Restore (model, ema, optimizer, RNG) in place; return (start_epoch, grad_step)."""
    state = torch.load(path, map_location=device)

    score.load_state_dict(state["model"])

    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
        # Shadow params may have been pickled on the wrong device.
        ema.shadow_params = [p.to(device) for p in ema.shadow_params]

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])

    rng = state.get("torch_rng", None)
    if rng is not None:
        torch.set_rng_state(rng.cpu() if hasattr(rng, "cpu") else rng)

    start_epoch = int(state["epoch"]) + 1
    grad_step = int(state.get("grad_step", 0))
    logging.info(
        f"load_resume_state: restored from {path} -> start_epoch={start_epoch}, "
        f"grad_step={grad_step}"
    )
    return start_epoch, grad_step
