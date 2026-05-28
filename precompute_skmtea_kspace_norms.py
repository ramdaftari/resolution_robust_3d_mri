"""Precompute per-volume k-space Frobenius norms for SKM-TEA.

The norm is consumed by `scale_target_by_kspacenorm` in the FastMRI2D trafo.
Computing it lazily inside `SkmteaSliceDataset.__getitem__` requires reading
~5.4 GB of HDF5 per volume on the first slice access — across an 86-volume
training set that's ~460 GB of redundant I/O at the start of every run.

This script reads each `*.h5` once, computes
    norm[echo] = || kspace[:, :, :, echo_idx, :] ||_2
for both echoes, and writes the result to a small JSON sidecar:

    {
      "MTR_001.h5": {"1": 5.376e10, "2": 4.81e10},
      "MTR_002.h5": {...},
      ...
    }

`SkmteaSliceDataset` loads this JSON at construction time and uses the
precomputed value, falling back to the lazy read if a volume is missing.

Usage:
    python precompute_skmtea_kspace_norms.py \
        --data_root /scratch/10846/armeet/datasets/skmtea/files_recon_calib-24 \
        --output    /scratch/10846/armeet/datasets/skmtea/kspace_norms.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def compute_norms_for_file(fp: Path, num_echoes: int = 2) -> dict[str, float]:
    """Return {echo_str: ||kspace[..., echo_idx, :]||_2} for `fp`."""
    out: dict[str, float] = {}
    with h5py.File(fp, "r") as hf:
        ks = hf["kspace"]                                                  # (X, Y, Z, echo, C)
        if ks.ndim != 5:
            raise RuntimeError(f"{fp}: expected 5D kspace, got shape {ks.shape}")
        max_echo = min(num_echoes, ks.shape[3])
        for echo in range(1, max_echo + 1):
            arr = np.asarray(ks[:, :, :, echo - 1, :])                     # one full echo
            out[str(echo)] = float(np.linalg.norm(arr))
            del arr
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Directory containing SKM-TEA *.h5 volumes.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path.")
    parser.add_argument("--num_echoes", type=int, default=2,
                        help="How many echoes to precompute per file (default 2).")
    parser.add_argument("--glob", type=str, default="*.h5",
                        help="Filename pattern to match (default *.h5).")
    parser.add_argument("--resume", action="store_true",
                        help="If set and --output exists, keep its existing entries "
                             "and only compute missing files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    files = sorted(args.data_root.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matching {args.glob} under {args.data_root}")

    existing: dict[str, dict[str, float]] = {}
    if args.resume and args.output.exists():
        with open(args.output, "r") as f:
            existing = json.load(f)
        logging.info(f"Resuming with {len(existing)} entries already in {args.output}")

    todo = [fp for fp in files if fp.name not in existing]
    logging.info(f"Computing kspace norms for {len(todo)}/{len(files)} files "
                 f"under {args.data_root}")

    for fp in tqdm(todo, desc="kspace norms"):
        try:
            existing[fp.name] = compute_norms_for_file(fp, num_echoes=args.num_echoes)
        except Exception as e:
            logging.error(f"{fp.name}: failed — {e}")
            continue
        # Flush after each file so a partial run is still usable.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)

    logging.info(f"Wrote {len(existing)} entries to {args.output}")


if __name__ == "__main__":
    main()
