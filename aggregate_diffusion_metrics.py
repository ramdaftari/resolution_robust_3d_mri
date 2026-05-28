#!/usr/bin/env python3
"""
Aggregate per-volume run.log SUMMARY tables across all 33 vol_*/run.log files
in one or more diffusion-baseline output dirs.

Usage:
    python aggregate_diffusion_metrics.py <job_dir> [<job_dir> ...]

Example:
    python aggregate_diffusion_metrics.py \
        $SCRATCH/peterwg/diffusion_baseline/qr_4x \
        $SCRATCH/peterwg/diffusion_baseline/qr_8x

Each <job_dir> is expected to contain vol_0/run.log ... vol_32/run.log.
Each run.log holds one SUMMARY table (n=1 per file since each python invocation
processes one volume), with rows for steps 10/25/50/75/100/150/200.
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

# Matches a SUMMARY data row inside run.log, regardless of any tee timestamp prefix:
#   "...  INFO     10     27.1265    0.5715    0.103708     1"
ROW_RE = re.compile(
    r"INFO\s+(?P<step>\d+)\s+"
    r"(?P<psnr>-?\d+\.\d+)\s+"
    r"(?P<ssim>-?\d+\.\d+)\s+"
    r"(?P<nmse>-?\d+\.\d+)\s+"
    r"\d+\s*$"
)
EXPECTED_STEPS = (10, 25, 50, 75, 100, 150, 200)


def parse_log(path: Path) -> dict[int, tuple[float, float, float]]:
    """Return {step: (psnr, ssim, nmse)} for the SUMMARY table in run.log."""
    found: dict[int, tuple[float, float, float]] = {}
    with path.open() as f:
        for line in f:
            m = ROW_RE.search(line)
            if not m:
                continue
            step = int(m["step"])
            if step in EXPECTED_STEPS:
                found[step] = (float(m["psnr"]), float(m["ssim"]), float(m["nmse"]))
    return found


def aggregate(job_dir: Path, label_final_as: int | None = None) -> None:
    logs = sorted(job_dir.glob("vol_*/run.log"), key=lambda p: int(p.parent.name.split("_")[1]))
    if not logs:
        print(f"  no vol_*/run.log under {job_dir}")
        return

    per_step: dict[int, list[tuple[float, float, float]]] = {s: [] for s in EXPECTED_STEPS}
    missing: list[str] = []

    for log in logs:
        rows = parse_log(log)
        if not rows:
            missing.append(log.parent.name)
            continue
        for step in EXPECTED_STEPS:
            if step in rows:
                per_step[step].append(rows[step])

    populated = [s for s in EXPECTED_STEPS if per_step[s]]
    # Auto-detect single-step runs: only the final 200-row populated → relabel as step 1
    # (close_sample_log always appends to ckpt_psnrs[200], regardless of actual iter count).
    if label_final_as is None and populated == [200] and "single_step" in str(job_dir):
        label_final_as = 1

    n_complete = (min(len(per_step[s]) for s in populated) if populated else 0)
    print(f"\n=== {job_dir} ===")
    print(f"  vol logs found: {len(logs)}   complete (all populated steps): {n_complete}")
    if missing:
        print(f"  no SUMMARY found in: {', '.join(missing)}")
    print(f"  {'step':>5}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'NMSE':>10}  {'n':>4}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*4}")
    for step in populated:
        rows = per_step[step]
        psnrs = [r[0] for r in rows]
        ssims = [r[1] for r in rows]
        nmses = [r[2] for r in rows]
        display_step = label_final_as if (step == 200 and label_final_as is not None) else step
        print(f"  {display_step:>5}  {statistics.mean(psnrs):>10.4f}  "
              f"{statistics.mean(ssims):>8.4f}  "
              f"{statistics.mean(nmses):>10.6f}  {len(rows):>4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dirs", nargs="+", type=Path)
    ap.add_argument("--label-final-as", type=int, default=None,
                    help="Relabel the final '200' row as this step number "
                         "(use --label-final-as 1 for single-step runs).")
    args = ap.parse_args()

    for d in args.job_dirs:
        if not d.exists():
            print(f"\n=== {d} ===\n  MISSING: directory does not exist", file=sys.stderr)
            continue
        aggregate(d, label_final_as=args.label_final_as)


if __name__ == "__main__":
    main()
