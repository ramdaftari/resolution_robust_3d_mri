#!/usr/bin/env python3
"""Read-only volume count for the 4 diffusion val LMDBs. Safe vs running jobs (lock=False)."""
import lmdb

PATHS = [
    ("half_4x", "/scratch/10846/armeet/datasets/skmtea_val_0.5first2_4x_lmdb"),
    ("half_8x", "/scratch/10846/armeet/datasets/ruibo/skmtea_val_256x256x160_8x_lmdb"),
    ("qr_4x",   "/scratch/10846/armeet/datasets/ruibo/skmtea_val_128x128x80_4x_lmdb"),
    ("qr_8x",   "/scratch/10846/armeet/datasets/ruibo/skmtea_val_128x128x80_8x_lmdb"),
]
for name, root in PATHS:
    try:
        env = lmdb.open(root + "/shapes", readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            keys = sorted((k.decode() for k, _ in txn.cursor()), key=int)
        env.close()
        print(f"{name:8s} {len(keys):3d} vols  first={keys[0]!r}  last={keys[-1]!r}  ({root})")
    except Exception as e:
        print(f"{name:8s} ERROR {e}  ({root})")
