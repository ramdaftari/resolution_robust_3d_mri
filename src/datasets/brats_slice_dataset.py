"""
BratsSliceDataset
-----------------
Serves 2D slices from BraTS LMDB volumes in the format expected by
mri2d_train_calc_mvue / FastMRI2DDataTransform — same per-slice contract
as SkmteaSliceDataset.

LMDB layout (from scripts/convert_brats_lmdb.py):
    kspace : (C, X, Y, Z)   complex64   -- FULL 3D k-space (np.fft.fftn, no norm)
    maps   : (C, X, Y, Z)   complex64   -- precomputed sensitivity maps
    target : (1, X, Y, Z)   complex64   -- magnitude-only image, in [0, 1]
    masks  : (1, X, Y, Z)   complex64   -- undersampling mask (unused here)
    shapes : JSON dict with the four shapes above

Difference vs SKM-TEA HDF5:
  * SKM-TEA `hf["kspace"]` is already 1D-IFFT'd along the readout axis (X);
    each X-index is therefore a 2D k-space slice in (Y, Z).
  * BraTS LMDB `kspace` is full 3D k-space — no axis is in image space yet.
  * To match SkmteaSliceDataset's contract, we apply a 1D IFFT along the
    readout axis once per volume and slice the result.

Normalization convention (matches fastMRI's ExtSliceDataset
exactly so the mri2d_train_calc_mvue trafo formula
    target_torch * sqrt(prod(target_vol_shape)) / kspace_vol_norm * scale
produces comparable scales across BraTS / SKM-TEA / fastMRI):
  * `kspace_vol_norm` = L2 norm of the *raw 3D k-space*, BEFORE the readout
    IFFT — same convention as fastmri_slice_dataset.py:262
    (`np.linalg.norm(hf["kspace"][()])`). Computing norm pre- vs
    post-IFFT only matches when the IFFT is orthonormal (Parseval);
    BraTS uses np.fft.ifft with default norm="backward", so we
    explicitly take the norm on the raw array to avoid the sqrt(N_readout)
    offset.
  * No /norm_constant pre-division (LMDBVolumeDataset's behavior is for
    the volume KNO/INO path; the 2D diffusion path uses scale_target_by_kspacenorm).

Output per sample: SliceDatasetSample(kspace, target, attrs) where
    kspace : real tensor shape (C, Y, Z, 2)
    target : real tensor shape (Y, Z, 2) — view_as_real of the precomputed
             complex MVUE-equivalent target from the LMDB (magnitude image
             in [0,1] cast to complex64 with imag=0). The BraTS LMDB stores
             ALREADY-MASKED k-space (see convert_brats_lmdb.py:70), so the
             trafo cannot recompute the MVUE via target_type='fullysampled_rec'
             — doing so would yield a 4x zero-filled reconstruction instead
             of a clean image. We therefore use target_type='mvue' (configured
             in hydra/exps/brats/base_training.yaml) which passes our target
             through to scaling+interpolation. The view_as_real layout matches
             what ifft2c emits on the SKM-TEA path so downstream code is
             identical from this point.
    attrs  : "sens_maps"        (C, Y, Z) complex numpy
             "kspace_vol_norm"  float
             "target_vol_shape" (X, Y, Z) tuple
             "num_coils", "num_slices", "kspace_shape"
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import lmdb
import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.fastmri_slice_dataset import SliceDatasetSample


# Module-level cache of lmdb Environment objects keyed by absolute sub-DB path.
# Reason: get_brats_dataset constructs THREE BratsSliceDataset instances
# (readout_axis = 0, 1, 2) over the same on-disk LMDB to realize the paper's
# three-plane training protocol. lmdb-py guards against opening the same
# Environment twice in a single process and raises
#   lmdb.Error: The environment '...' is already open in this process.
# Sharing the env handles across the three instances avoids the duplicate
# open while keeping each instance's slicing logic per-axis. Envs are opened
# read-only with lock=False so cross-worker / cross-instance reads are safe.
_LMDB_ENV_CACHE: Dict[str, "lmdb.Environment"] = {}


def _open_lmdb(path: Path) -> "lmdb.Environment":
    key = str(path.resolve())
    env = _LMDB_ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(key, readonly=True, lock=False, readahead=False)
        _LMDB_ENV_CACHE[key] = env
    return env


class BratsSliceDataset(BaseDataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        readout_axis: int = 0,
        return_sensmaps: bool = True,
        transform: Optional[Callable] = None,
        keys: Optional[Sequence[str]] = None,
        volume_cache_size: int = 1,
        **kwargs,
    ):
        """
        Args:
            root_dir: LMDB root containing kspace/, maps/, target/, masks/, shapes/.
            readout_axis: spatial axis (in the (X, Y, Z) volume layout) along
                which to apply the 1D IFFT and then slice.
                BraTS NIfTI convention -> 0 makes X the slice axis.
            return_sensmaps: include per-slice sens maps in attrs.
            transform: applied per sample after slicing (FastMRI2DDataTransform).
            keys: optional explicit list of LMDB keys; defaults to all keys in
                the shapes db, sorted by int value.
            volume_cache_size: how many post-IFFT volumes to keep cached in
                memory. 1 is enough since DataLoader workers shuffle slices
                across volumes; bump only if profiling says re-reads dominate.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.readout_axis = int(readout_axis)
        if self.readout_axis not in (0, 1, 2):
            raise ValueError(f"readout_axis must be 0, 1, or 2; got {readout_axis}")
        self.return_sensmaps = bool(return_sensmaps)
        self.transform = transform
        self.volume_cache_size = max(1, int(volume_cache_size))

        # Use the module-level env cache so three-plane construction
        # (readout_axis=0/1/2 over the same LMDB) doesn't trigger lmdb-py's
        # "already open in this process" guard.
        self.env_kspace = _open_lmdb(self.root_dir / "kspace")
        self.env_maps   = _open_lmdb(self.root_dir / "maps")
        self.env_target = _open_lmdb(self.root_dir / "target")
        self.env_shapes = _open_lmdb(self.root_dir / "shapes")

        if keys is None:
            with self.env_shapes.begin() as txn:
                keys = sorted([k.decode() for k, _ in txn.cursor()],
                              key=lambda x: int(x))
        self._volume_keys: List[str] = list(keys)

        # Build flat (vol_key, slice_idx, vol_meta) index up front.
        self.raw_samples: List[Tuple[str, int, Dict[str, Any]]] = []
        for vk in self._volume_keys:
            shp = self._read_shapes(vk)
            kspace_shape = tuple(shp["kspace"])             # (C, X, Y, Z)
            target_shape = tuple(shp["target"])             # (1, X, Y, Z)
            spatial = kspace_shape[1:]                      # (X, Y, Z)
            num_slices = int(spatial[self.readout_axis])
            meta = {
                "kspace_shape": kspace_shape,
                "target_vol_shape": target_shape[1:],       # (X, Y, Z)
                "num_coils": int(kspace_shape[0]),
                "num_slices": num_slices,
            }
            for s in range(num_slices):
                self.raw_samples.append((vk, s, meta))

        # post-IFFT volume cache: vol_key -> {"ks": ..., "maps": ..., "norm": ...}
        self._vol_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []

        logging.info(
            f"BratsSliceDataset: {len(self._volume_keys)} volumes, "
            f"{len(self.raw_samples)} slices "
            f"(readout_axis={self.readout_axis}, "
            f"return_sensmaps={self.return_sensmaps}, root={self.root_dir})"
        )

    # ------------------------------------------------------------------
    def _read_shapes(self, vk: str) -> Dict[str, Any]:
        with self.env_shapes.begin() as txn:
            raw = txn.get(vk.encode())
        if raw is None:
            raise KeyError(f"shape for key {vk!r} not in LMDB at {self.root_dir}")
        return json.loads(raw.decode())

    def _read_array(self, env, vk: str, shape: Tuple[int, ...]) -> np.ndarray:
        with env.begin() as txn:
            buf = txn.get(vk.encode())
        if buf is None:
            raise KeyError(f"value for key {vk!r} missing in {env.path()}")
        return np.frombuffer(buf, dtype=np.complex64).reshape(shape).copy()

    def _load_volume(self, vk: str) -> Dict[str, Any]:
        if vk in self._vol_cache:
            return self._vol_cache[vk]

        shp = self._read_shapes(vk)
        kspace = self._read_array(self.env_kspace, vk, tuple(shp["kspace"]))
        maps = self._read_array(self.env_maps, vk, tuple(shp["maps"])) \
               if self.return_sensmaps else None
        # target_full shape is (1, X, Y, Z) — drop the leading singleton channel.
        target_full = self._read_array(self.env_target, vk, tuple(shp["target"]))[0]

        # kspace_vol_norm on the RAW 3D k-space, before any IFFT, so it
        # matches fastmri_slice_dataset.py:262 (`np.linalg.norm(hf["kspace"])`).
        # See module docstring for why this is computed pre-IFFT.
        norm = float(np.linalg.norm(kspace))

        # 1D IFFT along readout (axis 1+readout_axis in the (C,X,Y,Z) layout).
        ax = 1 + self.readout_axis
        ks_after = np.fft.ifftshift(kspace, axes=ax)
        ks_after = np.fft.ifft(ks_after, axis=ax)
        ks_after = np.fft.fftshift(ks_after, axes=ax).astype(np.complex64)

        entry = {
            "ks": ks_after,
            "maps": maps,                     # (C, X, Y, Z) complex or None
            "target": target_full,            # (X, Y, Z) complex64
            "norm": norm,
            "kspace_shape": tuple(shp["kspace"]),
            "target_vol_shape": tuple(shp["target"][1:]),
        }
        # tiny LRU
        if len(self._cache_order) >= self.volume_cache_size:
            evict = self._cache_order.pop(0)
            self._vol_cache.pop(evict, None)
        self._vol_cache[vk] = entry
        self._cache_order.append(vk)
        return entry

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, i: int) -> SliceDatasetSample:
        vk, s_idx, meta = self.raw_samples[i]
        vol = self._load_volume(vk)

        ax = 1 + self.readout_axis
        kspace_slice = np.take(vol["ks"], s_idx, axis=ax)              # (C, *, *)
        kspace_t = torch.from_numpy(np.ascontiguousarray(kspace_slice))
        kspace_t = torch.view_as_real(kspace_t).contiguous()           # (C, *, *, 2)

        attrs: Dict[str, Any] = {
            "kspace_shape": meta["kspace_shape"],
            "target_vol_shape": meta["target_vol_shape"],
            "num_coils": meta["num_coils"],
            "num_slices": meta["num_slices"],
            "kspace_vol_norm": vol["norm"],
        }

        if self.return_sensmaps and vol["maps"] is not None:
            sens_slice = np.take(vol["maps"], s_idx, axis=ax)          # (C, *, *)
            attrs["sens_maps"] = np.ascontiguousarray(sens_slice).astype(np.complex64)

        # Slice the target along the same axis as kspace, but adjust the axis
        # index: `vol["ks"]` has a leading coil dim so kspace slicing uses
        # axis = 1 + readout_axis; `vol["target"]` has no coil dim so the
        # target axis = readout_axis. Convert to view_as_real (Y, Z, 2) so
        # downstream interpolate / Conv2D pipeline (which doesn't accept
        # complex tensors) works identically to the SKM-TEA path.
        target_slice = np.take(vol["target"], s_idx, axis=self.readout_axis)
        target_complex = torch.from_numpy(np.ascontiguousarray(target_slice)).to(torch.complex64)
        target_t = torch.view_as_real(target_complex).contiguous()       # (Y, Z, 2)

        sample = SliceDatasetSample(kspace=kspace_t, target=target_t, attrs=attrs)
        if self.transform is not None:
            return self.transform(sample)
        return sample
