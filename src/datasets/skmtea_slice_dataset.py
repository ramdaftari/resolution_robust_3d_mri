"""
SkmteaSliceDataset
------------------
Serves 2D slices from SKM-TEA HDF5 volumes in the format expected by
Krainovic's mri2d_train_calc_mvue transform / FastMRI2DDataTransform.

SKM-TEA file layout (per volume):
    kspace : (X, Y, Z, echo, C)    complex   -- 1D FFT'd along X (readout)
    maps   : (X, Y, Z, C, 1)       complex   -- precomputed sensitivity maps
    target : (X, Y, Z, echo, 1)    complex

Since kspace ships already 1D-FFT'd along the readout axis (X), we treat X as
the slice axis: slice i is kspace[i, :, :, echo, :], which is already image
space along X and still k-space in (Y, Z). This mirrors how fastMRI's slice
dataset treats the first axis.

Output per sample: SliceDatasetSample(kspace, target, attrs) where
    kspace : real tensor shape (C, Y, Z, 2)
    target : complex tensor shape (Y, Z)
    attrs  : dict with "sens_maps" (C, Y, Z) complex numpy,
             "kspace_vol_norm" (scalar float), "target_vol_shape" (tuple)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.fastmri_slice_dataset import SliceDatasetSample


class SkmteaSliceDataset(BaseDataset):
    def __init__(
        self,
        data_root: Union[str, Path],
        file_list: Sequence[str],
        echo: int = 1,
        transform: Optional[Callable] = None,
        return_sensmaps: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        if echo not in (1, 2):
            raise ValueError(f"echo must be 1 or 2, got {echo}")
        self.echo_idx = echo - 1
        self.transform = transform
        self.return_sensmaps = return_sensmaps

        self.fpaths: List[Path] = []
        for fn in file_list:
            fp = self.data_root / fn
            if fp.exists():
                self.fpaths.append(fp)
            else:
                logging.warning(f"SKM-TEA file not found, skipping: {fp}")
        if len(self.fpaths) == 0:
            raise RuntimeError(
                f"No SKM-TEA files found under {self.data_root} "
                f"from file_list ({len(file_list)} entries)."
            )

        # Build flat (fname, slice_idx, volume_metadata) list.
        self.raw_samples: List[Tuple[Path, int, Dict[str, Any]]] = []
        for fp in self.fpaths:
            meta = self._retrieve_metadata(fp)
            num_slices = int(meta["num_slices"])
            for s in range(num_slices):
                self.raw_samples.append((fp, s, meta))

        logging.info(
            f"SkmteaSliceDataset: {len(self.fpaths)} volumes, "
            f"{len(self.raw_samples)} slices (echo={echo})"
        )

    def _retrieve_metadata(self, fname: Path) -> Dict[str, Any]:
        with h5py.File(fname, "r") as hf:
            kspace_full = np.asarray(hf["kspace"][:, :, :, self.echo_idx, :])
            target_vol_shape = hf["target"].shape[:3]
            num_slices = kspace_full.shape[0]
            num_coils = kspace_full.shape[3]
            meta: Dict[str, Any] = {
                "num_slices": num_slices,
                "num_coils": num_coils,
                "kspace_shape": tuple(kspace_full.shape),
                "kspace_vol_norm": float(np.linalg.norm(kspace_full)),
                "target_vol_shape": tuple(target_vol_shape),
            }
        return meta

    def __len__(self) -> int:
        return len(self.raw_samples)

    def calc_sensmap_files(self):
        """No-op: SKM-TEA files already contain sens maps in the 'maps' key."""
        return

    def __getitem__(self, i: int) -> SliceDatasetSample:
        fname, slice_idx, volume_meta = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace_slice_np = np.asarray(
                hf["kspace"][slice_idx, :, :, self.echo_idx, :]
            )
            target_slice_np = np.asarray(
                hf["target"][slice_idx, :, :, self.echo_idx, 0]
            )
            if self.return_sensmaps:
                sens_slice_np = np.asarray(
                    hf["maps"][slice_idx, :, :, :, 0], dtype=np.complex64
                )
            else:
                sens_slice_np = None

        kspace_t = torch.from_numpy(kspace_slice_np).to(torch.complex64)
        kspace_t = kspace_t.permute(2, 0, 1).contiguous()     # (C, Y, Z)
        kspace_t = torch.view_as_real(kspace_t).contiguous()  # (C, Y, Z, 2)

        target_t = torch.from_numpy(target_slice_np).to(torch.complex64)

        attrs: Dict[str, Any] = dict(volume_meta)
        if self.return_sensmaps and sens_slice_np is not None:
            attrs["sens_maps"] = np.moveaxis(sens_slice_np, -1, 0).astype(np.complex64)

        sample = SliceDatasetSample(kspace=kspace_t, target=target_t, attrs=attrs)
        if self.transform is not None:
            return self.transform(sample)
        return sample
