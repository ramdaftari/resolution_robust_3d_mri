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
        file_list: Optional[Sequence[str]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        echo: int = 1,
        transform: Optional[Callable] = None,
        return_sensmaps: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        if (file_list is None) == (csv_path is None):
            raise ValueError(
                "SkmteaSliceDataset: provide exactly one of `file_list` or `csv_path`"
            )
        if csv_path is not None:
            file_list = self._parse_file_csv(Path(csv_path))
        if echo not in (1, 2):
            raise ValueError(f"echo must be 1 or 2, got {echo}")
        self.echo_idx = echo - 1
        self.transform = transform
        self.return_sensmaps = return_sensmaps
        self._norm_cache: Dict[Path, float] = {}
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
            ks_shape_full = hf["kspace"].shape         # (X, Y, Z, echo, C), no data read
            ks_shape = (ks_shape_full[0], ks_shape_full[1], ks_shape_full[2], ks_shape_full[4])
            target_vol_shape = hf["target"].shape[:3]
            meta: Dict[str, Any] = {
                "num_slices": int(ks_shape[0]),
                "num_coils":  int(ks_shape[3]),
                "kspace_shape": ks_shape,
                "kspace_vol_norm": None,  # sentinel; not consumed unless scale_target_by_kspacenorm=True
                "target_vol_shape": tuple(target_vol_shape),
            }
        return meta

    def __len__(self) -> int:
        return len(self.raw_samples)

    @staticmethod
    def _parse_file_csv(csv_path: Path) -> List[str]:
        """Parse Armeet's SKM-TEA split CSV.
        Expected columns: id,file_name,scan_id,subject_id (CRLF tolerated).
        Returns the list of file_name values in order."""
        import csv
        if not csv_path.exists():
            raise FileNotFoundError(f"SKM-TEA split CSV not found: {csv_path}")
        files: List[str] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "file_name" not in reader.fieldnames:
                raise ValueError(
                    f"SKM-TEA CSV {csv_path} missing 'file_name' column; "
                    f"got fieldnames={reader.fieldnames}"
                )
            for row in reader:
                fn = (row.get("file_name") or "").strip()
                if fn:
                    files.append(fn)
        if not files:
            raise ValueError(f"SKM-TEA CSV {csv_path} parsed to zero files")
        return files

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
            if fname not in self._norm_cache:
                kspace_full = np.asarray(hf["kspace"][:, :, :, self.echo_idx, :])
                self._norm_cache[fname] = float(np.linalg.norm(kspace_full))

        kspace_t = torch.from_numpy(kspace_slice_np).to(torch.complex64)
        kspace_t = kspace_t.permute(2, 0, 1).contiguous()     # (C, Y, Z)
        kspace_t = torch.view_as_real(kspace_t).contiguous()  # (C, Y, Z, 2)

        target_t = torch.from_numpy(target_slice_np).to(torch.complex64)

        attrs: Dict[str, Any] = dict(volume_meta)
        attrs["kspace_vol_norm"] = self._norm_cache[fname]
        if self.return_sensmaps and sens_slice_np is not None:
            attrs["sens_maps"] = np.moveaxis(sens_slice_np, -1, 0).astype(np.complex64)

        sample = SliceDatasetSample(kspace=kspace_t, target=target_t, attrs=attrs)
        if self.transform is not None:
            return self.transform(sample)
        return sample
