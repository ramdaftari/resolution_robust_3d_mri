"""
SkmteaVolumeDataset
-------------------
Serves full 3D SKM-TEA volumes in the layout expected by FastMRI3DDataTransform
(= the `mri3d` recon trafo, fastmri_3d_trafo.py).

SKM-TEA h5 layout (per volume, one echo selected):
    kspace : (X, Y, Z, echo, C)    complex    -- already 1D-FFT'd along X (readout)
    target : (X, Y, Z, echo, 1)    complex
    maps   : (X, Y, Z, C, 1)       complex    -- precomputed ESPIRiT sensitivities

fastmri_3d_trafo.py expects:
    kspace  : torch real, shape (C, S1, S2, S3, 2)  — coil first, spatial in last 3 dims
              (Poisson-disc mask is sampled over shape[-3] x shape[-2], which must be the
               two phase-encoding axes; for SKM-TEA those are Y and Z.)
              ifft3c does 3D IFFT over the spatial dims -> needs FULLY 3D kspace.
    sens_maps: numpy complex, shape (S1, S2, S3, C) — coil LAST
              (trafo does movedim(-1, 0) internally to align with target.)

Therefore this dataset:
  * selects one echo, yielding kspace (X, Y, Z, C), target (X, Y, Z), maps (X, Y, Z, C)
  * rearranges kspace to (C, X, Y, Z) and applies fft1c along the readout axis (X)
    so the trafo's ifft3c sees fully-3D kspace. fft1c is norm-preserving, so
    `kspace_vol_norm` (= norm of the kspace after this step) matches what the
    training pipeline computed on the raw h5 kspace.
  * leaves sens_maps as (X, Y, Z, C) with coils last, matching the trafo convention.
  * mask order (Y, Z) is preserved because with layout (C, X, Y, Z) we have
    kspace.shape[-3] == Y and kspace.shape[-2] == Z.

Output per sample: VolumeDatasetSample(kspace, target, attrs) where
    kspace : torch.Tensor real, shape (C, X, Y, Z, 2)
    target : torch.Tensor complex, shape (X, Y, Z)    -- h5 target, only used if
             target_type != "fullysampled_rec" in the trafo; otherwise the trafo
             recomputes target from kspace and ignores this field. Passed non-None
             so the trafo's `if target is not None` branch runs.
    attrs  : dict with
             "sens_maps"       : numpy complex (X, Y, Z, C), coils LAST
             "kspace_vol_norm" : float, ||kspace||_2 over the full volume
             "target_vol_shape": (X, Y, Z)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.fastmri_volume_dataset import VolumeDatasetSample
from src.utils.fftn3d import ifftshift, fftshift


class SkmteaVolumeDataset(BaseDataset):
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
                "SkmteaVolumeDataset: provide exactly one of `file_list` or `csv_path`"
            )
        if csv_path is not None:
            file_list = self._parse_file_csv(Path(csv_path))
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

        self.samples: List[Tuple[Path, Dict[str, Any]]] = []
        for fp in self.fpaths:
            meta = self._retrieve_metadata(fp)
            self.samples.append((fp, meta))

        logging.info(
            f"SkmteaVolumeDataset: {len(self.fpaths)} volumes (echo={echo})"
        )

    @staticmethod
    def _parse_file_csv(csv_path: Path) -> List[str]:
        """Parse Armeet's SKM-TEA split CSV. CRLF tolerated.
        Expected columns include 'file_name'."""
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

    def _retrieve_metadata(self, fname: Path) -> Dict[str, Any]:
        with h5py.File(fname, "r") as hf:
            ks_shape_full = hf["kspace"].shape  # (X, Y, Z, echo, C)
            target_vol_shape = hf["target"].shape[:3]
            meta: Dict[str, Any] = {
                "num_slices":       int(ks_shape_full[0]),
                "num_coils":        int(ks_shape_full[4]),
                "kspace_shape":     (ks_shape_full[0], ks_shape_full[1],
                                     ks_shape_full[2], ks_shape_full[4]),
                "target_vol_shape": tuple(target_vol_shape),
            }
        return meta

    def calc_sensmap_files(self):
        """No-op: SKM-TEA files already contain sens maps in the 'maps' key."""
        return

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _fft1c(data_real: torch.Tensor, dim: int, norm: str = "ortho") -> torch.Tensor:
        """Centered 1D FFT along `dim` on a real-as-complex tensor of shape (..., 2).

        Mirrors FastMRIVolumeDataset._fft1c (fastmri_volume_dataset.py:283-296).
        """
        if data_real.shape[-1] != 2:
            raise ValueError("Expected last dim = 2 (real/imag).")
        data_real = ifftshift(data_real, dim=[dim])
        data_real = torch.view_as_real(
            torch.fft.fftn(torch.view_as_complex(data_real), dim=[dim], norm=norm)
        )
        data_real = fftshift(data_real, dim=[dim])
        return data_real

    def __getitem__(self, i: int) -> VolumeDatasetSample:
        fname, volume_meta = self.samples[i]

        with h5py.File(fname, "r") as hf:
            # kspace: (X, Y, Z, echo, C) -> select echo -> (X, Y, Z, C)
            ksp_np = np.asarray(
                hf["kspace"][:, :, :, self.echo_idx, :], dtype=np.complex64
            )
            # target: (X, Y, Z, echo, 1) -> (X, Y, Z)
            tgt_np = np.asarray(
                hf["target"][:, :, :, self.echo_idx, 0], dtype=np.complex64
            )
            # maps: (X, Y, Z, C, 1) -> (X, Y, Z, C). Coils LAST per trafo convention.
            if self.return_sensmaps:
                sens_np = np.asarray(
                    hf["maps"][:, :, :, :, 0], dtype=np.complex64
                )
            else:
                sens_np = None

        # Rearrange kspace to (C, X, Y, Z). X stays second so mask (shape[-3]=Y,
        # shape[-2]=Z) samples the phase-encoding axes.
        ksp_np = np.moveaxis(ksp_np, -1, 0)  # (C, X, Y, Z)

        # fft1c along X (dim=1 after coil move) to make fully-3D kspace.
        # Input h5 kspace is pre-1D-FFT'd along X (readout); trafo needs full kspace.
        ksp_real = torch.view_as_real(
            torch.from_numpy(ksp_np).to(torch.complex64)
        ).contiguous()  # (C, X, Y, Z, 2)
        ksp_real = self._fft1c(ksp_real, dim=1, norm="ortho")

        # kspace_vol_norm: computed after fft1c. fft1c is unitary, so this equals
        # the norm of the raw pre-fft1c kspace -> matches training pipeline.
        kspace = ksp_real.contiguous()
        kspace_vol_norm = float(torch.linalg.vector_norm(kspace).item())

        target = torch.from_numpy(tgt_np).to(torch.complex64)

        attrs: Dict[str, Any] = dict(volume_meta)
        attrs["kspace_vol_norm"] = kspace_vol_norm
        attrs["target_vol_shape"] = tuple(tgt_np.shape)  # (X, Y, Z)
        if sens_np is not None:
            attrs["sens_maps"] = sens_np  # (X, Y, Z, C) complex, coils LAST

        sample = VolumeDatasetSample(kspace=kspace, target=target, attrs=attrs)
        if self.transform is not None:
            return self.transform(sample)
        return sample