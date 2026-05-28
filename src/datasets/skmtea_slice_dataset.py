"""
SkmteaSliceDataset
------------------
Serves 2D slices from SKM-TEA HDF5 volumes for FastMRI2DDataTransform, in the
**same shape contract** the FastMRI / Stanford / Calgary / BraTS pipelines use:
each sample carries a 2D coil-wise kspace, a 2D MVUE target, and 2D sens maps.

Multi-orientation training (Krainovic et al., Sec. 5):
    "We train the diffusion models on complex 2D slices taken from the MVUE
    reference volumes, at all three anatomical planes (coronal, sagittal and
    axial)."
For FastMRI this is realized by pre-slicing each 3D volume into per-orientation
2D HDF5 files (cor/, sag/, ax/) and concatenating them. SKM-TEA stores the 3D
MVUE in the h5 `target` field and 3D sens maps in `maps`, so we do the same
slicing on the fly. For every chosen slice we synthesize the 2D coil-wise
kspace as `fft2c(target_2D * S_2D_coil)` per coil. With centered orthonormal
FFTs this round-trips exactly through `ifft2c + SENSE-adjoint reduction`, so
the downstream trafo (`mri2d_train_calc_mvue`, `target_type=fullysampled_rec`)
behaves identically to Stanford/Calgary.

SKM-TEA h5 layout (per volume):
    kspace : (X, Y, Z, echo, C)    complex   -- 1D FFT'd along X (readout)
    maps   : (X, Y, Z, C, 1)       complex   -- precomputed ESPIRiT sens maps
    target : (X, Y, Z, echo, 1)    complex   -- precomputed 3D MVUE image

Per-sample contract (SliceDatasetSample):
    target : complex tensor of shape (H, W); the 2D MVUE slice
              axis=0 -> (Y, Z) ; axis=1 -> (X, Z) ; axis=2 -> (X, Y)
    kspace : real tensor (C, H, W, 2); synthesized 2D coil-wise kspace
    attrs  : dict with
             "sens_maps"       : complex numpy (C, H, W), coils first
             "kspace_vol_norm" : float, per-volume ||kspace||_2 (scaling pipeline)
             "target_vol_shape": full 3D shape (X, Y, Z)
             "slice_axis"      : int, 0/1/2 — which anatomical plane this slice is from
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import fastmri

from src.datasets.base_dataset import BaseDataset
from src.datasets.fastmri_slice_dataset import SliceDatasetSample


_ANATOMICAL_AXES: Tuple[int, int, int] = (0, 1, 2)


class SkmteaSliceDataset(BaseDataset):
    def __init__(
        self,
        data_root: Union[str, Path],
        file_list: Optional[Sequence[str]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        echo: int = 1,
        transform: Optional[Callable] = None,
        return_sensmaps: bool = True,
        kspace_norms_path: Optional[Union[str, Path]] = None,
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

        # Optional precomputed kspace_vol_norm sidecar (avoids ~5.4 GB lazy
        # read per volume on first slice access). See precompute_skmtea_kspace_norms.py.
        # JSON layout: {"<filename>": {"<echo>": <float>, ...}, ...}.
        self._norms_json: Dict[str, Dict[str, float]] = {}
        if kspace_norms_path is not None:
            kspace_norms_path = Path(kspace_norms_path)
            if kspace_norms_path.exists():
                import json as _json
                with open(kspace_norms_path, "r") as f:
                    self._norms_json = _json.load(f)
                logging.info(
                    f"SkmteaSliceDataset: loaded {len(self._norms_json)} precomputed "
                    f"kspace norms from {kspace_norms_path}"
                )
            else:
                logging.warning(
                    f"SkmteaSliceDataset: kspace_norms_path {kspace_norms_path} "
                    f"does not exist — falling back to lazy per-volume norm read."
                )

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

        # Flat (fname, slice_idx, axis, volume_meta) list — enumerate slices
        # along all three anatomical axes per volume, matching the paper.
        self.raw_samples: List[Tuple[Path, int, int, Dict[str, Any]]] = []
        per_axis_counts: Dict[int, int] = {a: 0 for a in _ANATOMICAL_AXES}
        precomputed_hits = 0
        for fp in self.fpaths:
            meta = self._retrieve_metadata(fp)
            spatial = meta["kspace_shape"][:3]  # (X, Y, Z)
            for axis in _ANATOMICAL_AXES:
                n = int(spatial[axis])
                per_axis_counts[axis] += n
                for s in range(n):
                    self.raw_samples.append((fp, s, axis, meta))

            # Seed _norm_cache from the JSON sidecar so __getitem__ never reads
            # the full kspace volume just to compute a scalar.
            entry = self._norms_json.get(fp.name)
            if entry is not None:
                v = entry.get(str(echo))
                if v is not None:
                    self._norm_cache[fp] = float(v)
                    precomputed_hits += 1

        if self._norms_json:
            logging.info(
                f"SkmteaSliceDataset: precomputed kspace_vol_norm hits = "
                f"{precomputed_hits}/{len(self.fpaths)}"
            )

        logging.info(
            f"SkmteaSliceDataset: {len(self.fpaths)} volumes, "
            f"{len(self.raw_samples)} slices "
            f"(echo={echo}, per-axis counts={per_axis_counts})"
        )

    def _retrieve_metadata(self, fname: Path) -> Dict[str, Any]:
        with h5py.File(fname, "r") as hf:
            ks_shape_full = hf["kspace"].shape         # (X, Y, Z, echo, C), no data read
            ks_shape = (ks_shape_full[0], ks_shape_full[1], ks_shape_full[2], ks_shape_full[4])
            target_vol_shape = hf["target"].shape[:3]
            meta: Dict[str, Any] = {
                "num_slices":     int(ks_shape[0]),
                "num_coils":      int(ks_shape[3]),
                "kspace_shape":   ks_shape,
                "kspace_vol_norm": None,  # filled lazily on first sample per volume
                "target_vol_shape": tuple(target_vol_shape),
            }
        return meta

    def __len__(self) -> int:
        return len(self.raw_samples)

    @staticmethod
    def _parse_file_csv(csv_path: Path) -> List[str]:
        """Parse Armeet's SKM-TEA split CSV.
        Expected columns: id,file_name,scan_id,subject_id (CRLF tolerated)."""
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

    @staticmethod
    def _load_2d_slice(
        hf: h5py.File, axis: int, slice_idx: int, echo_idx: int, want_sens: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (target_2D, sens_2D) where
            target_2D : complex (H, W)
            sens_2D   : complex (C, H, W) or None
        H/W depend on the axis: (Y,Z), (X,Z), (X,Y) for axis 0, 1, 2.
        """
        if axis == 0:
            target_2d = np.asarray(hf["target"][slice_idx, :, :, echo_idx, 0])    # (Y, Z)
            sens_raw = (np.asarray(hf["maps"][slice_idx, :, :, :, 0])              # (Y, Z, C)
                        if want_sens else None)
        elif axis == 1:
            target_2d = np.asarray(hf["target"][:, slice_idx, :, echo_idx, 0])    # (X, Z)
            sens_raw = (np.asarray(hf["maps"][:, slice_idx, :, :, 0])              # (X, Z, C)
                        if want_sens else None)
        elif axis == 2:
            target_2d = np.asarray(hf["target"][:, :, slice_idx, echo_idx, 0])    # (X, Y)
            sens_raw = (np.asarray(hf["maps"][:, :, slice_idx, :, 0])              # (X, Y, C)
                        if want_sens else None)
        else:
            raise ValueError(f"Unsupported axis {axis}")

        target_2d = target_2d.astype(np.complex64)
        if sens_raw is not None:
            sens_2d = np.moveaxis(sens_raw.astype(np.complex64), -1, 0)            # (C, H, W)
        else:
            sens_2d = None
        return target_2d, sens_2d

    def __getitem__(self, i: int) -> SliceDatasetSample:
        fname, slice_idx, axis, volume_meta = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            target_np, sens_np = self._load_2d_slice(
                hf, axis=axis, slice_idx=slice_idx,
                echo_idx=self.echo_idx, want_sens=self.return_sensmaps,
            )
            if fname not in self._norm_cache:
                kspace_full = np.asarray(hf["kspace"][:, :, :, self.echo_idx, :])
                self._norm_cache[fname] = float(np.linalg.norm(kspace_full))

        target_t = torch.from_numpy(target_np)                                     # complex (H, W)

        # Synthesize 2D coil-wise kspace = fft2c(target * S_c). With centered
        # orthonormal FFTs this is the exact pre-image of `ifft2c + sum_c(* conj S_c)`,
        # so the trafo's target_type=fullysampled_rec path recovers `target`
        # (up to the ESPIRiT sens map's |S|^2 ~= 1 over the support).
        if sens_np is None:
            raise RuntimeError(
                "SkmteaSliceDataset requires return_sensmaps=True to synthesize "
                "the 2D coil-wise kspace expected by the FastMRI2DDataTransform."
            )
        sens_t = torch.from_numpy(sens_np)                                         # complex (C, H, W)
        coil_images = sens_t * target_t.unsqueeze(0)                                # (C, H, W) complex
        coil_kspace = fastmri.fft2c(torch.view_as_real(coil_images).contiguous())   # (C, H, W, 2)
        kspace_t = coil_kspace.contiguous()

        attrs: Dict[str, Any] = dict(volume_meta)
        attrs["kspace_vol_norm"] = self._norm_cache[fname]
        attrs["slice_axis"] = int(axis)
        if self.return_sensmaps:
            attrs["sens_maps"] = sens_np                                            # (C, H, W) complex64

        sample = SliceDatasetSample(kspace=kspace_t, target=target_t, attrs=attrs)
        if self.transform is not None:
            return self.transform(sample)
        return sample
