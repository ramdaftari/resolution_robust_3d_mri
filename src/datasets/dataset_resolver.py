from pathlib import Path
from typing import Tuple, List, Callable

import numpy as np
import logging

from torch import Tensor

from torch.utils.data import ConcatDataset
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import h5py
import numpy as np
import os

from omegaconf import DictConfig

from src.datasets.fastmri_slice_dataset import ExtSliceDataset
from src.datasets.fastmri_volume_dataset import FastMRIVolumeDataset
from fastmri.data.mri_data import FastMRIRawDataSample
from .base_dataset import BaseDataset

from torch.utils.data import Dataset

def get_fastmri_dataset(
        fold_overwrite : Optional[str],
        fold : str,
        dataset_trafo : Optional[Any],
        data_path_train : Dict,
        data_path_val : Dict,
        data_path_test : Dict,
        data_path_sensmaps_train : Dict,
        data_path_sensmaps_val : Dict,
        data_path_sensmaps_test : Dict,
        volume_filter_train : Optional[str],
        volume_filter_val : Optional[str],
        volume_filter_test : Optional[str],
        path_resolver : Optional[Callable],
        raw_sample_filter : Dict,
        # raw_sample_filter_enabled : bool,
        # raw_sample_filter_encoding_size: Optional[int],
        data_object_type : str = "slices",
        **dataset_kwargs
        ):

    dataset = None
    data_path = None
    data_path_sensmaps = None
    volume_filter = None

    from src.problem_trafos.dataset_trafo.fastmri_2d_trafo import FastMRI2DDataTransform
    from src.problem_trafos.dataset_trafo.fastmri_3d_trafo import FastMRI3DDataTransform 
    assert isinstance(dataset_trafo, FastMRI3DDataTransform) or isinstance(dataset_trafo, FastMRI2DDataTransform), "Dataset transform must be of type FastMRI3DDataTransform or FastMRI2DDataTransform, but is: {}".format(type(dataset_trafo))
    requires_sensmaps = dataset_trafo.requires_sensmaps()

    if fold_overwrite is not None and fold != fold_overwrite:
        logging.info(f"Overwriting fold {fold} with {fold_overwrite}")
        fold = fold_overwrite
    
    if "train" in fold:
        data_path = path_resolver(data_path_train)
        data_path_sensmaps = path_resolver(data_path_sensmaps_train)
        volume_filter = volume_filter_train
    elif "val" in fold:
        data_path = path_resolver(data_path_val)
        data_path_sensmaps = path_resolver(data_path_sensmaps_val)
        volume_filter = volume_filter_val
    elif "test" in fold:
        data_path = path_resolver(data_path_test)
        data_path_sensmaps = path_resolver(data_path_sensmaps_test)
        volume_filter = volume_filter_test
    else:
        raise NotImplementedError(f"Fold {fold} not supported")

    if data_object_type == "slices":

        raw_sample_filter_func = None

        if isinstance(data_path, str):
            dataset = ExtSliceDataset(root=data_path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=data_path_sensmaps, return_sensmaps=requires_sensmaps, volume_filter=volume_filter, **dataset_kwargs)

        else:
            dataset = ConcatDataset([ExtSliceDataset(root=path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=path_sense, return_sensmaps=requires_sensmaps, volume_filter=volume_filter, **dataset_kwargs) for path, path_sense in zip(data_path, data_path_sensmaps)])

    elif data_object_type == "volumes":

        raw_sample_filter_func = None

        if isinstance(data_path, str):
            dataset = FastMRIVolumeDataset(root=data_path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=data_path_sensmaps, volume_filter=volume_filter, return_sensmaps=requires_sensmaps, **dataset_kwargs)
        else:
            dataset = ConcatDataset([FastMRIVolumeDataset(root=path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=path_sense, volume_filter=volume_filter, return_sensmaps=requires_sensmaps, **dataset_kwargs) for path, path_sense in zip(data_path, data_path_sensmaps)])

    else:
        raise NotImplementedError(f"Dataset type {data_object_type} not supported")

    # check if we need to calculate sensemaps    
    if requires_sensmaps:
        logging.info("Sensmaps required, generate if necessary...")
        if isinstance(data_path, str):
            dataset.calc_sensmap_files()
        elif isinstance(data_path, list):
            for ds in dataset.datasets:
                ds.calc_sensmap_files()


    return dataset

def get_skmtea_dataset(
        fold_overwrite,
        fold,
        dataset_trafo,
        data_root,
        train_files=None,
        val_files=None,
        test_files=None,
        train_csv=None,
        val_csv=None,
        test_csv=None,
        echo=1,
        path_resolver=None,
        **dataset_kwargs,
        ):
    if fold_overwrite is not None and fold != fold_overwrite:
        logging.info(f"Overwriting fold {fold} with {fold_overwrite}")
        fold = fold_overwrite

    if "train" in fold:
        file_list, csv_path = train_files, train_csv
    elif "val" in fold:
        file_list, csv_path = val_files, val_csv
    elif "test" in fold:
        file_list, csv_path = test_files, test_csv
    else:
        raise NotImplementedError(f"Fold {fold} not supported")

    if (file_list is None) == (csv_path is None):
        raise ValueError(
            f"get_skmtea_dataset: fold={fold} must specify exactly one of "
            f"file list or csv (got file_list={file_list}, csv_path={csv_path})"
        )

    if path_resolver is not None:
        data_root = path_resolver(data_root)

    # Dispatch on data_object_type (default 'slices' preserves training behavior).
    data_object_type = dataset_kwargs.pop("data_object_type", "slices")
    # drop fields that pass through dataset yaml but aren't used by these Skmtea classes
    for _k in ("data_path_sensmaps_train", "data_path_sensmaps_val", "data_path_sensmaps_test",
               "volume_filter_train", "volume_filter_val", "volume_filter_test",
               "raw_sample_filter"):
        dataset_kwargs.pop(_k, None)

    if data_object_type == "slices":
        from src.datasets.skmtea_slice_dataset import SkmteaSliceDataset
        kspace_norms_path = dataset_kwargs.pop("kspace_norms_path", None)
        return SkmteaSliceDataset(
            data_root=data_root,
            file_list=list(file_list) if file_list is not None else None,
            csv_path=csv_path,
            echo=echo,
            transform=dataset_trafo,
            kspace_norms_path=kspace_norms_path,
        )
    elif data_object_type == "volumes":
        from src.datasets.skmtea_volume_dataset import SkmteaVolumeDataset
        return SkmteaVolumeDataset(
            data_root=data_root,
            file_list=list(file_list) if file_list is not None else None,
            csv_path=csv_path,
            echo=echo,
            transform=dataset_trafo,
        )
    else:
        raise NotImplementedError(
            f"SKM-TEA data_object_type {data_object_type!r} not supported "
            f"(expected 'slices' or 'volumes')."
        )

def get_brats_dataset(
        fold_overwrite,
        fold,
        dataset_trafo,
        data_root_train,
        data_root_val,
        data_root_test,
        readout_axes=(0, 1, 2),
        return_sensmaps=True,
        volume_cache_size=1,
        max_volumes=None,
        path_resolver=None,
        **dataset_kwargs,
        ):
    """Construct a multi-plane BraTS slice dataset.

    Mirrors the fastMRI three-plane recipe: build one BratsSliceDataset
    per readout axis (axial / sagittal / coronal) over the same LMDB,
    then ConcatDataset them so each epoch sees slices from all three
    anatomical planes — matching the 2D diffusion training protocol of
    Krainovic et al. (2024), Section 4.1.
    """
    if fold_overwrite is not None and fold != fold_overwrite:
        logging.info(f"Overwriting fold {fold} with {fold_overwrite}")
        fold = fold_overwrite

    if "train" in fold:
        data_root = data_root_train
    elif "val" in fold:
        data_root = data_root_val
    elif "test" in fold:
        data_root = data_root_test
    else:
        raise NotImplementedError(f"Fold {fold} not supported")

    if path_resolver is not None:
        data_root = path_resolver(data_root)

    # drop fields passed through the dataset yaml that BratsSliceDataset
    # does not consume (mirrors get_skmtea_dataset's defensive pop).
    for _k in ("volume_filter_train", "volume_filter_val", "volume_filter_test",
               "raw_sample_filter", "data_object_type"):
        dataset_kwargs.pop(_k, None)

    from src.datasets.brats_slice_dataset import BratsSliceDataset

    # Resolve the volume key list once (truncated if max_volumes set), so
    # each per-axis dataset sees the same volumes. None => use all keys.
    # We go through the module-level cache (_open_lmdb) so this enumeration
    # share the same Environment handle with the BratsSliceDataset instances
    # constructed below — lmdb-py refuses two opens of the same path per
    # process.
    keys = None
    if max_volumes is not None:
        from src.datasets.brats_slice_dataset import _open_lmdb
        env_shapes = _open_lmdb(Path(data_root) / "shapes")
        with env_shapes.begin() as txn:
            all_keys = sorted([k.decode() for k, _ in txn.cursor()],
                              key=lambda x: int(x))
        keys = all_keys[: int(max_volumes)]
        logging.info(f"get_brats_dataset: truncated to first {len(keys)} volumes "
                     f"(max_volumes={max_volumes})")

    per_axis = [
        BratsSliceDataset(
            root_dir=data_root,
            readout_axis=int(ax),
            return_sensmaps=return_sensmaps,
            transform=dataset_trafo,
            keys=keys,
            volume_cache_size=volume_cache_size,
        )
        for ax in readout_axes
    ]
    if len(per_axis) == 1:
        return per_axis[0]
    return ConcatDataset(per_axis)


def get_dataset(
        name : str,
        dataset_trafo : Optional[Any] = None,
        path_resolver : Optional[Callable] = None,
        fold_overwrite : Optional[str] = None,
        **cfg_kwargs
        ) -> BaseDataset:
    if name == "FastMRIDataset":

        from src.datasets.dataset_resolver import get_fastmri_dataset
        dataset = get_fastmri_dataset(
            dataset_trafo=dataset_trafo,
            path_resolver=path_resolver,
            fold_overwrite=fold_overwrite,
            **cfg_kwargs
        )
    elif name == "SkmteaDataset":
        dataset = get_skmtea_dataset(
            dataset_trafo=dataset_trafo,
            path_resolver=path_resolver,
            fold_overwrite=fold_overwrite,
            **cfg_kwargs,
        )
    elif name == "BratsDataset":
        dataset = get_brats_dataset(
            dataset_trafo=dataset_trafo,
            path_resolver=path_resolver,
            fold_overwrite=fold_overwrite,
            **cfg_kwargs,
        )
    else:
        raise NotImplementedError(f"Dataset {name} not supported")

    return dataset