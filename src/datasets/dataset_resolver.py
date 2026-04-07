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

    from src.datasets.skmtea_slice_dataset import SkmteaSliceDataset
    return SkmteaSliceDataset(
        data_root=data_root,
        file_list=list(file_list) if file_list is not None else None,
        csv_path=csv_path,
        echo=echo,
        transform=dataset_trafo,
    )

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
    else: 
        raise NotImplementedError(f"Dataset {name} not supported")

    return dataset