from brukerapi.dataset import Dataset
from brukerapi.splitters import *
import pytest


def test_split(test_split_data, data_path, tmp_path):
    dataset = Dataset(data_path / test_split_data['path'])

    if test_split_data['splitter'] == 'SlicePackage':
        SlicePackageSplitter().split(dataset, write=True, path_out=tmp_path)
    elif test_split_data['splitter'] == 'FG_ECHO':
        FrameGroupSplitter('FG_ECHO').split(dataset, write=True, path_out=tmp_path)

    for ref in test_split_data['results'].values():
        ds_split = Dataset(tmp_path / ref['path'])
        assert ds_split.shape == tuple(ref['shape'])

