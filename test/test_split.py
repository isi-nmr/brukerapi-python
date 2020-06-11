from brukerapi.dataset import Dataset
from brukerapi.splitters import *
import pytest
from pathlib import Path


def test_split(test_split_data, tmp_path):
    dataset = Dataset(Path(test_split_data[1]) / test_split_data[0]['path'])

    if test_split_data[0]['splitter'] == 'SlicePackage':
        SlicePackageSplitter().split(dataset, write=True, path_out=tmp_path)
    elif test_split_data[0]['splitter'] == 'FG_ECHO':
        FrameGroupSplitter('FG_ECHO').split(dataset, write=True, path_out=tmp_path)

    for ref in test_split_data[0]['results'].values():
        ds_split = Dataset(tmp_path / ref['path'])
        assert ds_split.shape == tuple(ref['shape'])

