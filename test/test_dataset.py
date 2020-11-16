from brukerapi.dataset import Dataset
from brukerapi.schemas import *
import numpy as np
import json
import os
from pathlib import Path
import shutil
import pytest

def test_properties(test_io_data):
    d = Dataset(Path(test_io_data[1]) / Path(test_io_data[0]['path']), load=False)

    d.load_parameters()
    d.load_properties()

    # Test if properties are loaded correctly
    assert d.to_dict() == test_io_data[0]['properties']

def test_read(test_io_data):
    path = Path(test_io_data[1]) / Path(test_io_data[0]['path'])
    d = Dataset(path)

    """
    The bruker2nifti_qa data set does not posses the reference data
    """
    try:
        data_ref = np.load('{}.npz'.format(str(path)))['data']
    except:
        data_ref = None

    if data_ref is not None:
        assert np.array_equal(d.data, data_ref)

def test_write(test_io_data, tmp_path, WRITE_TOLERANCE):
    d_ref = Dataset(Path(test_io_data[1]) / Path(test_io_data[0]['path']))

    if d_ref.subtype is None:
        path_out = tmp_path / d_ref.type
    else:
        path_out = tmp_path / (d_ref.type + '.' + d_ref.subtype)

    d_ref.write(path_out)
    d_test = Dataset(path_out)

    diff = d_ref.data - d_test.data
    max_error = np.max(np.abs(diff))

    try:
        assert np.array_equal(d_ref.data, d_test.data)
    except AssertionError:
        pass

    if max_error > 0.0:
        try:
            assert max_error < WRITE_TOLERANCE
            print('Arrays are not identical, but max difference: {} is tolerated'.format(max_error))
        except AssertionError as e:
            raise e

    # Test if properties are loaded correctly
    assert d_test.to_dict() == test_io_data[0]['properties']


def schemas_one(d, r):
    if isinstance(d.schema, SchemaFid):
        assert r['acq_schema'] == d.schema._meta['id']
        assert r['layouts']['storage'] == list(d.schema.layouts['storage'])
        assert r['layouts']['acquisition_position'] == list(d.schema.layouts['acquisition_position'])
        assert r['layouts']['encoding_space'] == list(d.schema.layouts['encoding_space'])
        assert r['layouts']['k_space'] == list(d.schema.layouts['k_space'])
    elif isinstance(d.schema, Schema2dseq):
        assert r['layouts']['frame_groups'] == list(d.shape_fg)
        assert r['layouts']['frames'] == list(d.shape_frames)
    elif isinstance(d.schema, SchemaRawdata):
        assert r['layouts']['raw'] == list(d.schema.layouts['raw'])


