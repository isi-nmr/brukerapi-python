from brukerapi.dataset import Dataset
from brukerapi.schemes import *
import numpy as np
import json
import os
from pathlib import Path
import shutil
import pytest


def test_read(test_io_data, data_path):
    d = Dataset(data_path / test_io_data['path'], load=False)
    d.load_parameters()
    d.load_scheme()
    # Test if schemes are loaded correctly
    schemes_one(d, test_io_data)
    d.load_data()
    # Test if schemes are loaded correctly
    read_one(d)


def test_write(test_io_data, data_path, tmp_path, WRITE_TOLERANCE):
    d_ref = Dataset(data_path / test_io_data['path'])

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

    schemes_one(d_test, test_io_data)


def schemes_one(d, r):
    if isinstance(d.scheme, SchemeFid):
        assert r['acq_scheme'] == d.scheme._meta['id']
        assert r['layouts']['storage'] == list(d.scheme.layouts['storage'])
        assert r['layouts']['acquisition_position'] == list(d.scheme.layouts['acquisition_position'])
        assert r['layouts']['encoding_space'] == list(d.scheme.layouts['encoding_space'])
        assert r['layouts']['k_space'] == list(d.scheme.layouts['k_space'])
    elif isinstance(d.scheme, Scheme2dseq):
        assert r['layouts']['frame_groups'] == list(d.scheme.layouts['frame_groups'])
        assert r['layouts']['frames'] == list(d.scheme.layouts['frames'])
    elif isinstance(d.scheme, SchemeRawdata):
        assert r['layouts']['raw'] == list(d.scheme.layouts['raw'])


    assert r['dim_type'] == d.scheme.dim_type

def read_one(d):
    assert d.data is not None

