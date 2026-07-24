import numpy as np
import pytest

from brukerapi.data import DataRandomAccess
from brukerapi.dataset import Dataset
from brukerapi.exceptions import UnsupportedDatasetType


def test_ra(test_ra_data):
    loaded = Dataset(test_ra_data[0])

    if loaded.type != "2dseq":
        with pytest.raises(UnsupportedDatasetType, match=rf"mmap=True.*{loaded.type}"):
            Dataset(test_ra_data[0], mmap=True)
        return

    mmap = Dataset(test_ra_data[0], mmap=True)

    assert isinstance(mmap.data, DataRandomAccess)

    core = tuple(slice(None) for _ in range(loaded.encoded_dim))
    frame_shape = loaded.shape[loaded.encoded_dim :]

    index = tuple(0 for _ in frame_shape)
    assert np.array_equal(loaded.data[core + index], mmap.data[core + index])

    subarray = tuple(slice(0, min(size, 2)) for size in frame_shape)
    assert np.array_equal(loaded.data[core + subarray], mmap.data[core + subarray])


def test_2dseq_ra_honors_encoded_axis_slices(test_ra_data):
    loaded = Dataset(test_ra_data[0])
    if loaded.type != "2dseq":
        return

    mmap = Dataset(test_ra_data[0], mmap=True)
    encoded_slices = tuple(slice(0, min(size, 2)) for size in loaded.shape[: loaded.encoded_dim])
    frame_index = tuple(0 for _ in loaded.shape[loaded.encoded_dim :])
    selection = encoded_slices + frame_index

    assert np.array_equal(loaded.data[selection], mmap.data[selection])
