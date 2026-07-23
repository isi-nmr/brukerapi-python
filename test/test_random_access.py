import numpy as np

from brukerapi.dataset import Dataset


def test_ra(test_ra_data):
    loaded = Dataset(test_ra_data[0])
    mmap = Dataset(test_ra_data[0], mmap=True)

    core = tuple(slice(None) for _ in range(loaded.encoded_dim))
    frame_shape = loaded.shape[loaded.encoded_dim :]

    index = tuple(0 for _ in frame_shape)
    assert np.array_equal(loaded.data[core + index], mmap.data[core + index])

    subarray = tuple(slice(0, min(size, 2)) for size in frame_shape)
    assert np.array_equal(loaded.data[core + subarray], mmap.data[core + subarray])
