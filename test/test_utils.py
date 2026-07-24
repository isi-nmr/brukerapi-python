from types import SimpleNamespace

import numpy as np

from brukerapi.utils import combine_channels


def test_combine_channels_uses_root_sum_of_squared_magnitudes():
    dataset = SimpleNamespace(dim_type=["sample", "channel"])
    data = np.array([[3 + 4j, 1 + 0j]])

    combined = combine_channels(dataset, data)

    assert np.array_equal(combined, np.array([[np.sqrt(26)]]))
    assert not np.iscomplexobj(combined)


def test_combine_channels_leaves_channel_free_data_unchanged():
    data = np.array([3 + 4j, 1 + 0j])
    dataset = SimpleNamespace(data=data, dim_type=["sample"])

    assert combine_channels(dataset) is data
