from brukerapi.dataset import Dataset
import numpy as np

def test_ra(test_ra_data, data_path):

    d1 = Dataset(data_path/test_ra_data['path'])
    core_index = tuple(slice(None) for i in range(d1.encoded_dim))
    d2 = Dataset(data_path/test_ra_data['path'], random_access=True)

    if "slices" in test_ra_data.keys():
        for s in test_ra_data['slices']:
            slice_ = json_to_slice(s)
            assert np.array_equal(d1.data[slice_], d2.data[slice_])
    else:
        # test by single slice - index
        for index in np.ndindex(d1.shape[d1.encoded_dim:]):
            assert np.array_equal(d1.data[core_index+index], d2.data[core_index+index])

        # test all possible slices
        for slice_ in generate_slices(d1.shape[d1.encoded_dim:]):
            assert np.array_equal(d1.data[core_index + slice_], d2.data[core_index + slice_])

def generate_slices(shape):
    slices = []
    for i1 in np.ndindex(shape):
        for i2 in np.ndindex(shape):
            if np.all(np.array(i1) <= np.array(i2)):
                slice_ = tuple(slice(i1_, i2_+1) for i1_, i2_ in zip(i1, i2))
                slices.append(slice_)
    return slices

def json_to_slice(s):
    slice_ = []
    for item in s:
        if isinstance(item,list):
            if item[0] == 'None':
                item_0 = None
            else:
                item_0 = item[0]
            if item[1] == 'None':
                item_1 = None
            else:
                item_1 = item[1]
            if item[2] == 'None':
                item_2 = None
            else:
                item_2 = item[2]
            slice_.append(slice(item_0,item_1,item_2))
        elif isinstance(item, int):
            slice_.append(item)
    return tuple(slice_)

