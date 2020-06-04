from brukerapi.dataset import Dataset
from brukerapi.schemes import *
from brukerapi.splitters import *
import numpy as np
import json
import unittest
import os
from pathlib import Path
import shutil
import requests

results_path = Path('results')
config_data = Path('test_random_access_qa.json')
path_data = Path('C:/data/bruker2nifti_qa/raw')

with open(config_data) as json_file:
    reference = json.load(json_file)

class TestRandomAccess(unittest.TestCase):
    def test_ra(self):
        for r in reference.items():
            print("TestRandomAccess/test_ra:{}".format(r[0]))
            d1 = Dataset(path_data/r[1]['path'])
            core_index = tuple(slice(None) for i in range(d1.encoded_dim))
            d2 = Dataset(path_data/r[1]['path'], random_access=True)

            if "slices" in r[1].keys():
                for s in r[1]['slices']:
                    slice_ = self.json_to_slice(s)
                    print("TestRandomAccess/test_ra:{}/{}".format(r[0],slice_))
                    self.assertTrue(np.array_equal(d1.data[slice_], d2.data[slice_]))
            else:
                # test by single slice - index
                for index in np.ndindex(d1.shape[d1.encoded_dim:]):
                    print("TestRandomAccess/test_ra:{}/{}".format(r[0], core_index + index))
                    self.assertTrue(np.array_equal(d1.data[core_index+index], d2.data[core_index+index]))

                # test all possible slices
                for slice_ in self.generate_slices(d1.shape[d1.encoded_dim:]):
                    print("TestRandomAccess/test_fid:{}/{}".format(r[0],core_index + slice_))
                    self.assertTrue(np.array_equal(d1.data[core_index + slice_], d2.data[core_index + slice_]))

    def generate_slices(self, shape):
        slices = []
        for i1 in np.ndindex(shape):
            for i2 in np.ndindex(shape):
                if np.all(np.array(i1) <= np.array(i2)):
                    slice_ = tuple(slice(i1_, i2_+1) for i1_, i2_ in zip(i1, i2))
                    slices.append(slice_)
        return slices

    def json_to_slice(self, s):
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

