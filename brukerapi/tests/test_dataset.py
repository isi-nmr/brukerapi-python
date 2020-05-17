from brukerapi.dataset import Dataset
from brukerapi.schemes import *
import numpy as np
import json
import unittest
import os
from pathlib2 import Path
import shutil

data_path = Path('C:/data/') # ['C:/data/', '/home/tomas/data/']
results_path = Path('results')
WRITE_TOLERANCE = 1.e6

with open('test_dataset.json') as json_file:
    reference = json.load(json_file)


class TestDataset(unittest.TestCase):

    def test_read(self):

        reference_ = reference['test_read_write']

        for ref in reference_.items():
            path = data_path / ref[1]['path']
            print("test_read: {}".format(ref[0]))
            d = Dataset(path, load=False)
            d.load_parameters()
            d.load_scheme()
            self.schemes_one(d, ref[1])
            d.load_data()
            self.read_one(d, ref)

    def test_write(self):

        reference_ = reference['test_read_write']

        try:
            os.mkdir(results_path)
        except:
            self.clear_results(results_path)

        for ref in reference_.items():
            print("test_write: {}".format(ref[0]))
            self.write_one(ref[1])

    def test_exceptions(self):
        reference_ = reference['test_exceptions']

        path = data_path / reference_['INCOMPLETE_FID']['path']

        with self.assertRaises(IncompleteDataset):
            Dataset(path)

        path = data_path / reference_['INCOMPLETE_2DSEQ']['path']
        with self.assertRaises(IncompleteDataset):
            Dataset(path)





    def clear_results(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def schemes_one(self, d, r):
        if isinstance(d.scheme, SchemeFid):
            self.assertEqual(r['block_size'], d.scheme.block_size)
            self.assertEqual(r['single_acq_length'], d.scheme.single_acq_length)
            self.assertEqual(r['acq_scheme'], d.scheme.meta['id'])
            self.assertEqual(r['layouts']['encoding_space'], list(d.scheme.layouts['encoding_space']))
            self.assertEqual(r['layouts']['k_space'], list(d.scheme.layouts['k_space']))
        elif isinstance(d.scheme, Scheme2dseq):
            self.assertEqual(r['layouts']['frame_groups'], list(d.scheme.layouts['frame_groups']))
            self.assertEqual(r['layouts']['frames'], list(d.scheme.layouts['frames']))
        elif isinstance(d.scheme, SchemeRawdata):
            self.assertEqual(r['layouts']['raw'], list(d.scheme.layouts['raw']))


        self.assertEqual(r['dim_type'], d.scheme.dim_type)

    def read_one(self, d_test, ref):
        self.assertIsNotNone(d_test.data)

    def write_one(self, ref):
        d_ref = Dataset(data_path/ref['path'])

        if d_ref.subtype is None:
            path_out = results_path / d_ref.type
        else:
            path_out = results_path / (d_ref.type + '.' + d_ref.subtype)

        d_ref.write(path_out )
        d_test = Dataset(path_out)

        diff = d_ref.data - d_test.data
        max_error = np.max(diff)

        try:
            self.assertTrue(np.array_equal(d_ref.data, d_test.data))
        except AssertionError:
            pass

        if max_error > 0.0:
            try:
                self.assertLess(max_error, WRITE_TOLERANCE)
                print('Arrays are not identical, but max difference: {} is tolerated'.format(max_error))
            except AssertionError as e:
                raise e

        self.schemes_one(d_test, ref)

        self.clear_results(results_path)