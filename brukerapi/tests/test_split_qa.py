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
path_qa = Path('C:/data/bruker2nifti_qa/raw')


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

with open('test_split_qa.json') as json_file:
    reference = json.load(json_file)

class TestSplittersQA(unittest.TestCase):

    # def test_split_sp_qa(self):
    #     reference_ = reference['split_sp']
    #
    #     dataset = Dataset(path_qa / reference_['path'])
    #
    #     SlicePackageSplitter().split(dataset, write=True, path_out=results_path)
    #
    #     for ref in reference_['results'].values():
    #         ds_split = Dataset(results_path / ref['path'])
    #
    #         print(ds_split.shape)
    #
    #         self.assertEqual(ds_split.shape, tuple(ref['shape']))
    #
    #     clear_folder(results_path)

    def test_split_fg_echo_qa(self):
        reference_ = reference['split_echo']

        dataset = Dataset(path_qa / reference_['path'])

        FrameGroupSplitter('FG_ECHO').split(dataset, write=True, path_out=results_path)

        for ref in reference_['results'].values():
            ds_split = Dataset(results_path / ref['path'])

            print(ds_split.shape)

            self.assertEqual(ds_split.shape, tuple(ref['shape']))

        # clear_folder(results_path)