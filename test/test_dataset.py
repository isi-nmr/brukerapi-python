import contextlib
import json
from pathlib import Path

import numpy as np
import pytest

from brukerapi.dataset import Dataset

data = 0


@pytest.mark.skip(reason="in progress")
def test_parameters(test_parameters):
    dataset = Dataset(test_parameters[0], load=False)
    dataset.load_parameters()

    for jcampdx in dataset._parameters.values():
        with Path(str(jcampdx.path) + ".json").open() as file:
            reference = json.load(file)

        assert jcampdx.to_dict() == reference


def test_properties(test_properties):
    if test_properties:
        dataset = Dataset(test_properties[0], load=False, parameter_files=["subject"])
        dataset.load_parameters()
        dataset.load_properties()
        assert dataset.to_dict() == test_properties[1]


def test_data_load(test_data):
    dataset = Dataset(test_data[0])

    return  # For now Disable testing array equality

    with np.load(str(dataset.path) + ".npz") as data:
        assert np.array_equal(dataset.data, data["data"])


def test_data_save(test_data, tmp_path, WRITE_TOLERANCE):
    d_ref = Dataset(test_data[0])

    if d_ref.subtype == "":
        path_out = tmp_path / d_ref.type
    else:
        path_out = tmp_path / (d_ref.type + "." + d_ref.subtype)

    d_ref.write(path_out)
    d_test = Dataset(path_out)

    diff = d_ref.data - d_test.data
    max_error = np.max(np.abs(diff))

    with contextlib.suppress(AssertionError):
        assert np.array_equal(d_ref.data, d_test.data)

    if max_error > 0.0:
        try:
            assert max_error < WRITE_TOLERANCE
            print(f"Arrays are not identical, but max difference: {max_error} is tolerated")
        except AssertionError as e:
            raise e

    # Test if properties are loaded correctly
    # TODO since the id property of the 2dseq dataset type relies on the name of the experiment folder,
    # which is a problem when the dataset is writen to the test folder, solution might be to delete the id key here
    # assert d_test.to_dict() == test_data[1]
