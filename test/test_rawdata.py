from pathlib import Path

import numpy as np
import pytest

from brukerapi.dataset import Dataset
from brukerapi.exceptions import UnknownAcqSchemeException

RAWDATA_JOB_PATHS = sorted(Path("test/test_data").rglob("rawdata.job*"))


@pytest.mark.parametrize("rawdata_path", RAWDATA_JOB_PATHS, ids=[str(path) for path in RAWDATA_JOB_PATHS])
def test_rawdata_job_loads_directly(rawdata_path):
    dataset = Dataset(rawdata_path)

    assert dataset.type == "rawdata"
    assert dataset.subtype == rawdata_path.suffix.removeprefix(".")
    assert dataset.data.shape == dataset._schema.layouts["raw"]
    assert dataset.data.size > 0


def test_pv360_cartesian_rawdata_job_converts_to_ordered_k_space():
    dataset = Dataset("test/test_data/PV360V37/20260130_203108_example_sourceData_1_1/15/rawdata.job0")

    k_space = dataset.to_kspace()

    assert k_space.shape == (128, 128, 1, 1, 4)
    acquired = np.transpose(np.reshape(dataset.data, (128, 4, 128, 1, 1), order="F"), (0, 2, 3, 4, 1))
    expected = np.take(acquired, np.argsort(dataset["PVM_EncSteps1"].value), axis=1)
    assert np.array_equal(k_space, expected)


def test_pv360_3d_cartesian_rawdata_job_converts_to_k_space():
    dataset = Dataset("test/test_data/PV360V37/20260130_203108_example_sourceData_1_1/22/rawdata.job0")

    assert dataset.to_kspace().shape == (256, 128, 32, 1, 4)


def test_pv360_epi_rawdata_job_requires_an_acquisition_specific_reader():
    dataset = Dataset("test/test_data/PV360V37/20260130_203108_example_sourceData_1_1/10/rawdata.job0")

    with pytest.raises(UnknownAcqSchemeException, match="is EPI"):
        dataset.to_kspace()
