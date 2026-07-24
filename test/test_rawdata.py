from pathlib import Path

import numpy as np
import pytest

from brukerapi.dataset import Dataset
from brukerapi.exceptions import UnknownAcqSchemeException

RAWDATA_JOB_PATHS = sorted(Path("test/test_data").rglob("rawdata.job*"))


def _raw_complex_stream(dataset):
    stored = dataset._read_binary_file(dataset.path, dataset.numpy_dtype, dataset.shape_storage)
    return stored[0::2, ...] + 1j * stored[1::2, ...]


def _rawdata_with_kspace(test_data_root, predicate):
    for path in sorted(test_data_root.rglob("rawdata.job*")):
        dataset = Dataset(path)
        try:
            k_space = dataset.to_kspace()
        except UnknownAcqSchemeException:
            continue
        if predicate(k_space):
            return dataset, k_space
    pytest.skip("The selected corpus has no rawdata job with the required Cartesian layout")


def _epi_rawdata(test_data_root):
    for path in sorted(test_data_root.rglob("rawdata.job*")):
        dataset = Dataset(path)
        try:
            dataset.to_kspace()
        except UnknownAcqSchemeException as error:
            if "is EPI" in str(error):
                return dataset
    pytest.skip("The selected corpus has no EPI rawdata job")


@pytest.mark.parametrize("rawdata_path", RAWDATA_JOB_PATHS, ids=[str(path) for path in RAWDATA_JOB_PATHS])
def test_rawdata_job_loads_directly(rawdata_path):
    dataset = Dataset(rawdata_path)
    with pytest.warns(FutureWarning, match="format-dependent legacy semantics"):
        legacy_data = dataset.data

    assert dataset.type == "rawdata"
    assert dataset.subtype == rawdata_path.suffix.removeprefix(".")
    assert legacy_data.shape == dataset._schema.layouts["raw"]
    assert legacy_data.size > 0
    assert np.array_equal(dataset.raw, np.transpose(legacy_data, (0, 2, 1)))


def test_pv360_cartesian_rawdata_job_converts_to_ordered_k_space(test_data_root):
    dataset, k_space = _rawdata_with_kspace(test_data_root, lambda data: data.ndim == 5 and data.shape[2:4] == (1, 1))

    assert np.array_equal(dataset.kspace, k_space)
    assert k_space.shape[-1] == dataset.channels
    assert k_space.size == dataset.raw.size

    bart = dataset.to_kspace(bart=True)
    assert bart.ndim == 16
    assert bart.shape[3] == dataset.channels
    assert bart.size == k_space.size


def test_pv360_3d_cartesian_rawdata_job_converts_to_k_space(test_data_root):
    dataset, k_space = _rawdata_with_kspace(test_data_root, lambda data: data.ndim == 5 and data.shape[2] > 1)

    assert k_space.shape[-1] == dataset.channels
    assert k_space.size == dataset.raw.size


def test_pv360_self_gated_rawdata_exposes_acquired_cardiac_frames(test_data_root):
    dataset, k_space = _rawdata_with_kspace(test_data_root, lambda data: data.ndim == 6 and data.shape[2] > 1 and data.shape[3] > 1)

    assert np.array_equal(dataset.kspace, k_space)
    assert k_space.shape[-1] == dataset.channels
    assert k_space.size == dataset.raw.size

    bart = dataset.to_kspace(bart=True)
    assert bart.ndim == 16
    assert bart.shape[3] == dataset.channels
    assert bart.size == k_space.size


def test_pv360_epi_rawdata_job_requires_an_acquisition_specific_reader(test_data_root):
    dataset = _epi_rawdata(test_data_root)

    with pytest.raises(UnknownAcqSchemeException, match="is EPI"):
        dataset.to_kspace()
