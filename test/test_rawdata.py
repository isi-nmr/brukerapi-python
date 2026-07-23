from pathlib import Path

import pytest

from brukerapi.dataset import Dataset

RAWDATA_JOB_PATHS = sorted(Path("test/test_data").rglob("rawdata.job*"))


@pytest.mark.parametrize("rawdata_path", RAWDATA_JOB_PATHS, ids=[str(path) for path in RAWDATA_JOB_PATHS])
def test_rawdata_job_loads_directly(rawdata_path):
    dataset = Dataset(rawdata_path)

    assert dataset.type == "rawdata"
    assert dataset.subtype == rawdata_path.suffix.removeprefix(".")
    assert dataset.data.shape == dataset._schema.layouts["raw"]
    assert dataset.data.size > 0
