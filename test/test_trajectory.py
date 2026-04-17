from pathlib import Path

import pytest

from brukerapi.dataset import Dataset


TRAJ_PATHS = sorted(Path("test/test_data").rglob("traj"))


@pytest.mark.parametrize("traj_path", TRAJ_PATHS, ids=[str(path) for path in TRAJ_PATHS])
def test_traj_loads_directly(traj_path):
    dataset = Dataset(traj_path)

    assert dataset.data.shape == dataset.shape_storage
    assert dataset.data.size > 0


@pytest.mark.parametrize("traj_path", TRAJ_PATHS, ids=[str(path) for path in TRAJ_PATHS])
def test_traj_loads_from_parent_fid(traj_path):
    dataset = Dataset(traj_path.parent / "fid")

    assert dataset.traj.shape == dataset._traj.shape_storage
    assert dataset.traj.size > 0
