from types import SimpleNamespace

import numpy as np

from brukerapi.dataset import Dataset
from brukerapi.splitters import FrameGroupSplitter, SlicePackageSplitter, Splitter
from test.synthetic import axial_orientation, stacked_positions, write_2dseq


def test_split_transposition_is_noop_when_parameter_is_absent():
    dataset = SimpleNamespace(shape_final=(2, 3), encoded_dim=2)

    result = Splitter()._split_VisuCoreTransposition(dataset, {}, 0, 2)

    assert result is None


def test_split_transposition_uses_the_relative_frame_group_axis():
    transposition = SimpleNamespace(value=np.array([0, 1, 1]), size=(3,))
    dataset = SimpleNamespace(shape_final=(2, 3, 3), encoded_dim=2)

    Splitter()._split_VisuCoreTransposition(dataset, {"VisuCoreTransposition": transposition}, slice(1, 2), 0)

    assert transposition.size == (1,)
    assert np.array_equal(transposition.value, np.array([1]))


def test_split(test_split_data, tmp_path):
    tmp_path /= "FG/"
    dataset = Dataset(test_split_data[0])

    if "FG_ECHO" not in dataset.dim_type:
        return

    datasets = FrameGroupSplitter("FG_ECHO").split(dataset, write=True, path_out=tmp_path)

    assert len(datasets) == dataset.shape[dataset.dim_type.index("FG_ECHO")]


def test_splitSlicePkg(test_split_data, tmp_path):
    tmp_path /= "Slice/"
    dataset = Dataset(test_split_data[0])

    if "FG_SLICE" not in dataset.dim_type:
        return
    if "VisuCoreSlicePacksSlices" not in dataset:
        return

    datasets = SlicePackageSplitter().split(dataset, write=True, path_out=tmp_path)

    assert len(datasets) == dataset["VisuCoreSlicePacksSlices"].size[0]


def test_slice_package_splitter_infers_and_synthesises_pv51_packages(tmp_path):
    """PV5.1 has no §7.10 package parameters; orientation delimits packages."""
    sagittal = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
    positions = np.vstack(
        [
            stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), 2),
            stacked_positions((0.0, -20.0, -20.0), (-1.0, 0.0, 0.0), 3),
        ]
    )
    orientations = np.vstack([axial_orientation(2), np.tile(sagittal, (3, 1))])
    path = write_2dseq(
        tmp_path / "9" / "pdata" / "1",
        creator_version="5.1",
        frame_groups=(("FG_SLICE", 5),),
        positions=positions,
        orientations=orientations,
    )

    dataset = Dataset(path)
    packages = SlicePackageSplitter().split(dataset, write=True)

    assert dataset.num_slice_packages == 2
    assert [package.data.shape[2] for package in packages] == [2, 3]
    assert all(package.num_slice_packages == 1 for package in packages)
    assert [package["VisuCoreSlicePacksSlices"].nested for package in packages] == [[[0, 2]], [[0, 3]]]
    assert [package["VisuCoreSlicePacksDef"].value for package in packages] == [[0, 1], [0, 1]]
    assert [Dataset(package.path).num_slice_packages for package in packages] == [1, 1]


def _echo_dataset(tmp_path, echoes=2, slices=3):
    path = write_2dseq(
        tmp_path / "5" / "pdata" / "1",
        frame_groups=(("FG_ECHO", echoes), ("FG_SLICE", slices)),
        positions=stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), echoes * slices),
        extra={"VisuAcqEchoTime": np.array([11.0, 22.0])[:echoes]},
    )
    return Dataset(path)


def test_frame_group_split_writes_no_files_when_it_is_not_asked_to(tmp_path):
    dataset = _echo_dataset(tmp_path)

    datasets = FrameGroupSplitter("FG_ECHO").split(dataset, write=False)

    assert len(datasets) == 2
    # The in-memory split used to create a *directory* named 2dseq for every
    # part, which both polluted the dataset tree and made write() impossible.
    assert sorted(entry.name for entry in (tmp_path / "5" / "pdata").iterdir()) == ["1"]
    assert all(not part.path.exists() for part in datasets)


def test_frame_group_split_writes_datasets_that_can_be_read_back(tmp_path):
    dataset = _echo_dataset(tmp_path)
    expected = [dataset.data[:, :, index, :] for index in range(2)]

    parts = FrameGroupSplitter("FG_ECHO").split(dataset, write=True)

    for index, part in enumerate(parts):
        assert part.path.is_file()
        written = Dataset(part.path)
        assert np.array_equal(np.squeeze(written.data), np.squeeze(expected[index]))
        assert written["VisuAcqEchoTime"].value == [11.0, 22.0][index]


def test_frame_group_split_honours_the_output_folder(tmp_path):
    dataset = _echo_dataset(tmp_path)
    out = tmp_path / "out"

    parts = FrameGroupSplitter("FG_ECHO").split(dataset, write=True, path_out=out)

    written = sorted(path.relative_to(out).as_posix() for path in out.rglob("2dseq"))
    assert written == ["1_FG_ECHO_0/2dseq", "1_FG_ECHO_1/2dseq"]
    assert all(not part.path.exists() for part in parts)


def test_slice_package_split_keeps_a_fractional_slice_distance(tmp_path):
    """Spec 7.10: VisuCoreSlicePacksSliceDist is a double[].

    Casting it to int truncated 0.7 mm to 0 and 1.5 mm to 1 for every split
    package, which then reports a slice spacing the data does not have.
    """
    path = write_2dseq(
        tmp_path / "4" / "pdata" / "1",
        frame_groups=(("FG_SLICE", 4),),
        positions=stacked_positions((-20.0, -20.0, -1.05), (0.0, 0.0, 0.7), 4),
        frame_thickness=0.7,
        slice_packs=(0, [(0, 2), (2, 2)]),
        slice_pack_distance=[0.7, 0.7],
    )
    dataset = Dataset(path)

    packages = SlicePackageSplitter().split(dataset, write=False)

    for package in packages:
        assert package["VisuCoreSlicePacksSliceDist"].value == 0.7
        assert np.isclose(package.resolution[2], 0.7)
