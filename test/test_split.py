from types import SimpleNamespace

import numpy as np

from brukerapi.dataset import Dataset
from brukerapi.splitters import FrameGroupSplitter, SlicePackageSplitter, Splitter
from test.synthetic import stacked_positions, write_2dseq


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

    if "<{}>".format("FG_ECHO") not in dataset.dim_type:
        return

    datasets = FrameGroupSplitter("FG_ECHO").split(dataset, write=True, path_out=tmp_path)

    assert len(datasets) == dataset.shape[dataset.dim_type.index("<{}>".format("FG_ECHO"))]


def test_splitSlicePkg(test_split_data, tmp_path):
    tmp_path /= "Slice/"
    dataset = Dataset(test_split_data[0])

    if "<{}>".format("FG_SLICE") not in dataset.dim_type:
        return
    if "VisuCoreSlicePacksSlices" not in dataset:
        return

    datasets = SlicePackageSplitter().split(dataset, write=True, path_out=tmp_path)

    assert len(datasets) == dataset["VisuCoreSlicePacksSlices"].size[0]


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
