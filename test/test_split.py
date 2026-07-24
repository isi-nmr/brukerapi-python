from types import SimpleNamespace

import numpy as np

from brukerapi.dataset import Dataset
from brukerapi.splitters import FrameGroupSplitter, SlicePackageSplitter, Splitter


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
