from types import SimpleNamespace

from brukerapi.dataset import Dataset
from brukerapi.splitters import FrameGroupSplitter, SlicePackageSplitter, Splitter


def test_split_transposition_is_noop_when_parameter_is_absent():
    dataset = SimpleNamespace(shape_final=(2, 3), encoded_dim=2)

    result = Splitter()._split_VisuCoreTransposition(dataset, {}, 0, 2)

    assert result is None


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
