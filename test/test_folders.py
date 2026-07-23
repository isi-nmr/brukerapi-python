import copy
import pickle
from pathlib import Path

import pytest

from brukerapi.dataset import Dataset
from brukerapi.folders import Folder, Processing, Study, TypeFilter

PV51_STUDY_PATH = Path("test/test_data/PV51/0.2H2")


def test_folder_attribute_miss_supports_hasattr_deepcopy_and_pickle(tmp_path):
    (tmp_path / "child").mkdir()
    folder = Folder(tmp_path, recursive=False)

    assert not hasattr(folder, "missing")
    with pytest.raises(AttributeError, match="missing"):
        _ = folder.missing
    with pytest.raises(KeyError, match="Child 'missing' not found"):
        _ = folder["missing"]

    copied = copy.deepcopy(folder)
    restored = pickle.loads(pickle.dumps(folder))

    assert copied.path == folder.path
    assert restored.path == folder.path
    assert [child.path.name for child in copied.children] == [child.path.name for child in folder.children]
    assert [child.path.name for child in restored.children] == [child.path.name for child in folder.children]


def test_type_filter_forwards_nondefault_filter_options(tmp_path):
    folder = Folder(tmp_path, recursive=False)
    type_filter = TypeFilter(Dataset, in_place=False, recursive=False)

    filtered = type_filter.filter(folder)

    assert type_filter.query is None
    assert type_filter.in_place is False
    assert type_filter.recursive is False
    assert isinstance(filtered, Folder)
    assert filtered is not folder


def test_folder_clean_and_in_place_filter_return_folder(tmp_path):
    folder = Folder(tmp_path, recursive=False)

    assert folder.clean() is folder
    assert TypeFilter(Dataset, in_place=True).filter(folder) is folder


def test_folder_traversal_skips_processed_spectra(tmp_path):
    experiment_path = tmp_path / "1"
    processing_path = experiment_path / "pdata" / "1"
    processing_path.mkdir(parents=True)

    for path in [
        experiment_path / "acqp",
        experiment_path / "method",
        experiment_path / "fid",
        processing_path / "visu_pars",
        processing_path / "reco",
        processing_path / "2dseq",
        processing_path / "1r",
        processing_path / "1i",
    ]:
        path.write_text("")

    folder = Folder(
        tmp_path,
        dataset_state={"parameter_files": [], "property_files": [], "load": 0},
    )

    experiment = folder["1"]
    experiment_datasets = {child.path.name for child in experiment.children if isinstance(child, Dataset)}
    assert experiment_datasets == {"fid"}

    processing = next(child for child in experiment.get_processing_list() if isinstance(child, Processing))
    processing_datasets = {child.path.name for child in processing.children if isinstance(child, Dataset)}
    assert processing_datasets == {"2dseq"}


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_study_get_dataset_returns_fid_and_2dseq():
    study = Study(
        PV51_STUDY_PATH,
        dataset_state={"parameter_files": [], "property_files": [], "load": 0},
    )

    fid = study.get_dataset(exp_id="10")
    reconstructed = study.get_dataset(exp_id="10", proc_id="1")

    assert fid.path.name == "fid"
    assert reconstructed.path.name == "2dseq"

    with fid, reconstructed:
        assert fid.data.size > 0
        assert reconstructed.data.size > 0
