import copy
import pickle
from pathlib import Path

import numpy as np
import pytest

from brukerapi.dataset import Dataset
from brukerapi.folders import (
    DEFAULT_DATASET_STATE,
    Experiment,
    Folder,
    Processing,
    Study,
    TypeFilter,
)
from test.synthetic import Verbatim, write_binary, write_jcampdx

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


def test_folder_is_not_iterable_and_points_at_children(tmp_path):
    """`__getitem__` must not make a folder iterable by the legacy protocol.

    Without an explicit opt out, `for child in folder` calls `folder[0]`, which
    misses and raises `KeyError: 0` against a name-keyed tree. `children` is the
    route that says whether names or objects are wanted.
    """
    (tmp_path / "child").mkdir()

    for cls in (Folder, Study, Experiment, Processing):
        assert cls.__iter__ is None, f"{cls.__name__} must opt out of iteration"

    folder = Folder(tmp_path)
    with pytest.raises(TypeError, match="not iterable"):
        iter(folder)
    with pytest.raises(TypeError, match="not iterable"):
        list(folder)

    assert [child.path.name for child in folder.children] == ["child"]
    assert folder["child"] is folder.children[0]


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


def test_folder_dataset_state_defaults_are_fresh_and_inputs_are_not_mutated(tmp_path):
    custom_state = {
        "parameter_files": ["custom-parameter"],
        "property_files": [Path("custom-property.json")],
        "load": 1,
    }
    original_custom_state = copy.deepcopy(custom_state)
    original_default_state = copy.deepcopy(DEFAULT_DATASET_STATE)

    configured = Folder(tmp_path, recursive=False, dataset_state=custom_state)
    defaulted = Folder(tmp_path, recursive=False)

    assert custom_state == original_custom_state
    assert original_default_state == DEFAULT_DATASET_STATE
    assert configured._dataset_state["parameter_files"][-1] == "custom-parameter"
    assert configured._dataset_state["property_files"][-1] == Path("custom-property.json")
    assert defaulted._dataset_state == original_default_state
    assert configured._dataset_state is not defaulted._dataset_state


def test_folder_discovery_reuses_dataset_rawdata_subtype_rules(tmp_path):
    for name in [
        "rawdata.job0",
        "rawdata.Navigator",
        "rawdata.npz",
        "rawdata.json",
        "rawdata.anything",
    ]:
        (tmp_path / name).touch()
    write_jcampdx(
        tmp_path / "acqp",
        {"ACQ_jobs": Verbatim("( 1 )\n(8, 20, 5, 1, 101, 1, 1, 1, <echoNavigator>)")},
    )
    (tmp_path / "rawdata.echoNavigator").touch()

    folder = Folder(
        tmp_path,
        recursive=False,
        dataset_state={"parameter_files": [], "property_files": [], "load": 0},
    )

    datasets = {child.path.name for child in folder.children if isinstance(child, Dataset)}
    assert datasets == {"rawdata.job0", "rawdata.Navigator", "rawdata.echoNavigator"}


def test_folder_skips_empty_reconstruction_with_warning(tmp_path):
    (tmp_path / "2dseq").touch()
    (tmp_path / "visu_pars").touch()

    with pytest.warns(RuntimeWarning, match="Skipping invalid dataset.*empty or incomplete reconstruction"):
        folder = Folder(tmp_path, recursive=False)

    assert not any(isinstance(child, Dataset) for child in folder.children)


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


def test_study_get_dataset_falls_back_to_rawdata_when_there_is_no_fid(tmp_path):
    """Spec 13.1: ParaVision 360 writes no file named `fid`.

    Raw data lives in rawdata.jobN, so an unconditional exp["fid"] makes
    Study.get_dataset unusable for every PV360 study.
    """
    study = tmp_path / "20250814_100419_std_PV360_1_1"
    write_jcampdx(study / "subject", {"SUBJECT_id": ["<phantom>"], "SUBJECT_study_nr": 1})
    experiment = study / "26"
    write_jcampdx(
        experiment / "acqp",
        {
            "ACQ_word_size": "_32_BIT",
            "ACQ_sw_version": "<PV-360.3.7>",
            "BYTORDA": "little",
            "ACQ_jobs": Verbatim("( 1 )\n(8, 20, 5, 4, 101, 178571.4, 4, 1, <job0>)"),
        },
    )
    write_jcampdx(experiment / "method", {"Method": "<Bruker:FLASH>", "PVM_EncNReceivers": 1})
    write_binary(experiment / "rawdata.job0", np.arange(8 * 1 * 4, dtype="<i4"), np.dtype("<i4"))

    dataset = Study(study).get_dataset("26")

    assert dataset.path.name == "rawdata.job0"
    assert dataset.type == "rawdata"
    assert dataset.shape_storage == (8, 1, 4)


def test_study_get_dataset_still_raises_when_an_experiment_has_no_raw_data(tmp_path):
    study = tmp_path / "20250814_100419_std_PV360_1_1"
    write_jcampdx(study / "subject", {"SUBJECT_id": ["<phantom>"], "SUBJECT_study_nr": 1})
    write_jcampdx(study / "26" / "acqp", {"ACQ_scan_name": ["<empty>"]})

    with pytest.raises(KeyError, match="fid"):
        Study(study).get_dataset("26")
