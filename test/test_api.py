"""Public API surface: metadata groups, reports, the CLI and split state.

Synthetic datasets only -- see test/synthetic.py.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from brukerapi.cli import report as cli_report
from brukerapi.cli import split as cli_split
from brukerapi.dataset import LOAD_STAGES, Dataset
from brukerapi.exceptions import UnsupportedDatasetType
from brukerapi.folders import Folder
from brukerapi.splitters import SlicePackageSplitter
from test.synthetic import stacked_positions, write_2dseq, write_jcampdx


def study(tmp_path, **kwargs):
    """A one-experiment study with a subject file, and its 2dseq path."""
    root = tmp_path / "20200612_094625_study_1_1"
    write_jcampdx(root / "subject", {"SUBJECT_id": ["<phantom>"], "SUBJECT_study_nr": 1})
    return write_2dseq(root / "8" / "pdata" / "1", **kwargs)


EQUIPMENT = {
    "VisuManufacturer": ["<Bruker BioSpin MRI GmbH>"],
    "VisuAcqSoftwareVersion": ["<PV-360.3.7>"],
    "VisuInstitution": ["<Institute>"],
    "VisuStation": ["<System C1>"],
    "VisuAcquisitionProtocol": ["<T1_FLASH>"],
    "VisuAcqEchoTime": 3.5,
    "VisuExperimentNumber": 8,
    "VisuProcessingNumber": 1,
    "VisuUid": ["<2.16.756.5.5.100.1>"],
    "VisuInstanceType": "STANDARD_INSTANCE",
}


def test_metadata_groups_follow_the_specification(tmp_path):
    """Spec 7.1/7.7/7.8: several groups cannot be recognised from a name prefix.

    No parameter is called VisuEquipment*, so that bucket was structurally
    always empty, VisuManufacturer/VisuInstitution/VisuStation and the
    experiment/processing numbers matched no group at all, and the 7.1
    administration group was missing. Stripping the VisuAcq prefix without
    checking for a word boundary also produced `acquisition_protocol` as
    `uisition_protocol`.
    """
    dataset = Dataset(study(tmp_path, extra=EQUIPMENT), load=LOAD_STAGES["properties"])

    metadata = dataset.metadata

    assert metadata["visu_equipment"] == {
        "manufacturer": "Bruker BioSpin MRI GmbH",
        "acq_software_version": "PV-360.3.7",
        "institution": "Institute",
        "station": "System C1",
    }
    assert metadata["visu_instance"]["uid"] == "2.16.756.5.5.100.1"
    assert metadata["visu_instance"]["type"] == "STANDARD_INSTANCE"
    assert metadata["visu_instance"]["creator_version"] == "6.0.1"
    assert metadata["visu_series"]["experiment_number"] == 8
    assert metadata["visu_series"]["processing_number"] == 1
    assert metadata["visu_acq"]["protocol"] == "T1_FLASH"
    assert metadata["visu_acq"]["echo_time"] == 3.5
    assert "uisition_protocol" not in metadata["visu_acq"]


def test_metadata_reports_the_same_string_as_the_property_that_reads_it(tmp_path):
    """`subj_id` and `metadata` read VisuSubjectId; they must agree.

    `subj_id` stripped the `<...>` delimiters in its recipe, `metadata` -- the
    newer surface -- did not, so the two APIs reported different values for one
    parameter for every string-valued field of every group.
    """
    dataset = Dataset(study(tmp_path), load=LOAD_STAGES["properties"])

    assert dataset.metadata["visu_subject"]["id"] == "phantom"
    assert dataset.subj_id == dataset.metadata["visu_subject"]["id"]


def test_subj_id_is_the_subject_identifier_not_the_subject_name(tmp_path):
    """A 2dseq read VisuSubjectName while fid/rawdata/traj read SUBJECT_id (#216).

    ParaVision keeps the two apart -- VisuSubjectName is the DICOM patient
    name, `family^given^middle^prefix^suffix` on PV360 -- and VisuSubjectId is
    the Visu copy of SUBJECT_id. `subj_id`, and the `id` built from it, must
    mean the same thing for every dataset type.
    """
    dataset = Dataset(study(tmp_path), add_parameters=["subject"], load=LOAD_STAGES["properties"])

    assert dataset.metadata["visu_subject"]["name"] == "synthetic"
    assert dataset.subj_id == dataset["SUBJECT_id"].value == "phantom"
    assert dataset.id == "2DSEQ_8_1_phantom_1"


def test_get_returns_a_default_where_a_property_does_not_resolve(tmp_path):
    """Which properties a dataset carries is ParaVision-version dependent.

    A recipe whose parameters the files do not contain leaves its property
    unset, which attribute access can only report as an AttributeError -- so a
    caller had to wrap every read in try/except, and that swallows typos and
    genuine recipe errors along with the absence.
    """
    with_echo_time = Dataset(study(tmp_path / "with", extra={"VisuAcqEchoTime": 3.5}), load=LOAD_STAGES["properties"])
    without = Dataset(study(tmp_path / "without"), load=LOAD_STAGES["properties"])

    assert with_echo_time.get("TE") == with_echo_time.TE == 3.5
    assert not hasattr(without, "TE")
    assert without.get("TE") is None
    assert without.get("TE", "n/a") == "n/a"


def test_get_still_raises_for_a_name_that_is_not_a_property(tmp_path):
    """A misspelling must not quietly become the default."""
    dataset = Dataset(study(tmp_path), load=LOAD_STAGES["properties"])

    with pytest.raises(AttributeError, match="TEE"):
        dataset.get("TEE")

    # Nor may `get` turn a property's own diagnosis into an absence: a
    # spectroscopy scan has no image geometry, and says so.
    spectroscopy = Dataset(
        write_2dseq(tmp_path / "spectroscopy" / "9" / "pdata" / "1", dim=1, dim_desc=("spectroscopic",), size=(16,), extent=(1.0,), frame_groups=()),
        load=LOAD_STAGES["properties"],
    )
    with pytest.raises(UnsupportedDatasetType, match="rather than purely spatial"):
        spectroscopy.get("affine")


def test_metadata_can_be_exported(tmp_path):
    dataset = Dataset(study(tmp_path, extra=EQUIPMENT), load=LOAD_STAGES["properties"])

    exported = dataset.to_dict(props=["metadata"])

    # arrays inside the grouped dict used to make json.dump fail outright
    assert json.loads(json.dumps(exported))["metadata"]["visu_equipment"]["station"] == "System C1"


def test_report_honours_the_requested_format(tmp_path):
    """`-f yml` was ignored for a single dataset: report always appended .json."""
    dataset = Dataset(study(tmp_path), load=LOAD_STAGES["properties"])
    out = tmp_path / "out"
    out.mkdir()

    dataset.report(out, props=["id"], format_="yml")
    dataset.report(props=["id"], format_="yml")

    assert [file.name for file in out.iterdir()] == [f"{dataset.id}.yml"]
    assert (dataset.path.parent / f"{dataset.id}.yml").is_file()
    with pytest.raises(ValueError, match="unsupported report format"):
        dataset.report(props=["id"], format_="csv")


def test_cli_report_creates_an_output_folder_that_does_not_exist_yet(tmp_path):
    """`-i <dir> -o <new dir>` matched no branch and exited 0 having done nothing."""
    path = study(tmp_path)
    out = tmp_path / "reports"

    cli_report(SimpleNamespace(input=str(path.parents[3]), output=str(out), format="json", props=["id"], verbose=False))

    assert [file.suffix for file in out.iterdir()] == [".json"]


def test_cli_split_writes_to_the_output_folder(tmp_path):
    """`bruker split -o out/` parsed path_out and then never passed it on."""
    path = study(
        tmp_path,
        frame_groups=(("FG_ECHO", 2),),
        positions=stacked_positions((-20.0, -20.0, 0.0), (0.0, 0.0, 1.0), 2),
        extra={"VisuAcqEchoTime": np.array([11.0, 22.0])},
    )
    out = tmp_path / "split"

    cli_split(SimpleNamespace(path_in=str(path), path_out=str(out), slice_package=False, frame_group="FG_ECHO"))

    assert sorted(file.relative_to(out).as_posix() for file in out.rglob("2dseq")) == ["1_FG_ECHO_0/2dseq", "1_FG_ECHO_1/2dseq"]


def test_folder_to_json_serialises_instead_of_recursing(tmp_path):
    """`to_json` called itself forever, and report(write=False) yielded strings."""
    study(tmp_path)
    folder = Folder(tmp_path)

    exported = json.loads(folder.to_json(props=["id"]))
    reported = folder.report(write=False, props=["id"])

    assert list(exported) == list(reported)
    assert all(isinstance(value, dict) for value in reported.values())


def test_slice_package_split_keeps_the_parent_state(tmp_path):
    """A package used to be constructed from DEFAULT_STATES only, so the
    parent's scale/combine_complex/property_files were dropped."""
    path = study(
        tmp_path,
        frame_groups=(("FG_SLICE", 4),),
        positions=stacked_positions((-20.0, -20.0, -1.5), (0.0, 0.0, 1.0), 4),
        slice_packs=(0, [(0, 2), (2, 2)]),
        slice_pack_distance=[1.0, 1.0],
        slope=2.0,
        offset=1.0,
    )
    dataset = Dataset(path, scale=False)

    packages = SlicePackageSplitter().split(dataset, write=False)

    assert [package._state["scale"] for package in packages] == [False, False]
    assert np.array_equal(packages[0].data, dataset.data[:, :, :2])


def test_procno_files_are_reachable(tmp_path):
    """Spec 13/13.2 document methreco and pvmeta; both were missing from
    RELATIVE_PATHS, so add_parameter_file raised KeyError."""
    path = study(tmp_path)
    write_jcampdx(path.parent / "methreco", {"RecoMethMode": "Default"})
    write_jcampdx(path.parent / "pvmeta", {"RefCopyId": 1})

    dataset = Dataset(path, load=LOAD_STAGES["parameters"])
    dataset.add_parameter_file("methreco")
    dataset.add_parameter_file("pvmeta")

    assert dataset["RecoMethMode"].value == "Default"
    assert dataset["RefCopyId"].value == 1


def test_documented_optional_parameter_files_are_reachable(tmp_path):
    """FILE_FORMAT.md 1.1-1.3 place further JCAMP-DX files at the study,
    experiment and reconstruction levels; none were in RELATIVE_PATHS, so
    add_parameter_file raised KeyError for all of them."""
    path = study(tmp_path)
    procno, expno, study_dir = path.parent, path.parents[2], path.parents[3]
    locations = {
        "uxnmr.par": (expno, {"SFO1": 400.3}),
        "specpar": (expno, {"LOCNUC": ["<2H>"]}),
        "ResultState": (study_dir, {"AdjResultChain": 1}),
        "study.MR": (study_dir, {"MR_study_gradient_system": ["<Micro 2.5>"]}),
        "study.PT": (study_dir, {"PT_study_version": 1}),
        "id": (procno, {"DATASET_KEY": ["<key>"]}),
        "procs": (procno, {"OFFSET": 62.4}),
        "roi": (procno, {"ROI_n": 1}),
        "isa": (procno, {"ISA_first_image": 25}),
    }
    for name, (folder, records) in locations.items():
        write_jcampdx(folder / name, records)

    dataset = Dataset(path, load=LOAD_STAGES["parameters"])
    for name in locations:
        dataset.add_parameter_file(name)

    assert set(locations) <= set(dataset._parameters)
    assert dataset["MR_study_gradient_system"].value == "Micro 2.5"
    assert dataset["ISA_first_image"].value == 25


def test_the_scan_configuration_is_optional_and_reachable(tmp_path):
    """configscan sits in the scan folder next to acqp and method, and is the
    only record of the gradient system. It was in neither RELATIVE_PATHS nor
    the optional files, so its parameters could not be read at all."""
    path = study(tmp_path)
    write_jcampdx(path.parents[2] / "configscan", {"CONFIG_SCAN_gradient_system": ["<Micro 2.5>"]})

    assert Dataset(path, load=LOAD_STAGES["parameters"])["CONFIG_SCAN_gradient_system"].value == "Micro 2.5"

    # Optional, like reco and d3proc: not every export carries one.
    without = Dataset(study(tmp_path / "no_configscan"), load=LOAD_STAGES["parameters"])
    assert "CONFIG_SCAN_gradient_system" not in without


def test_single_slice_datasets_do_not_claim_a_third_spatial_axis(tmp_path):
    """The synthetic axis a single-slice dataset gets is not an encoding axis;
    labelling it `spatial` made dim_type[encoded_dim:] start with a bogus entry."""
    path = study(
        tmp_path,
        frame_groups=(("FG_MOVIE", 3),),
        positions=np.array([[-20.0, -20.0, 0.0]]),
    )
    dataset = Dataset(path)

    assert dataset.is_single_slice
    assert dataset.dim_type == ["spatial", "spatial", "frame", "FG_MOVIE"]
    assert len(dataset.dim_type) == dataset.data.ndim


def test_random_access_can_select_one_half_of_a_complex_axis(tmp_path):
    """Selecting only the real (or only the imaginary) component raised
    InvalidDataset instead of returning that component."""
    path = study(
        tmp_path,
        frame_groups=(("FG_COMPLEX", 2),),
        positions=np.array([[-20.0, -20.0, 0.0]]),
    )
    full = Dataset(path)
    selected = Dataset(path, mmap=True)

    assert np.iscomplexobj(full.data)
    assert np.array_equal(selected.data[:, :, 0, 0], np.real(full.data[:, :, 0]))
