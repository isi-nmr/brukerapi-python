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
        "manufacturer": "<Bruker BioSpin MRI GmbH>",
        "acq_software_version": "<PV-360.3.7>",
        "institution": "<Institute>",
        "station": "<System C1>",
    }
    assert metadata["visu_instance"]["uid"] == "<2.16.756.5.5.100.1>"
    assert metadata["visu_instance"]["type"] == "STANDARD_INSTANCE"
    assert metadata["visu_instance"]["creator_version"] == "<6.0.1>"
    assert metadata["visu_series"]["experiment_number"] == 8
    assert metadata["visu_series"]["processing_number"] == 1
    assert metadata["visu_acq"]["protocol"] == "<T1_FLASH>"
    assert metadata["visu_acq"]["echo_time"] == 3.5
    assert "uisition_protocol" not in metadata["visu_acq"]


def test_metadata_can_be_exported(tmp_path):
    dataset = Dataset(study(tmp_path, extra=EQUIPMENT), load=LOAD_STAGES["properties"])

    exported = dataset.to_dict(props=["metadata"])

    # arrays inside the grouped dict used to make json.dump fail outright
    assert json.loads(json.dumps(exported))["metadata"]["visu_equipment"]["station"] == "<System C1>"


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
    assert dataset.dim_type == ["spatial", "spatial", "frame", "<FG_MOVIE>"]
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
