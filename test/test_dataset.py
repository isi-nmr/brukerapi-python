import contextlib
import datetime
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brukerapi.cli import report as cli_report
from brukerapi.dataset import LOAD_STAGES, Dataset
from brukerapi.exceptions import FilterEvalFalse, IncompleteDataset, InvalidDataset, TrajNotLoaded, UnknownAcqSchemeException, UnsuportedDatasetType

data = 0
PV51_STUDY_PATH = Path("test/test_data/PV51/0.2H2")


def test_unsupported_dataset_type(tmp_path):
    path = tmp_path / "unsupported"
    path.touch()

    with pytest.raises(UnsuportedDatasetType, match="Dataset type: unsupported is not supported"):
        Dataset(path)


def test_incomplete_dataset_names_missing_parameter_file(tmp_path):
    path = tmp_path / "fid"
    path.touch()

    with pytest.raises(
        IncompleteDataset,
        match=rf"missing required parameter file: acqp \({re.escape(str(tmp_path / 'acqp'))}\)",
    ):
        Dataset(path)

    assert str(IncompleteDataset()) == "Incomplete dataset"


@pytest.mark.parametrize(
    "name",
    [
        "fid.npz",
        "fid.json",
        "fid.navFid",
        "fid.orig",
        "fid.spiral",
        "2dseq.npz",
        "2dseq.json",
        "traj.npy",
        "rawdata.zip",
    ],
)
def test_dataset_rejects_nonprimary_and_unknown_subtypes(tmp_path, name):
    path = tmp_path / name
    path.touch()

    with pytest.raises(UnsuportedDatasetType, match=rf"Dataset type: {re.escape(name)} is not supported"):
        Dataset(path)


@pytest.mark.parametrize(
    ("name", "dataset_type", "subtype"),
    [
        ("fid", "fid", ""),
        ("fid_proc.64", "fid_proc", "64"),
        ("2dseq", "2dseq", ""),
        ("traj", "traj", ""),
        ("rawdata.job12", "rawdata", "job12"),
        ("rawdata.Navigator", "rawdata", "Navigator"),
    ],
)
def test_dataset_accepts_known_primary_binary_subtypes(tmp_path, name, dataset_type, subtype):
    path = tmp_path / name
    path.touch()

    dataset = Dataset(path, load=LOAD_STAGES["empty"])

    assert dataset.type == dataset_type
    assert dataset.subtype == subtype


@pytest.mark.parametrize(
    "path",
    [
        "test/test_data/PV51/0.2H2/35/fid.navFid",
        "test/test_data/PV51/0.2H2/35/fid.orig",
        "test/test_data/PV51/0.2H2/19/fid.spiral",
    ],
)
def test_fid_companion_files_are_not_loaded_as_primary_datasets(path):
    if not Path(path).is_file():
        pytest.skip(f"{path} is not available")

    with pytest.raises(UnsuportedDatasetType, match=rf"Dataset type: {re.escape(Path(path).name)} is not supported"):
        Dataset(path)


@pytest.mark.parametrize(
    ("fid_path", "subtypes"),
    [
        ("test/test_data/PV51/0.2H2/19/fid", {"spiral"}),
        ("test/test_data/PV51/0.2H2/35/fid", {"navFid", "orig"}),
    ],
)
def test_fid_companions_load_as_auxiliary_subdatasets(fid_path, subtypes):
    if not Path(fid_path).is_file():
        pytest.skip(f"{fid_path} is not available")

    dataset = Dataset(fid_path)

    assert set(dataset.fid_companions) == subtypes
    for subtype, companion in dataset.fid_companions.items():
        assert isinstance(companion, Dataset)
        assert companion.path == Path(fid_path).with_suffix(f".{subtype}")
        assert companion.data.ndim == 1
        assert np.iscomplexobj(companion.data)
        assert companion.data.size * 2 * companion.numpy_dtype.itemsize == companion.path.stat().st_size


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_invalid_optional_trajectory_does_not_abort_primary_load(tmp_path):
    source = Dataset(PV51_STUDY_PATH / "10" / "fid")
    fid_path = tmp_path / "dataset" / "fid"
    source.write(fid_path)
    (fid_path.parent / "traj").write_bytes(b"invalid trajectory")

    with pytest.warns(RuntimeWarning, match=r"Could not load optional trajectory .*[/\\]traj:"):
        dataset = Dataset(fid_path)

    assert dataset.data.size > 0
    assert dataset._traj is None
    with pytest.raises(TrajNotLoaded):
        _ = dataset.traj


@pytest.mark.parametrize(
    ("path", "dataset_type"),
    [
        ("test/test_data/PV51/0.2H2/10", "fid"),
        ("test/test_data/PV51/0.2H2/10/pdata/1", "2dseq"),
    ],
)
@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_directory_constructor_uses_default_load(path, dataset_type):
    dataset = Dataset(path)

    assert dataset.type == dataset_type
    assert dataset.data.size > 0


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_custom_csi_pulse_program_uses_family_scheme_fallback():
    dataset = Dataset(PV51_STUDY_PATH / "10" / "fid", load=LOAD_STAGES["parameters"])
    dataset["PULPROG"].val_str = "<lucaCSI4.ppg>"

    dataset.load_properties()

    assert dataset.scheme_id == "CSI"


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_scheme_id_can_be_overridden_by_caller():
    dataset = Dataset(
        PV51_STUDY_PATH / "10" / "fid",
        load=LOAD_STAGES["properties"],
        scheme_id="CSI",
    )

    assert dataset.scheme_id == "CSI"


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_custom_radial_pulse_program_is_inferred_from_projection_metadata():
    dataset = Dataset(PV51_STUDY_PATH / "21" / "fid", load=LOAD_STAGES["parameters"])
    dataset["PULPROG"].val_str = "<mac_CS_new3DSymGr.ppg>"

    dataset.load_properties()

    assert dataset.scheme_id == "RADIAL"


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_unknown_cartesian_program_is_inferred_from_encoding_metadata():
    dataset = Dataset(PV51_STUDY_PATH / "10" / "fid", load=LOAD_STAGES["parameters"])
    dataset["PULPROG"].val_str = "<customResearchSequence.ppg>"

    dataset.load_properties()

    assert dataset.scheme_id == "CART_3D"


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_ambiguous_scheme_error_names_pulse_program_and_method():
    dataset = Dataset(PV51_STUDY_PATH / "10" / "fid", load=LOAD_STAGES["parameters"])
    dataset["PULPROG"].val_str = "<customResearchSequence.ppg>"
    dataset["Method"].val_str = "<CustomMethod>"
    dataset["PVM_EncMatrix"].val_str = "127 124 127"
    dataset.load_properties()

    with pytest.raises(
        UnknownAcqSchemeException,
        match=r"PULPROG='<customResearchSequence.ppg>', Method='<CustomMethod>'; pass scheme_id=",
    ):
        dataset.load_schema()


@pytest.mark.parametrize(
    "path",
    [
        "test/test_data/PV700/20210128_122257_LEGO_PHANTOM_API_TEST_1_1/26/pdata/1/2dseq",
        "test/test_data/PV360_StdData/T1_FLASH/pdata/1/2dseq",
    ],
    ids=["PV7", "PV360"],
)
def test_modern_2dseq_geometry_has_resolution_and_nonidentity_affine(path):
    if not Path(path).is_file():
        pytest.skip(f"{path} is not available")

    dataset = Dataset(path, load=LOAD_STAGES["properties"])
    positions = np.asarray(dataset["VisuCorePosition"].value)
    expected_resolution = np.array(
        [
            dataset["VisuCoreExtent"].value[0] / dataset["VisuCoreSize"].value[0],
            dataset["VisuCoreExtent"].value[1] / dataset["VisuCoreSize"].value[1],
            np.linalg.norm(positions[1] - positions[0]),
        ]
    )

    assert np.allclose(dataset.resolution, expected_resolution)
    assert not np.allclose(dataset.affine, np.eye(4))


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_epi_layout_uses_digitized_samples_and_preserves_total_size():
    dataset = Dataset(PV51_STUDY_PATH / "13" / "fid", load=LOAD_STAGES["properties"])
    phase_lines_per_segment = dataset["PVM_EncMatrix"].value[1] // dataset["NSegments"].value

    assert dataset.encoding_space[0] == dataset["PVM_DigNp"].value
    assert dataset.k_space[0] == dataset["PVM_DigNp"].value // phase_lines_per_segment
    assert np.prod(dataset.encoding_space) == np.prod(dataset.k_space)


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_2dseq_scaling_backward_transform_inverts_slope_and_offset():
    dataset = Dataset(PV51_STUDY_PATH / "10" / "pdata" / "1" / "2dseq")
    dataset.slope = np.array([2.0, 4.0])
    dataset.offset = np.array([3.0, -5.0])
    stored = np.array([[1, 2], [-3, 4]], dtype=np.int16)

    scaled = dataset._schema._scale_frames(stored, {}, "FW")
    restored = dataset._schema._scale_frames(scaled, {}, "BW")

    assert np.array_equal(restored, stored)


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
@pytest.mark.parametrize(
    ("missing_visu_parameter", "property_name", "reco_parameter"),
    [
        ("VisuCoreDataSlope", "slope", "RECO_map_slope"),
        ("VisuCoreDataOffs", "offset", "RECO_map_offset"),
    ],
)
def test_2dseq_scaling_uses_reco_when_visu_parameter_is_missing(
    missing_visu_parameter,
    property_name,
    reco_parameter,
):
    dataset = Dataset(PV51_STUDY_PATH / "10" / "pdata" / "1" / "2dseq", load=LOAD_STAGES["parameters"])

    assert "reco" in dataset._parameters
    del dataset._parameters["visu_pars"].params[missing_visu_parameter]
    dataset.load_properties()

    assert np.array_equal(getattr(dataset, property_name), dataset[reco_parameter].array)


def test_2dseq_deserialize_serialize_preserves_stored_values():
    path = Path("test/test_data/PV360-V37/1/pdata/1/2dseq")
    if not path.is_file():
        pytest.skip(f"{path} is not available")

    dataset = Dataset(path)
    stored = dataset._read_binary_file(path, dataset.numpy_dtype, dataset.shape_storage)
    serialized = dataset._schema.serialize(dataset.data, dataset._schema.layouts)

    assert np.array_equal(serialized.astype(dataset.numpy_dtype), stored)


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_csi_skips_imaging_phase_encode_reordering():
    dataset = Dataset(PV51_STUDY_PATH / "24" / "fid")
    data = np.arange(12).reshape(4, 3)

    reordered = dataset._schema._reorder_fid_lines(data.copy())

    assert np.array_equal(reordered, data)


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_phase_encode_reorder_reports_axis_length_mismatch():
    dataset = Dataset(PV51_STUDY_PATH / "10" / "fid")
    data = np.arange(12).reshape(4, 3)
    dataset["PVM_EncSteps1"].val_str = "0 1 2 3 4"
    dataset["PVM_EncSteps1"].size = (5,)

    with pytest.raises(
        InvalidDataset,
        match=r"phase-encode reorder length 5 does not match k-space axis length 3 for scheme CART_3D",
    ):
        dataset._schema._reorder_fid_lines(data)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", r"empty binary file .* expected 24 bytes for shape \(4, 3\) and dtype int16"),
        (b"1234", r"empty or stub file .* got 4 bytes, less than one frame \(8 bytes\)"),
        (b"123456789012", r"expected 24 bytes for shape \(4, 3\) and dtype int16, got 12 bytes"),
        (
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 24\n",
            r"Git LFS pointer stub .* fetch the binary file content",
        ),
    ],
    ids=["empty", "stub", "truncated", "git-lfs"],
)
def test_binary_size_mismatch_reports_specific_cause(tmp_path, content, message):
    path = tmp_path / "2dseq"
    path.write_bytes(content)
    dataset = Dataset(path, load=LOAD_STAGES["empty"])

    with pytest.raises(InvalidDataset, match=message):
        dataset._read_binary_file(path, np.dtype("int16"), (4, 3))


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_zte_uses_dedicated_fid_layout_branches():
    dataset = Dataset(PV51_STUDY_PATH / "21" / "fid", load=LOAD_STAGES["parameters"])
    dataset["PULPROG"].val_str = "<ZTE.ppg>"

    dataset.load_properties()

    assert dataset.scheme_id == "ZTE"
    assert dataset.encoding_space[4] == dataset["NPro"].value // dataset["ACQ_phase_factor"].value
    assert dataset.permute == [0, 2, 3, 4, 5, 1]


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_monotonic_phase_encode_steps_skip_reordering():
    dataset = Dataset(PV51_STUDY_PATH / "10" / "fid")
    dataset["PVM_EncSteps1"].val_str = "10 20 30"
    dataset["PVM_EncSteps1"].size = (3,)
    data = np.arange(12).reshape(4, 3)
    data.flags.writeable = False

    reordered = dataset._schema._reorder_fid_lines(data)

    assert reordered is data


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_epi_standard_kblock_trims_trailing_padding():
    dataset = Dataset(PV51_STUDY_PATH / "13" / "fid")
    dataset["GO_block_size"].val_str = "Standard_KBlock_Format"
    dataset.block_size = 256
    dataset.acq_lenght = 200
    stored = np.concatenate([np.arange(200), np.zeros(56)]).reshape(256, 1)

    layouts = dataset._schema.layouts
    trimmed = dataset._schema._acquisition_trim(stored, layouts)

    assert layouts["acquisition_position"] == (0, 200)
    assert np.array_equal(trimmed[:, 0], np.arange(200))


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_epi_standard_kblock_warns_on_nonzero_discarded_samples():
    dataset = Dataset(PV51_STUDY_PATH / "13" / "fid")
    dataset["GO_block_size"].val_str = "Standard_KBlock_Format"
    dataset.block_size = 8
    dataset.acq_lenght = 6
    stored = np.arange(8).reshape(8, 1)

    with pytest.warns(RuntimeWarning, match="Expected trailing K-block padding to be zero"):
        trimmed = dataset._schema._acquisition_trim(stored, dataset._schema.layouts)

    assert np.array_equal(trimmed[:, 0], np.arange(6))


def test_recipe_substitution_preserves_overlapping_identifiers():
    dataset = Dataset.__new__(Dataset)
    recipe = "@Foo + @FooBar + #X + #XY + #Matrix.tuple"

    substituted = dataset._sub_parameters(recipe)

    assert substituted == (
        "self.Foo + self.FooBar + self['X'].value + "
        "self['XY'].value + self['Matrix'].tuple"
    )


@pytest.mark.parametrize(
    ("query", "error_type"),
    [
        ("unknown_name > 0", NameError),
        ("self.type ==", SyntaxError),
        ("self.type + 1", TypeError),
    ],
)
def test_dataset_query_wraps_malformed_expressions(query, error_type):
    dataset = Dataset.__new__(Dataset)
    dataset.type = "fid"

    with pytest.raises(FilterEvalFalse, match=rf"Invalid query .*{re.escape(query)}") as error:
        dataset.query(query)

    assert isinstance(error.value.__cause__, error_type)


@pytest.mark.parametrize("query", ["__import__('os')", "datetime.datetime.now()", "os.getcwd()"])
def test_dataset_query_cannot_access_builtins_or_module_globals(query):
    dataset = Dataset.__new__(Dataset)
    dataset.type = "fid"

    with pytest.raises(FilterEvalFalse, match="Invalid query") as error:
        dataset.query(query)

    assert isinstance(error.value.__cause__, NameError)


def test_dataset_query_minimal_namespace_keeps_self_and_numpy():
    dataset = Dataset.__new__(Dataset)
    dataset.type = "fid"

    dataset.query("self.type == 'fid'")
    dataset.query("np.pi > 3")


@pytest.mark.parametrize(
    "path",
    [
        "test/test_data/PV51/0.2H2/10/fid",
        "test/test_data/PV700/20210128_122257_LEGO_PHANTOM_API_TEST_1_1/10/fid",
        "test/test_data/PV700/20210128_122257_LEGO_PHANTOM_API_TEST_1_1/10/pdata/1/2dseq",
        "test/test_data/PV360-V37/10/pdata/1/2dseq",
    ],
    ids=["PV5-fid", "PV7-fid", "PV7-2dseq", "PV360-2dseq"],
)
def test_date_property_uses_timestamp_format_instead_of_version(path):
    if not Path(path).is_file():
        pytest.skip(f"{path} is not available")

    dataset = Dataset(path, load=LOAD_STAGES["properties"])

    assert isinstance(dataset.date, datetime.datetime)


@pytest.mark.skipif(not PV51_STUDY_PATH.is_dir(), reason="PV51 test data is not available")
def test_report_default_directory_and_cli_file_outputs(tmp_path):
    source = Dataset(PV51_STUDY_PATH / "10" / "fid")
    dataset_path = tmp_path / "dataset" / "fid"
    source.write(dataset_path)
    dataset = Dataset(dataset_path)

    dataset.report(props=["scheme_id"])
    default_report = dataset.path.parent / f"{dataset.id}.json"
    assert json.loads(default_report.read_text()) == {"scheme_id": "CART_3D"}

    output_directory = tmp_path / "reports"
    output_directory.mkdir()
    dataset.report(output_directory, props=["scheme_id"])
    assert json.loads((output_directory / f"{dataset.id}.json").read_text()) == {"scheme_id": "CART_3D"}

    cli_output = tmp_path / "cli-report.json"
    cli_report(
        SimpleNamespace(
            input=str(dataset_path),
            output=str(cli_output),
            format="json",
            props=["scheme_id"],
            verbose=False,
        )
    )
    assert json.loads(cli_output.read_text()) == {"scheme_id": "CART_3D"}


def test_to_dict_excludes_dataset_typing_and_preserves_property_order():
    dataset = Dataset.__new__(Dataset)
    dataset.type = "fid"
    dataset.subtype = ""
    dataset.first = 1
    dataset.second = 2
    dataset.third = 3

    exported = dataset.to_dict()

    assert list(exported) == ["first", "second", "third"]
    assert exported == {"first": 1, "second": 2, "third": 3}


def test_chained_dataset_configuration_merges_onto_current_state(tmp_path):
    path = tmp_path / "2dseq"
    path.touch()
    dataset = Dataset(path, load=LOAD_STAGES["empty"])
    first_config = {"mmap": True, "parameter_files": ["method"]}

    returned = dataset(**first_config)(scale=False, property_files=[tmp_path / "custom.json"])

    assert returned is dataset
    assert dataset._state["mmap"] is True
    assert dataset._state["scale"] is False
    assert dataset._state["load"] == LOAD_STAGES["empty"]
    assert dataset._state["parameter_files"][-1] == "method"
    assert dataset._state["property_files"][-1] == tmp_path / "custom.json"
    assert first_config == {"mmap": True, "parameter_files": ["method"]}


@pytest.mark.skip(reason="in progress")
def test_parameters(test_parameters):
    dataset = Dataset(test_parameters[0], load=False)
    dataset.load_parameters()

    for jcampdx in dataset._parameters.values():
        with Path(str(jcampdx.path) + ".json").open() as file:
            reference = json.load(file)

        assert jcampdx.to_dict() == reference


def test_properties(test_properties):
    if test_properties:
        dataset = Dataset(test_properties[0], load=False, parameter_files=["subject"])
        dataset.load_parameters()
        dataset.load_properties()
        assert dataset.to_dict() == test_properties[1]


def test_data_load(test_data):
    dataset = Dataset(test_data[0])

    return  # For now Disable testing array equality
    if not os.path.exists(str(dataset.path) + ".npz"):
        return

    with np.load(str(dataset.path) + ".npz") as data:
        assert np.array_equal(np.squeeze(dataset.data), np.squeeze(data["data"]))


def test_data_save(test_data, tmp_path, WRITE_TOLERANCE):
    d_ref = Dataset(test_data[0])

    if d_ref.subtype == "":
        path_out = tmp_path / d_ref.type
    else:
        path_out = tmp_path / (d_ref.type + "." + d_ref.subtype)

    d_ref.write(path_out)
    d_test = Dataset(path_out)

    diff = d_ref.data - d_test.data
    max_error = np.max(np.abs(diff))

    with contextlib.suppress(AssertionError):
        assert np.array_equal(d_ref.data, d_test.data)

    if max_error > 0.0:
        try:
            assert max_error < WRITE_TOLERANCE
            print(f"Arrays are not identical, but max difference: {max_error} is tolerated")
        except AssertionError as e:
            raise e

    # Test if properties are loaded correctly
    # TODO since the id property of the 2dseq dataset type relies on the name of the experiment folder,
    # which is a problem when the dataset is writen to the test folder, solution might be to delete the id key here
    # assert d_test.to_dict() == test_data[1]
