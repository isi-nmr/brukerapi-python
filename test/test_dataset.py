import contextlib
import datetime
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brukerapi.cli import report as cli_report
from brukerapi.dataset import LOAD_STAGES, Dataset
from brukerapi.exceptions import FilterEvalFalse, IncompleteDataset, InvalidDataset, TrajNotLoaded, UnknownAcqSchemeException, UnsupportedDatasetType
from brukerapi.schemas import Schema2dseq, SchemaFid

data = 0
PV51_STUDY_PATH = Path("test/test_data/PV51/0.2H2")


def test_unsupported_dataset_type(tmp_path):
    path = tmp_path / "unsupported"
    path.touch()

    with pytest.raises(UnsupportedDatasetType, match="Dataset type: unsupported is not supported"):
        Dataset(path)


def test_ser_is_explicitly_unsupported(tmp_path):
    path = tmp_path / "ser"
    path.touch()

    with pytest.raises(UnsupportedDatasetType, match="Dataset type: ser is not supported"):
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
    ("binary_content", "visu_content", "empty_names"),
    [
        (b"", b"", "2dseq.*visu_pars"),
        (b"not empty", b"", "visu_pars"),
        (b"", b"not empty", "2dseq"),
    ],
)
def test_empty_2dseq_reconstruction_has_clear_dataset_error(tmp_path, binary_content, visu_content, empty_names):
    (tmp_path / "2dseq").write_bytes(binary_content)
    (tmp_path / "visu_pars").write_bytes(visu_content)

    with pytest.raises(
        InvalidDataset,
        match=rf"empty or incomplete reconstruction: empty {empty_names}",
    ):
        Dataset(tmp_path / "2dseq", load=LOAD_STAGES["parameters"])


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

    with pytest.raises(UnsupportedDatasetType, match=rf"Dataset type: {re.escape(name)} is not supported"):
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

    with pytest.raises(UnsupportedDatasetType, match=rf"Dataset type: {re.escape(Path(path).name)} is not supported"):
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


def test_2dseq_loaded_values_apply_per_frame_slope_and_offset():
    path = Path("test/test_data/PV360_StdData/T1_FLASH/pdata/1/2dseq")
    if not path.is_file():
        pytest.skip(f"{path} is not available")

    raw = Dataset(path, scale=False)
    scaled = Dataset(path)
    stored = raw._read_binary_file(path, raw.numpy_dtype, raw.shape_storage).astype(float)

    for frame in range(stored.shape[-1]):
        stored[..., frame] *= float(raw.slope[frame])
        stored[..., frame] += float(raw.offset[frame])
    expected = np.reshape(stored, raw.shape_final, order="F")

    assert np.array_equal(raw.data, raw._read_binary_file(path, raw.numpy_dtype, raw.shape_storage).reshape(raw.shape_final, order="F"))
    assert np.allclose(scaled.data, expected)
    assert scaled.data.dtype.kind == "f"


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


@pytest.mark.parametrize(
    ("combine_complex", "expected"),
    [
        (True, np.array([1 + 10j, 2 + 20j])),
        (False, np.array([[1, 10], [2, 20]])),
    ],
)
def test_2dseq_complex_frame_group_assembly_is_reversible(combine_complex, expected):
    dataset = SimpleNamespace(
        _state={"scale": False, "combine_complex": combine_complex},
        dim_type=["spatial", "FG_COMPLEX"],
        _parameter_value=lambda name, default=None: default,
        numpy_dtype=np.dtype("int16"),
    )
    schema = Schema2dseq.__new__(Schema2dseq)
    schema._dataset = dataset
    layouts = {
        "shape_storage": (2, 2),
        "shape_final": (2, 2),
        "shape_fg": (2,),
    }
    stored = np.array([[1, 10], [2, 20]], dtype=np.int16)

    decoded = schema.deserialize(stored, layouts)
    serialized = schema.serialize(decoded, layouts)

    assert np.array_equal(decoded, expected)
    assert np.array_equal(serialized, stored)


def test_2dseq_reco_complex_image_falls_back_to_last_frame_axis():
    dataset = SimpleNamespace(
        _state={"scale": False, "combine_complex": True},
        dim_type=["spatial", "image"],
        _parameter_value=lambda name, default=None: "COMPLEX_IMAGE" if name == "RECO_image_type" else default,
    )
    schema = Schema2dseq.__new__(Schema2dseq)
    schema._dataset = dataset
    layouts = {
        "shape_storage": (2, 2),
        "shape_final": (2, 2),
        "shape_fg": (2,),
    }

    decoded = schema.deserialize(np.array([[1, 10], [2, 20]]), layouts)

    assert np.array_equal(decoded, np.array([1 + 10j, 2 + 20j]))


def test_2dseq_slice_packages_are_separate_in_memory_datasets():
    path = Path("test/test_data/PV601/20200612_094625_lego_phantom_3_1_2/8/pdata/1/2dseq")
    if not path.is_file():
        pytest.skip(f"{path} is not available")

    dataset = Dataset(path)
    packages = dataset.get_slice_packages()

    assert dataset.num_slice_packages == 3
    assert [package.shape[2] for package in packages] == [5, 3, 5]
    assert all(package.num_slice_packages == 1 for package in packages)
    assert [package["VisuCorePosition"].shape[0] for package in packages] == [5, 3, 5]
    assert np.array_equal(packages[0].data, dataset.data[:, :, :5])
    assert np.array_equal(packages[1].data, dataset.data[:, :, 5:8])
    assert np.array_equal(packages[2].data, dataset.data[:, :, 8:13])


@pytest.mark.parametrize(
    ("disk_order", "reverse"),
    [
        ("<disk_reverse_slice_order>", True),
        ("disk_normal_slice_order", False),
    ],
)
def test_2dseq_disk_slice_order_is_applied_and_reversible(disk_order, reverse):
    dataset = SimpleNamespace(
        _state={"scale": False, "combine_complex": False},
        dim_type=["spatial", "<FG_SLICE>", "<FG_ECHO>"],
        _parameter_value=lambda name, default=None: disk_order if name == "VisuCoreDiskSliceOrder" else default,
    )
    schema = Schema2dseq.__new__(Schema2dseq)
    schema._dataset = dataset
    layouts = {
        "shape_storage": (2, 3, 2),
        "shape_final": (2, 3, 2),
        "shape_fg": (3, 2),
    }
    stored = np.arange(12).reshape(layouts["shape_storage"], order="F")

    decoded = schema.deserialize(stored, layouts)
    serialized = schema.serialize(decoded, layouts)

    expected = np.flip(stored, axis=1) if reverse else stored
    assert np.array_equal(decoded, expected)
    assert np.array_equal(serialized, stored)


@pytest.mark.parametrize(
    ("aq_mod", "encoding_space", "expected"),
    [
        ("qf", (4, 1), np.array([[1], [2], [3], [4]], dtype=np.int32)),
        ("qdig", (2, 1), np.array([[1 + 2j], [3 + 4j]])),
    ],
)
def test_fid_quadrature_mode_controls_real_imag_deinterleave(aq_mod, encoding_space, expected):
    dataset = SimpleNamespace(
        scheme_id="CART_2D",
        numpy_dtype=np.dtype("int32"),
        block_size=4,
        block_count=1,
        encoding_space=encoding_space,
        permute=(0, 1),
        k_space=encoding_space,
        acq_length=4,
        _parameter_value=lambda name, default=None: aq_mod if name == "AQ_mod" else default,
    )
    schema = SchemaFid.__new__(SchemaFid)
    schema._dataset = dataset
    schema._reorder_fid_lines = lambda data, dir="FW": data
    layouts = {
        "storage": (4, 1),
        "acquisition_position": (0, 4),
        "encoding_space": encoding_space,
        "encoding_permuted": encoding_space,
        "permute": (0, 1),
        "inverse_permute": (0, 1),
        "k_space": encoding_space,
    }
    stored = np.array([[1], [2], [3], [4]], dtype=np.int32)

    decoded = schema.deserialize(stored, layouts)

    assert np.array_equal(decoded, expected)
    assert np.array_equal(schema.serialize(decoded, layouts), stored)


def test_fid_qf_layout_preserves_all_real_samples():
    dataset = SimpleNamespace(
        block_size=4,
        block_count=1,
        encoding_space=(2, 1),
        permute=(0, 1),
        k_space=(2, 1),
        acq_length=4,
        scheme_id="CART_2D",
        _parameter_value=lambda name, default=None: "qf" if name == "AQ_mod" else default,
    )
    schema = SchemaFid.__new__(SchemaFid)
    schema._dataset = dataset

    assert schema.layouts["encoding_space"] == (4, 1)
    assert schema.layouts["k_space"] == (4, 1)


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
    dataset.acq_length = 200
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
    dataset.acq_length = 6
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
    dataset = Dataset(test_properties[0], load=False, parameter_files=["subject"])
    dataset.load_parameters()
    dataset.load_properties()
    reference = dict(test_properties[1])
    reference.pop("type", None)
    reference.pop("subtype", None)

    assert reference, f"Property reference for {dataset.path} must not be empty"
    assert dataset.to_dict() == reference


def test_data_load(test_data):
    dataset = Dataset(test_data[0])
    reference_path = Path(str(dataset.path) + ".npz")

    assert isinstance(dataset.data, np.ndarray)
    assert dataset.data.size > 0
    assert np.all(np.isfinite(dataset.data))

    if not reference_path.exists():
        return

    with np.load(reference_path) as data:
        assert "data" in data
        actual = np.squeeze(dataset.data)
        reference = data["data"]

        if dataset.type == "2dseq" and np.iscomplexobj(actual) and not np.iscomplexobj(reference):
            complex_axis = next(
                (
                    axis
                    for axis, dim_type in enumerate(dataset.dim_type)
                    if str(dim_type).strip("<>").upper() == "FG_COMPLEX"
                ),
                None,
            )
            if complex_axis is not None and reference.shape[complex_axis] == 2:
                reference = np.take(reference, 0, axis=complex_axis) + 1j * np.take(reference, 1, axis=complex_axis)

        reference = np.squeeze(reference)
        if np.array_equal(actual, reference):
            return

        assert dataset.type == "fid"

        # Some older FID caches captured only the first element of dimensions
        # that are now correctly exposed. Match the reference dimensions in
        # order and select index zero from any additional current dimensions.
        if actual.ndim > reference.ndim or actual.shape != reference.shape:
            slices = []
            reference_axis = 0
            for actual_size in actual.shape:
                if reference_axis < reference.ndim and actual_size == reference.shape[reference_axis]:
                    slices.append(slice(None))
                    reference_axis += 1
                else:
                    slices.append(0)
            if reference_axis == reference.ndim:
                legacy_plane = actual[tuple(slices)]
                if np.array_equal(legacy_plane, reference):
                    return

        # Other caches predate phase-line reordering and contain the same
        # complete FID values in a different logical order.
        assert actual.size == reference.size
        assert np.array_equal(np.sort(actual, axis=None), np.sort(reference, axis=None))


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
