from pathlib import Path

import numpy as np
import pytest

from brukerapi.dataset import Dataset
from brukerapi.exceptions import InvalidDataset, UnknownAcqSchemeException
from test.synthetic import Verbatim, write_binary, write_jcampdx

RAWDATA_JOB_PATHS = sorted(Path("test/test_data").rglob("rawdata.job*"))


def _raw_complex_stream(dataset):
    stored = dataset._read_binary_file(dataset.path, dataset.numpy_dtype, dataset.shape_storage)
    return stored[0::2, ...] + 1j * stored[1::2, ...]


def _rawdata_with_kspace(test_data_root, predicate):
    for path in sorted(test_data_root.rglob("rawdata.job*")):
        if not any(part.startswith("PV360") for part in path.parts):
            continue
        dataset = Dataset(path)
        try:
            k_space = dataset.to_kspace()
        except (InvalidDataset, UnknownAcqSchemeException):
            continue
        if predicate(dataset, k_space):
            return dataset, k_space
    pytest.skip("The selected corpus has no rawdata job with the required Cartesian layout")


def _epi_rawdata(test_data_root):
    for path in sorted(test_data_root.rglob("rawdata.job*")):
        if not any(part.startswith("PV360") for part in path.parts):
            continue
        dataset = Dataset(path)
        try:
            dataset.to_kspace()
        except UnknownAcqSchemeException as error:
            if "is EPI" in str(error):
                return dataset
        except InvalidDataset:
            continue
    pytest.skip("The selected corpus has no EPI rawdata job")


@pytest.mark.parametrize("rawdata_path", RAWDATA_JOB_PATHS, ids=[str(path) for path in RAWDATA_JOB_PATHS])
def test_rawdata_job_loads_directly(rawdata_path):
    dataset = Dataset(rawdata_path)
    with pytest.warns(FutureWarning, match="format-dependent legacy semantics"):
        legacy_data = dataset.data

    assert dataset.type == "rawdata"
    assert dataset.subtype == rawdata_path.suffix.removeprefix(".")
    assert legacy_data.shape == dataset._schema.layouts["raw"]
    assert legacy_data.size > 0
    assert np.array_equal(dataset.raw, np.transpose(legacy_data, (0, 2, 1)))


def test_pv360_cartesian_rawdata_job_converts_to_ordered_k_space(test_data_root):
    dataset, k_space = _rawdata_with_kspace(test_data_root, lambda _dataset, data: data.ndim == 5)

    assert np.array_equal(dataset.kspace, k_space)
    assert k_space.shape[-1] == dataset.channels
    assert k_space.size == dataset.raw.size

    bart = dataset.to_kspace(bart=True)
    assert bart.ndim == 16
    assert bart.shape[3] == dataset.channels
    assert bart.size == k_space.size


def test_pv360_3d_cartesian_rawdata_job_converts_to_k_space(test_data_root):
    dataset, k_space = _rawdata_with_kspace(
        test_data_root,
        lambda dataset, data: data.ndim == 5 and dataset._parameter_value("ACQ_dim") == 3,
    )

    assert k_space.shape[-1] == dataset.channels
    assert k_space.size == dataset.raw.size


def test_pv360_self_gated_rawdata_exposes_acquired_cardiac_frames(test_data_root):
    dataset, k_space = _rawdata_with_kspace(
        test_data_root,
        lambda _dataset, data: data.ndim == 6 and data.shape[2] > 1 and data.shape[3] > 1,
    )

    assert np.array_equal(dataset.kspace, k_space)
    assert k_space.shape[-1] == dataset.channels
    assert k_space.size == dataset.raw.size

    bart = dataset.to_kspace(bart=True)
    assert bart.ndim == 16
    assert bart.shape[3] == dataset.channels
    assert bart.size == k_space.size


def test_pv360_epi_rawdata_job_requires_an_acquisition_specific_reader(test_data_root):
    dataset = _epi_rawdata(test_data_root)

    with pytest.raises(UnknownAcqSchemeException, match="is EPI"):
        dataset.to_kspace()


def test_to_kspace_nests_the_object_loop_inside_the_phase_loop(tmp_path):
    """Spec 5.2, "Acquisition loop nesting (default)":

        NS > ACQ_phase_factor > NSLICES > NI or NSLICES > NA
           > ACQ_size[1]/ACQ_phase_factor > ACQ_size[1] > ACQ_size[2] > NAE > NR

    The object level sits INSIDE the phase-encode-group level, so consecutive
    scans step the object first. Reading it the other way round interleaves the
    slice and phase-encode indices of every 2-D multi-slice job, silently: the
    array keeps its shape and dtype.
    """
    readout, phase, objects = 2, 3, 2
    # scan s = object + NI*group, and each scan carries 1000*object + group so a
    # misplaced sample is visible
    scans = [1000 * obj + group for group in range(phase) for obj in range(objects)]
    samples = np.zeros((2 * readout, 1, len(scans)), dtype="<i4")
    for index, value in enumerate(scans):
        samples[0::2, 0, index] = value

    experiment = tmp_path / "1"
    write_jcampdx(
        experiment / "acqp",
        {
            "ACQ_sw_version": ["<PV-360.3.6>"],
            "ACQ_word_size": "_32_BIT",
            "BYTORDA": "little",
            "ACQ_dim": 2,
            "ACQ_dim_desc": Verbatim("( 2 )\nSpatial Spatial"),
            # spec 13.1: ACQ_size[0] need not equal the job scan size
            "ACQ_size": np.array([8, phase]),
            "ACQ_phase_factor": 1,
            "NI": objects,
            "NR": 1,
            "ACQ_jobs": Verbatim(f"( 1 )\n({2 * readout}, 1, 0, {len(scans)}, 101, 5000, {len(scans)}, 1, <job0>)"),
        },
    )
    write_jcampdx(experiment / "method", {"PVM_EncNReceivers": 1, "PVM_EncMatrix": np.array([readout, phase])})
    write_binary(experiment / "rawdata.job0", samples, np.dtype("<i4"))

    k_space = Dataset(experiment / "rawdata.job0").to_kspace()

    assert k_space.shape == (readout, phase, objects, 1, 1)
    for obj in range(objects):
        for group in range(phase):
            assert np.all(k_space[:, group, obj, 0, 0].real == 1000 * obj + group)
