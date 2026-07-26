"""Raw acquisition layouts: FILE_FORMAT.md 3.1 (fid storage), 5.2 (loop counters), 6.3.

The datasets are synthetic, built by test/synthetic.py from the parameter values
of real acquisitions, shrunk so the whole binary fits in a few hundred words.
"""

import numpy as np
import pytest

from brukerapi.dataset import LOAD_STAGES, Dataset
from test.synthetic import Verbatim, write_binary, write_fid, write_jcampdx

# pv6 EPSI, shrunk: 6 read points, 4 phase steps, 2 segments, 4 spectral points
# per read line -- the real scan is 96 x 64 x 4 x 64.
EPSI_READ = 6
EPSI_PHASE = 4
EPSI_SEGMENTS = 2
EPSI_SPECTRAL_PER_SEGMENT = 4
EPSI_DIGITIZED = EPSI_READ * EPSI_SPECTRAL_PER_SEGMENT


def epsi_dataset(tmp_path, **state):
    path = write_fid(
        tmp_path / "34",
        acqp={
            "GO_block_size": "continuous",
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "AQ_mod": "qdig",
            "ACQ_dim": 3,
            "ACQ_dim_desc": Verbatim("( 3 )\nSpatial Spectroscopic Spatial"),
            "ACQ_size": np.array([2 * EPSI_DIGITIZED, EPSI_SEGMENTS, EPSI_PHASE]),
            "ACQ_phase_factor": 1,
            "ACQ_obj_order": 0,
            "PULPROG": "<EPSI.ppg>",
            "NI": 1,
            "NR": 1,
        },
        method={
            "Method": "<Bruker:EPSI>",
            "NSegments": EPSI_SEGMENTS,
            "PVM_DigNp": EPSI_DIGITIZED,
            "PVM_EncMatrix": np.array([EPSI_READ, EPSI_PHASE]),
            "PVM_EncNReceivers": 1,
            "PVM_EncSteps1": np.arange(EPSI_PHASE) - EPSI_PHASE // 2,
        },
        blocks=EPSI_SEGMENTS * EPSI_PHASE,
    )
    return Dataset(path, **state)


def test_epsi_accounts_for_every_acquired_sample(tmp_path):
    """Spec 3.1: with GO_block_size=continuous the whole block is digitized data.

    The EPSI layout divided the per-block sample count by NSegments while
    block_count already multiplied by it, so all but the last segment of every
    block was discarded -- 75 % of the file on a four-segment scan.
    """
    dataset = epsi_dataset(tmp_path, load=LOAD_STAGES["properties"])

    assert dataset.acq_length == dataset.block_size
    assert dataset.acq_length * dataset.block_count == dataset.path.stat().st_size // dataset.numpy_dtype.itemsize


def test_epsi_k_space_has_a_full_spectral_axis(tmp_path):
    dataset = epsi_dataset(tmp_path)

    # Spec 6.3: the reconstruction input is read x phase x spectral points.
    assert tuple(dataset.k_space) == (EPSI_READ, EPSI_PHASE, EPSI_SEGMENTS * EPSI_SPECTRAL_PER_SEGMENT, 1)
    assert dataset.data.shape == (EPSI_READ, EPSI_PHASE, EPSI_SEGMENTS * EPSI_SPECTRAL_PER_SEGMENT, 1)
    assert dataset.dim_type == [
        "k_space_encode_step_0",
        "k_space_encode_step_1",
        "k_space_encode_step_2",
        "channel",
    ]


def test_epsi_keeps_every_stored_complex_sample(tmp_path):
    dataset = epsi_dataset(tmp_path)
    stored = np.fromfile(dataset.path, dtype=dataset.numpy_dtype)
    expected = stored[0::2] + 1j * stored[1::2]

    assert dataset.data.size == expected.size
    assert np.array_equal(np.sort_complex(dataset.data.reshape(-1)), np.sort_complex(expected))


def test_fid_uses_pdata_one_reco_unless_a_different_one_is_selected(tmp_path):
    """`pdata/1` is conventional; `reco_path` selects another reconstruction."""
    path = write_fid(
        tmp_path / "30",
        acqp={
            "GO_block_size": "continuous",
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "ACQ_dim": 2,
            "ACQ_dim_desc": Verbatim("( 2 )\nSpatial Spatial"),
            "ACQ_size": np.array([8, 2]),
            "ACQ_phase_factor": 1,
            "ACQ_scan_size": "ACQ_phase_factor_scans",
            "PULPROG": "<FLASH.ppg>",
            "NI": 1,
            "NR": 1,
        },
        method={"Method": "<Bruker:FLASH>", "PVM_DigNp": 4, "PVM_EncMatrix": np.array([4, 2]), "PVM_EncNReceivers": 1},
        blocks=2,
    )
    write_jcampdx(path.parent / "pdata" / "1" / "reco", {"RECO_inp_order": "NO_REORDERING"})
    reco = write_jcampdx(path.parent / "pdata" / "2" / "reco", {"RECO_inp_order": "REV_ALT_ROWS"})

    default = Dataset(path, load=LOAD_STAGES["properties"])
    default.load_schema()
    assert "reco" in default.parameters
    assert default["RECO_inp_order"].value == "NO_REORDERING"
    assert not default._schema.mirror_odd_lines

    dataset = Dataset(path, reco_path=reco, load=LOAD_STAGES["properties"])
    dataset.load_schema()
    assert "reco" in dataset.parameters
    with pytest.warns(RuntimeWarning, match="ACQ_scan_size=.*disagrees with scheme_id"):
        assert dataset._schema.continuous_train
    with pytest.warns(RuntimeWarning, match="RECO_inp_order=.*disagrees with scheme_id"):
        assert dataset._schema.mirror_odd_lines

    lines = np.arange(8).reshape((4, 2), order="F")
    expected = np.column_stack((lines[:, 0], lines[::-1, 1]))
    assert np.array_equal(dataset._schema._mirror_odd_lines(lines), expected)


def kblock_fid(tmp_path, acqp, method, *, blocks, samples_per_block):
    """A Standard_KBlock_Format fid: real samples first, then zero padding."""
    block_size = int(np.ceil(int(np.atleast_1d(acqp["ACQ_size"])[0]) * int(method["PVM_EncNReceivers"]) * 4 / 1024.0) * 1024 // 4)
    stored = np.zeros((block_size, blocks), dtype="<i4")
    stored[:samples_per_block, :] = np.arange(1, samples_per_block * blocks + 1, dtype="<i4").reshape((samples_per_block, blocks), order="F")
    return write_fid(tmp_path, acqp, method, data=stored.flatten(order="F"))


def test_3d_radial_labels_every_stored_axis(tmp_path):
    """Spec 5.2: NI counts objects and NR repetitions, and both are stored axes.

    The 3-D radial layout declared a six-axis k-space but only five labels, so
    NI was labelled `repetition`, NR was labelled `channel`, the receiver axis
    was unlabelled, and to_kspace(bart=True) refused the array outright.
    """
    projections, partitions = 3, 2
    path = kblock_fid(
        tmp_path / "3",
        acqp={
            "GO_block_size": "Standard_KBlock_Format",
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "ACQ_dim": 3,
            "ACQ_dim_desc": Verbatim("( 3 )\nSpatial Spatial Spatial"),
            "ACQ_size": np.array([8, projections * partitions, 1]),
            "ACQ_phase_factor": 1,
            "PULPROG": "<UTE3D.ppg>",
            "NPro": projections,
            "NI": 1,
            "NR": 1,
        },
        method={"Method": "<Bruker:UTE3D>", "PVM_EncMatrix": np.array([4, 4, partitions]), "PVM_EncNReceivers": 1},
        blocks=projections * partitions,
        samples_per_block=8,
    )

    dataset = Dataset(path)

    assert dataset.dim_type == [
        "k_space_encode_step_0",
        "k_space_encode_step_1",
        "k_space_encode_step_2",
        "object",
        "repetition",
        "channel",
    ]
    assert len(dataset.dim_type) == dataset.data.ndim
    assert dataset.to_kspace(bart=True).ndim == 16


def test_spectroscopy_labels_objects_and_repetitions(tmp_path):
    """Spec 5.2: NI is objects per repetition, NR repetitions -- NA is not stored.

    Labelling NR `average` sends a dynamic series to BART's average dimension,
    where a caller that averages over it destroys the series.
    """
    objects, repetitions = 2, 3
    path = write_fid(
        tmp_path / "32",
        acqp={
            "GO_block_size": "continuous",
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "ACQ_dim": 1,
            "ACQ_dim_desc": "Spectroscopic",
            "ACQ_size": np.array([8]),
            "PULPROG": "<PRESS.ppg>",
            "NI": objects,
            "NR": repetitions,
            "NA": 4,
        },
        method={"Method": "<Bruker:PRESS>", "PVM_DigNp": 4, "PVM_EncNReceivers": 1},
        blocks=objects * repetitions,
    )

    dataset = Dataset(path)
    bart = dataset.to_kspace(bart=True)

    assert dataset.dim_type == ["k_space_encode_step_0", "object", "repetition"]
    assert dataset.data.shape == (4, objects, repetitions)
    assert [(axis, size) for axis, size in enumerate(bart.shape) if size > 1] == [(0, 4), (9, repetitions), (13, objects)]


def test_field_map_labels_echoes_and_counts_repetitions(tmp_path):
    """Spec 5.2: PVM_NEchoImages is an echo axis and NR a stored repetition axis.

    The echo axis was labelled `repetition` (so BART received echoes on its
    time dimension), and block_count carried no NR factor at all -- masked by
    every corpus field map having NR = 1.
    """
    echoes, phase, partitions, repetitions = 2, 3, 2, 2
    path = kblock_fid(
        tmp_path / "31",
        acqp={
            "GO_block_size": "Standard_KBlock_Format",
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "ACQ_dim": 3,
            "ACQ_dim_desc": Verbatim("( 3 )\nSpatial Spatial Spatial"),
            "ACQ_size": np.array([8, phase, partitions]),
            "ACQ_phase_factor": 1,
            "PULPROG": "<FieldMap.ppg>",
            "NI": echoes,
            "NR": repetitions,
        },
        method={
            "Method": "<Bruker:FieldMap>",
            "PVM_EncMatrix": np.array([4, phase, partitions]),
            "PVM_NEchoImages": echoes,
            "PVM_EncNReceivers": 1,
        },
        blocks=phase * partitions * echoes * repetitions,
        samples_per_block=8,
    )

    dataset = Dataset(path)

    assert dataset.dim_type == [
        "k_space_encode_step_0",
        "k_space_encode_step_1",
        "k_space_encode_step_2",
        "echo",
        "repetition",
        "channel",
    ]
    assert dataset.data.shape == (4, phase, partitions, echoes, repetitions, 1)
    assert [(axis, size) for axis, size in enumerate(dataset.to_kspace(bart=True).shape) if size > 1] == [
        (0, 4),
        (1, phase),
        (2, partitions),
        (9, repetitions),
        (10, echoes),
    ]


def test_a_fid_companion_is_shaped_into_acquisitions(tmp_path):
    """Spec 3.5: fid.spiral / fid.navFid hold whole acquisitions of PVM_DigNp
    complex points; returning a flat vector makes the caller rediscover that."""
    digitized, acquisitions, phase = 4, 3, 2
    experiment = tmp_path / "35"
    path = kblock_fid(
        experiment,
        acqp={
            "GO_block_size": "Standard_KBlock_Format",
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "ACQ_dim": 2,
            "ACQ_dim_desc": Verbatim("( 2 )\nSpatial Spatial"),
            "ACQ_size": np.array([2 * digitized, phase]),
            "ACQ_phase_factor": 1,
            "PULPROG": "<FLASH.ppg>",
            "NI": 1,
            "NR": 1,
        },
        method={"Method": "<Bruker:FLASH>", "PVM_DigNp": digitized, "PVM_EncMatrix": np.array([digitized, phase]), "PVM_EncNReceivers": 1},
        blocks=phase,
        samples_per_block=2 * digitized,
    )
    write_binary(experiment / "fid.navFid", np.arange(1, 2 * digitized * acquisitions + 1, dtype="<i4"), np.dtype("<i4"))

    companion = Dataset(path).fid_companions["navFid"]

    assert companion.dim_type == ["sample", "acquisition"]
    assert companion.data.shape == (digitized, acquisitions)
