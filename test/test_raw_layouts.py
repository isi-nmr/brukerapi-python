"""Raw acquisition layouts: FILE_FORMAT.md 3.1 (fid storage), 5.2 (loop counters), 6.3.

The datasets are synthetic, built by test/synthetic.py from the parameter values
of real acquisitions, shrunk so the whole binary fits in a few hundred words.
"""

import numpy as np

from brukerapi.dataset import LOAD_STAGES, Dataset
from test.synthetic import Verbatim, write_fid

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
