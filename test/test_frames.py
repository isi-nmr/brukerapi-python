"""Frame layout and frame-group metadata: FILE_FORMAT.md 7.2 and 7.4.

Synthetic datasets only -- see test/synthetic.py.
"""

import numpy as np

from brukerapi.dataset import Dataset
from test.synthetic import Verbatim, stacked_positions, write_2dseq

IMAGE = np.arange(12, dtype="int16").reshape(3, 4)


def transposed_dataset(tmp_path, size=(3, 4), image=None, **state):
    """Two frames of the same image, the second stored with its axes exchanged."""
    image = IMAGE if image is None else image
    stored = np.concatenate([image.flatten(order="F"), image.T.flatten(order="F")])
    path = write_2dseq(
        tmp_path / "5" / "pdata" / "1",
        size=size,
        frame_groups=(("FG_SLICE", 2),),
        positions=stacked_positions((-20.0, -20.0, 0.0), (0.0, 0.0, 1.0), 2),
        data=stored.reshape(tuple(size) + (2,), order="F"),
        extra={"VisuCoreTransposition": np.array([0, 1])},
    )
    return Dataset(path, **state)


def test_a_transposed_frame_is_read_in_its_stored_shape(tmp_path):
    """Spec 7.2: a nonzero VisuCoreTransposition exchanges two stored dimensions.

    Reshaping such a frame with VisuCoreSize interleaves its rows, which turns
    an image into diagonal-stripe noise without any error.
    """
    dataset = transposed_dataset(tmp_path)

    assert np.array_equal(dataset.data[..., 0], IMAGE)
    assert np.array_equal(dataset.data[..., 1], IMAGE)


def test_a_square_transposed_frame_is_delivered_unchanged(tmp_path):
    """An exchange between two equal-length dimensions does not move any pixel.

    Those frames measure as already consistent with VisuCoreOrientation, so
    transposing them would introduce the error instead of fixing one.
    """
    image = np.arange(16, dtype="int16").reshape(4, 4)
    dataset = transposed_dataset(tmp_path, size=(4, 4), image=image)

    assert np.array_equal(dataset.data[..., 0], image)
    assert np.array_equal(dataset.data[..., 1], image.T)


def test_writing_restores_the_stored_frame_layout(tmp_path):
    dataset = transposed_dataset(tmp_path)
    original = dataset.path.read_bytes()

    dataset.write(tmp_path / "out" / "2dseq")

    assert (tmp_path / "out" / "2dseq").read_bytes() == original
    assert np.array_equal(Dataset(tmp_path / "out" / "2dseq").data, dataset.data)


def test_random_access_indexes_transposition_by_absolute_frame(tmp_path):
    dataset = transposed_dataset(tmp_path, mmap=True)

    assert np.array_equal(dataset.data[:, :, 1], IMAGE)


def test_frame_group_values_follow_the_descriptor_window(tmp_path):
    """Spec 7.4: the (valsStart, valsCnt) of a descriptor owns its dependents.

    VisuGroupDepVals[k][1] is a start index into the dependent *parameter*
    array, not an index into VisuFGOrderDesc; reading it as the latter put
    every dependent parameter on frame group 0, and a size-matching rescue
    hides that only while the two groups have different lengths.
    """
    positions = stacked_positions((-20.0, -20.0, -2.0), (0.0, 0.0, 2.0), 3)
    path = write_2dseq(
        tmp_path / "27" / "pdata" / "1",
        frame_groups=(("FG_ECHO", 3, 0, 1), ("FG_SLICE", 3, 1, 2)),
        positions=positions,
        extra={
            "VisuAcqEchoTime": np.array([5.0, 10.0, 15.0]),
            "VisuGroupDepVals": Verbatim("( 3 )\n(<VisuAcqEchoTime>, 0) (<VisuCoreOrientation>, 0) (<VisuCorePosition>, 0)"),
        },
    )
    dataset = Dataset(path)

    values = dataset.frame_group_values

    assert dataset.dim_type == ["spatial", "spatial", "FG_ECHO", "FG_SLICE"]
    assert values["VisuAcqEchoTime"].shape == (1, 1, 3, 1)
    assert values["VisuCorePosition"].shape == (1, 1, 1, 3, 3)
    assert values["VisuCoreOrientation"].shape == (1, 1, 1, 3, 9)
    assert np.array_equal(np.squeeze(values["VisuCorePosition"]), positions)
