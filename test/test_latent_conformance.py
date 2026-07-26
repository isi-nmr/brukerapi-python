"""Paths the review corpus never exercises, proven from the specification.

None of these fires on any Bruker dataset available to the project, so each one
is reproduced synthetically: a MINIMAL_INSTANCE visu_pars without a slope, a
legacy d3proc-only reconstruction, an NAE>1 job, a 64-bit float job, and the
JCAMP-DX shapes 2.1/2.2 warn about.
"""

import numpy as np
import pytest

from brukerapi.dataset import LOAD_STAGES, Dataset
from brukerapi.jcampdx import JCAMPDX
from brukerapi.utils import transposed_size
from test.synthetic import Verbatim, write_2dseq, write_binary, write_jcampdx


def test_the_reco_scaling_fallback_is_the_inverse_of_the_visu_one(tmp_path):
    """Spec 3.4: real = pixel / RECO_map_slope + RECO_map_offset.

    So VisuCoreDataSlope = 1 / RECO_map_slope; multiplying by RECO_map_slope
    instead makes every value slope^2 too small.
    """
    records = {"VisuCoreDataMin": np.zeros(2), "VisuCoreDataMax": np.full(2, 10.0)}
    path = write_2dseq(
        tmp_path / "1" / "pdata" / "1",
        frame_groups=(("FG_SLICE", 2),),
        data=np.full((4, 4, 2), 100, dtype="int16"),
    )
    # a derived visu_pars that carries no scaling of its own
    text = path.parent.joinpath("visu_pars").read_text()
    for name in ("VisuCoreDataSlope", "VisuCoreDataOffs"):
        start = text.index(f"##${name}=")
        end = text.index("##$", start + 1)
        text = text[:start] + text[end:]
    path.parent.joinpath("visu_pars").write_text(text)
    write_jcampdx(path.parent / "reco", {"RECO_map_slope": np.full(2, 4.0), "RECO_map_offset": np.zeros(2), **records})

    dataset = Dataset(path)

    assert np.allclose(dataset.slope, 0.25)
    assert np.allclose(dataset.data, 25.0)


def test_a_single_scaling_value_applies_to_every_frame(tmp_path):
    """Spec 7.4 cardinality: one value broadcasts, it is not an IndexError."""
    path = write_2dseq(
        tmp_path / "1" / "pdata" / "1",
        frame_groups=(("FG_SLICE", 3),),
        data=np.full((4, 4, 3), 2, dtype="int16"),
        records={"VisuCoreDataSlope": np.array([1.5]), "VisuCoreDataOffs": np.array([1.0])},
    )

    dataset = Dataset(path)

    assert np.allclose(dataset.data, 4.0)


def test_a_legacy_d3proc_reconstruction_uses_the_in_plane_matrix_only(tmp_path):
    """Spec 8: IM_SIZ is the number of frames, which shape_frames already counts;
    folding it into the block shape asks for IM_SIX*IM_SIY*IM_SIZ pixels per frame."""
    proc = tmp_path / "1" / "pdata" / "1"
    write_jcampdx(
        proc / "visu_pars",
        {
            "VisuCreatorVersion": ["<6.0.1>"],
            "VisuCoreFrameCount": 3,
            "VisuCoreDim": 2,
            "VisuCoreDimDesc": Verbatim("( 2 )\nspatial spatial"),
            "VisuCoreExtent": np.array([20.0, 15.0]),
            "VisuCoreFrameThickness": np.array([1.0]),
            "VisuCoreOrientation": np.tile(np.eye(3).reshape(-1), (3, 1)),
            "VisuCorePosition": np.zeros((3, 3)),
            "VisuCoreDataSlope": np.ones(3),
            "VisuCoreDataOffs": np.zeros(3),
            "VisuFGOrderDescDim": 1,
            "VisuFGOrderDesc": Verbatim("( 1 )\n(3, <FG_SLICE>, <>, 0, 0)"),
            "VisuInstanceType": "MINIMAL_INSTANCE",
        },
    )
    write_jcampdx(proc / "d3proc", {"IM_SIX": 4, "IM_SIY": 3, "IM_SIZ": 3, "DATTYPE": "ip_short", "PR_STA": np.zeros(16)})
    write_binary(proc / "2dseq", np.arange(36, dtype="<i2"), np.dtype("<i2"))

    dataset = Dataset(proc / "2dseq")

    assert dataset.shape_block == (4, 3)
    assert dataset.numpy_dtype == np.dtype("int16")
    assert dataset.data.shape == (4, 3, 3)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("ip_short", "int16"), ("ip_u_short", "uint16"), ("ip_int", "int32"), ("ip_u_int", "uint32"), ("2", "uint8"), ("1", "int8")],
)
def test_the_d3proc_word_type_covers_every_member_and_both_forms(symbol, expected):
    from brukerapi.utils import DATTYPE_WORD_TYPES

    # spec 8: 0=ip_bit ... 6=ip_u_int, written as the symbol or the ordinal
    assert DATTYPE_WORD_TYPES[symbol] == expected


def test_reco_size_is_read_in_the_non_transposed_order():
    """Spec 6.9: RECO_transposition does not change how RECO_size is written,
    and the values are 1-based directions -- for two dimensions both 1 and 2
    mean the same exchange."""
    assert transposed_size((96, 128), 0) == (96, 128)
    assert transposed_size((96, 128), 1) == (128, 96)
    assert transposed_size((96, 128), 2) == (128, 96)
    assert transposed_size((96, 128, 32), 2) == (96, 32, 128)
    assert transposed_size((96, 128, 32), 3) == (32, 128, 96)


# spec 13.1: (storeDataMode, storageDataType, displayMode, logTimeStamp, accumMode, ...)
SETTINGS_64BIT = (
    "STORE_processed, STORE_64bit_float, DISPLAY_each_accumulation, LOG_none, ACCUM_average, "
    "0, 0, 0, 0, 4, 4, NORMALIZE_none, PIPELINE_processed, 0, STREAMING_none, DISPLAY_CoilsSideBySide, 1"
)


def rawdata_experiment(tmp_path, job, settings=None, *, words=None, dtype="<i4"):
    experiment = tmp_path / "26"
    write_jcampdx(
        experiment / "acqp",
        {
            "ACQ_word_size": "_32_BIT",
            "ACQ_sw_version": "<PV-360.3.7>",
            "BYTORDA": "little",
            "ACQ_jobs": Verbatim(f"( 1 )\n({job})"),
            **({"ACQ_ScanPipeJobSettings": Verbatim(f"( 1 )\n({settings})")} if settings else {}),
        },
    )
    write_jcampdx(experiment / "method", {"Method": "<Bruker:FLASH>", "PVM_EncNReceivers": 1})
    write_binary(experiment / "rawdata.job0", np.arange(words, dtype=dtype), np.dtype(dtype))
    return experiment / "rawdata.job0"


def test_an_eight_field_job_is_sized_from_the_scans_that_were_written(tmp_path):
    """Spec 3.3: `[3]` is nTotalScans -- what the experiment acquires, NAE times
    what it writes. nStoredScans is the last element of the 8-field form."""
    path = rawdata_experiment(tmp_path, "8, 20, 5, 12, 101, 178571.4, 1, 4", words=8 * 4)

    dataset = Dataset(path, load=LOAD_STAGES["properties"])

    assert dataset.shape_storage == (8, 1, 4)
    assert Dataset(path).data.shape == (4, 1, 4)


def test_a_64_bit_float_job_is_read_as_float64(tmp_path):
    """Spec 13.1: ACQ_ScanPipeJobSettings[j].storageDataType is the job word type."""
    path = rawdata_experiment(
        tmp_path,
        "8, 20, 5, 4, 101, 178571.4, 4, 1, <job0>",
        SETTINGS_64BIT,
        words=8 * 4,
        dtype="<f8",
    )

    dataset = Dataset(path, load=LOAD_STAGES["properties"])

    assert dataset.numpy_dtype == np.dtype("<f8")


def test_a_dollar_comment_inside_a_string_is_data(tmp_path):
    """Spec 2.2: the text inside `<...>` is free-form, `$$` included."""
    path = tmp_path / "method"
    path.write_text(
        "##TITLE=Parameter List\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$PVM_Comment=( 64 )\n"
        "<a\n"
        "$$b>\n"
        "##END=\n"
    )

    assert JCAMPDX(path)["PVM_Comment"].value == "<a$$b>"


def test_a_scalar_struct_is_not_eaten_as_a_size_bracket(tmp_path):
    """`##$VisuCoreSlicePacksDef=(0, 1)` is a value, not a size -- including when
    the line has a trailing blank."""
    path = tmp_path / "visu_pars"
    path.write_text(
        "##TITLE=Parameter List\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$VisuCoreSlicePacksDef=(0, 1) \n"
        "##END=\n"
    )

    assert JCAMPDX(path)["VisuCoreSlicePacksDef"].value == [0, 1]
