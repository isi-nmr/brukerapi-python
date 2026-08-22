"""Builders for small, self-contained ParaVision datasets.

The parameter shapes here are copied from real ParaVision files (PV5.1, PV6.0.1,
PV7.0.0 and PV360 3.x scans), reduced to the minimum a reader needs.  Tests can
therefore exercise the file format -- geometry, frame groups, scaling, slice
packages -- without any vendor data being present.

Values are written the way ParaVision writes them:

* a scalar goes on the assignment line (``##$VisuCoreDim=2``),
* an array declares its size and puts the values on the following lines
  (``##$VisuCoreSize=( 2 )`` / ``256 256``),
* long value blocks are hard-wrapped near column 80 **at a space**,
* ``$$`` comment lines may appear anywhere between records.

Pass :class:`Verbatim` when a test needs a record written exactly as given --
for example to place a wrap inside a ``<...>`` string.
"""

import numpy as np

MAX_LINE_LEN = 78


class Verbatim:
    """A record body written exactly as given, size bracket included."""

    def __init__(self, text):
        self.text = text


def _wrap(text):
    """Hard-wrap a value block at a space, the way ParaVision does."""
    lines = []
    for physical_line in text.split("\n"):
        rest = physical_line
        while len(rest) > MAX_LINE_LEN:
            cut = rest.rfind(" ", 0, MAX_LINE_LEN + 1)
            if cut <= 0:
                break
            lines.append(rest[: cut + 1])
            rest = rest[cut + 1 :]
        lines.append(rest)
    return "\n".join(lines)


def _format_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def format_record(key, value):
    """Serialize one ``##$key=value`` record."""
    if isinstance(value, Verbatim):
        return f"##${key}={value.text}"

    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind in "US" or (array.dtype == object):
            size = f"( {array.size}, 65 )" if array.ndim == 1 else f"( {', '.join(str(n) for n in array.shape)}, 65 )"
            body = " ".join(_format_scalar(item) for item in array.reshape(-1))
        else:
            size = f"( {array.shape[0]} )" if array.ndim == 1 else f"( {', '.join(str(n) for n in array.shape)} )"
            body = " ".join(_format_scalar(item) for item in array.reshape(-1))
        return f"##${key}={size}\n{_wrap(body)}"

    return f"##${key}={_format_scalar(value)}"


def write_jcampdx(path, records, *, version="4.24", title="Parameter List, synthetic", owner="brukerapi", comments=()):
    """Write `records` (a mapping) as a JCAMP-DX parameter file."""
    lines = [
        f"##TITLE={title}",
        f"##JCAMPDX={version}",
        "##DATATYPE=Parameter Values",
        "##ORIGIN=Bruker BioSpin MRI GmbH",
        f"##OWNER={owner}",
        *(f"$$ {comment}" for comment in comments),
    ]
    lines.extend(format_record(key, value) for key, value in records.items())
    lines.append("##END=")
    path.parent.mkdir(parents=True, exist_ok=True)
    # spec 2.2: ParaVision 360 writes parameter files as UTF-8
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_binary(path, array, dtype):
    """Write `array` as a Bruker binary file (Fortran order, like the vendor)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.asarray(array, dtype=dtype).tobytes(order="F"))
    return path


def axial_orientation(count):
    """`count` copies of the identity orientation matrix, as VisuCoreOrientation."""
    return np.tile(np.eye(3).reshape(-1), (count, 1))


def stacked_positions(first, step, count):
    """Slice-centre positions advancing by `step` from `first`."""
    return np.asarray(first, dtype=float) + np.outer(np.arange(count), np.asarray(step, dtype=float))


def visu_pars_records(
    *,
    size=(4, 4),
    dim=2,
    dim_desc=("spatial", "spatial"),
    extent=(40.0, 40.0),
    frame_thickness=1.5,
    positions=None,
    orientations=None,
    frame_groups=(("FG_SLICE", 3),),
    creator_version="6.0.1",
    subject_position="Head_Supine",
    word_type="_16BIT_SGN_INT",
    slice_packs=None,
    slice_pack_distance=None,
    slope=1.0,
    offset=0.0,
    extra=None,
):
    """Records of a `visu_pars` describing one reconstructed image series."""
    frame_count = int(np.prod([group[1] for group in frame_groups])) if frame_groups else 1
    positions = np.atleast_2d(np.asarray(positions if positions is not None else stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, frame_thickness), frame_count), dtype=float))
    orientations = np.atleast_2d(np.asarray(orientations if orientations is not None else axial_orientation(positions.shape[0]), dtype=float))

    records = {
        "VisuVersion": 3,
        "VisuCreator": ["<ParaVision>"],
        "VisuCreatorVersion": [f"<{creator_version}>"],
        "VisuCoreFrameCount": frame_count,
        "VisuCoreDim": dim,
        "VisuCoreSize": np.asarray(size, dtype=int),
        "VisuCoreDimDesc": Verbatim(f"( {len(dim_desc)} )\n{' '.join(dim_desc)}"),
        "VisuCoreExtent": np.asarray(extent, dtype=float),
        "VisuCoreFrameThickness": np.atleast_1d(np.asarray(frame_thickness, dtype=float)),
        "VisuCoreUnits": ["<mm>"] * len(dim_desc),
        "VisuCoreOrientation": orientations,
        "VisuCorePosition": positions,
        "VisuCoreDataMin": np.zeros(frame_count),
        "VisuCoreDataMax": np.full(frame_count, 1000.0),
        "VisuCoreDataOffs": np.full(frame_count, float(offset)),
        "VisuCoreDataSlope": np.full(frame_count, float(slope)),
        "VisuCoreFrameType": Verbatim("( 1 )\nMAGNITUDE_IMAGE"),
        "VisuCoreWordType": word_type,
        "VisuCoreByteOrder": "littleEndian",
        "VisuSubjectPosition": subject_position,
        "VisuSubjectName": ["<synthetic>"],
        "VisuSubjectId": ["<phantom>"],
        "VisuStudyNumber": 1,
    }

    if slice_packs is not None:
        records["VisuCoreSlicePacksDef"] = Verbatim(f"({slice_packs[0]}, {len(slice_packs[1])})")
        records["VisuCoreSlicePacksSlices"] = Verbatim(f"( {len(slice_packs[1])} )\n" + " ".join(f"({first}, {count})" for first, count in slice_packs[1]))
    if slice_pack_distance is not None:
        distances = np.atleast_1d(np.asarray(slice_pack_distance, dtype=float))
        records["VisuCoreSlicePacksSliceDist"] = distances

    if frame_groups:
        # A descriptor is (len, groupId, groupComment, valsStart, valsCnt); the last two
        # index the VisuGroupDepVals window owned by that group (spec 7.4).
        records["VisuFGOrderDescDim"] = len(frame_groups)
        descriptors = [f"({group[1]}, <{group[0]}>, <>, {group[2] if len(group) > 2 else 0}, {group[3] if len(group) > 3 else 0})" for group in frame_groups]
        records["VisuFGOrderDesc"] = Verbatim(f"( {len(descriptors)} )\n" + " ".join(descriptors))

    if extra:
        records.update(extra)
    return records


FID_DTYPES = {
    "GO_32BIT_SGN_INT": np.dtype("int32"),
    "GO_16BIT_SGN_INT": np.dtype("int16"),
    "GO_32BIT_FLOAT": np.dtype("float32"),
}


def write_fid(directory, acqp, method, *, blocks=1, data=None):
    """Write an experiment folder (`acqp`, `method`, `fid`) and return the fid path.

    Without `data` the binary is sized from the records themselves --
    ``ACQ_size[0]`` words per block for ``GO_block_size = continuous`` -- and
    filled with distinct values, so a test can tell which samples were used.
    """
    directory.mkdir(parents=True, exist_ok=True)
    write_jcampdx(directory / "acqp", acqp)
    write_jcampdx(directory / "method", method)

    dtype = FID_DTYPES[str(acqp["GO_raw_data_format"])].newbyteorder("<" if str(acqp["BYTORDA"]) == "little" else ">")
    if data is None:
        block_size = int(np.atleast_1d(acqp["ACQ_size"])[0]) * int(method["PVM_EncNReceivers"])
        data = np.arange(1, block_size * blocks + 1, dtype=dtype)
    write_binary(directory / "fid", data, dtype)
    return directory / "fid"


WORD_TYPES = {
    "_8BIT_UNSGN_INT": np.dtype("uint8"),
    "_16BIT_SGN_INT": np.dtype("int16"),
    "_32BIT_SGN_INT": np.dtype("int32"),
    "_32BIT_FLOAT": np.dtype("float32"),
}


def write_2dseq(directory, records=None, data=None, **kwargs):
    """Write a complete ``pdata`` reconstruction and return the 2dseq path.

    `records` overrides the generated `visu_pars`; any other keyword argument is
    forwarded to :func:`visu_pars_records`.
    """
    records = {**visu_pars_records(**kwargs), **(records or {})}
    directory.mkdir(parents=True, exist_ok=True)
    write_jcampdx(directory / "visu_pars", records)

    size = tuple(int(length) for length in np.atleast_1d(records["VisuCoreSize"]))
    frames = int(records["VisuCoreFrameCount"])
    dtype = WORD_TYPES[records["VisuCoreWordType"]]
    if data is None:
        data = np.arange(int(np.prod(size)) * frames, dtype=dtype).reshape(size + (frames,), order="F")
    write_binary(directory / "2dseq", data, dtype)
    return directory / "2dseq"
