"""Datasets read through the pathlib read protocol, not the filesystem.

A ``zipfile.Path`` stands in for a real path, so a dataset inside a
``.zip``/``.PvDatasets`` archive is readable without extracting it. The dataset
below is synthesised, so these tests need no test-data corpus.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from brukerapi.dataset import Dataset
from brukerapi.folders import Experiment, Folder
from brukerapi.jcampdx import JCAMPDX
from brukerapi.paths import as_path, file_size, listdir, traverse, with_suffix
from test.synthetic import write_binary, write_jcampdx

VISU_PARS = """##TITLE=Parameter List, ParaVision 6.0.1
##JCAMPDX=4.24
##$VisuCreatorVersion=<6.0.1>
##$VisuCoreDim=2
##$VisuCoreSize=( 2 )
4 3
##$VisuCoreDimDesc=( 2 )
spatial spatial
##$VisuCoreExtent=( 2 )
20 15
##$VisuCoreFrameCount=2
##$VisuCoreWordType=_16BIT_SGN_INT
##$VisuCoreByteOrder=littleEndian
##$VisuCoreDataSlope=( 2 )
1 1
##$VisuCoreDataOffs=( 2 )
0 0
##$VisuCoreOrientation=( 2, 9 )
1 0 0 0 1 0 0 0 1 1 0 0 0 1 0 0 0 1
##$VisuCorePosition=( 2, 3 )
0 0 0 0 0 1
##$VisuFGOrderDescDim=1
##$VisuFGOrderDesc=( 1 )
(2, <FG_SLICE>, <>, 0, 0)
##END=
"""

ACQP = "##TITLE=Parameter List\n##JCAMPDX=4.24\n##$ACQ_scan_name=( 64 )\n<demo>\n##END=\n"

DATA = np.arange(24, dtype="<i2").reshape((4, 3, 2), order="F")


@pytest.fixture
def study_dir(tmp_path):
    """A one-experiment, one-processing study on the filesystem."""
    proc = tmp_path / "study" / "1" / "pdata" / "1"
    proc.mkdir(parents=True)
    (tmp_path / "study" / "1" / "acqp").write_text(ACQP)
    (proc / "visu_pars").write_text(VISU_PARS)
    (proc / "2dseq").write_bytes(DATA.tobytes(order="F"))
    return tmp_path / "study"


@pytest.fixture
def study_zip(study_dir, tmp_path):
    """The same study inside an archive, as a :class:`zipfile.Path` root."""
    archive = tmp_path / "study.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(study_dir.rglob("*")):
            if path.is_file():
                zf.write(path, str(path.relative_to(study_dir.parent)))
    return next(child for child in zipfile.Path(zipfile.ZipFile(archive)).iterdir() if child.is_dir())


def test_dataset_reads_from_archive_identically(study_dir, study_zip):
    """The array and the axis labels do not depend on where the data was read from."""
    from_dir = Dataset(study_dir / "1" / "pdata" / "1" / "2dseq", scale=False)
    from_zip = Dataset(study_zip / "1" / "pdata" / "1" / "2dseq", scale=False)

    assert np.array_equal(from_zip.data, DATA)
    assert np.array_equal(from_zip.data, from_dir.data)
    assert list(from_zip.dim_type) == list(from_dir.dim_type)
    assert from_zip.shape_final == from_dir.shape_final


def test_dataset_reads_from_a_read_only_source(study_dir):
    """A dataset whose files are not writable must still load.

    Scanner archives are normally exposed read-only, and nothing here writes: the
    array is copied out of the map immediately. numpy's default memmap mode is
    "r+", which asks the operating system for write access and fails with EACCES
    on such a source, so the mode has to be given explicitly.
    """
    for path in sorted(study_dir.rglob("*")):
        if path.is_file():
            path.chmod(0o444)

    try:
        dataset = Dataset(study_dir / "1" / "pdata" / "1" / "2dseq", scale=False)
        assert np.array_equal(dataset.data, DATA)
    finally:
        # restore, so pytest can clean the temporary directory up
        for path in sorted(study_dir.rglob("*")):
            if path.is_file():
                path.chmod(0o644)


def test_parameters_resolve_through_relative_paths(study_dir, study_zip):
    """``../../acqp`` resolves inside an archive, where ``..`` is not collapsed."""
    dataset = Dataset(study_zip / "1" / "pdata" / "1" / "2dseq", scale=False, parameter_files=["acqp"])
    assert dataset["ACQ_scan_name"].value == "demo"


def test_folder_traverses_an_archive(study_zip):
    folder = Folder(study_zip)
    assert [child.path.name for child in folder.children if isinstance(child, Experiment)] == ["1"]


def test_jcampdx_reads_from_archive(study_zip):
    visu_pars = JCAMPDX(study_zip / "1" / "pdata" / "1" / "visu_pars")
    assert visu_pars.get_value("VisuCoreDim") == 2


def test_get_value_default_for_absent_key(study_dir):
    """Which parameters exist is ParaVision-version dependent, so absence needs a default."""
    visu_pars = JCAMPDX(study_dir / "1" / "pdata" / "1" / "visu_pars")
    assert visu_pars.get_value("VisuCoreDim", 99) == 2
    assert visu_pars.get_value("NoSuchParameter") is None
    assert visu_pars.get_value("NoSuchParameter", "fallback") == "fallback"
    with pytest.raises(KeyError):
        visu_pars["NoSuchParameter"]


def test_path_helpers_work_for_both_kinds(study_dir, study_zip):
    assert isinstance(as_path(str(study_dir)), Path)
    assert as_path(study_zip) is study_zip  # passed through, not coerced

    for root in (study_dir, study_zip):
        proc = root / "1" / "pdata" / "1"
        assert sorted(listdir(proc)) == ["2dseq", "visu_pars"]
        assert file_size(proc / "2dseq") == DATA.nbytes
        assert traverse(proc, "../../acqp").name == "acqp"
        assert traverse(proc, "../../acqp").exists()
        assert with_suffix(proc / "2dseq", ".job0").name == "2dseq.job0"


TRAJ_PROJECTIONS = 4
TRAJ_SAMPLES = 5
TRAJ = np.arange(2 * TRAJ_SAMPLES * TRAJ_PROJECTIONS, dtype="f8").reshape((2, TRAJ_SAMPLES, TRAJ_PROJECTIONS), order="F")


@pytest.fixture
def radial_zip(tmp_path):
    """A radial experiment -- acqp, method and traj -- inside an archive."""
    experiment = tmp_path / "radial" / "23"
    write_jcampdx(
        experiment / "acqp",
        {
            "ACQ_dim": 2,
            "PULPROG": "<UTE.ppg>",
            "NPro": TRAJ_PROJECTIONS,
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
        },
    )
    write_jcampdx(experiment / "method", {"Method": "<Bruker:UTE>", "PVM_EncNReceivers": 1})
    write_binary(experiment / "traj", TRAJ, np.dtype("f8"))

    archive = tmp_path / "radial.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted((tmp_path / "radial").rglob("*")):
            if path.is_file():
                zf.write(path, str(path.relative_to(tmp_path / "radial")))
    return zipfile.Path(zipfile.ZipFile(archive))


def test_traj_reads_from_an_archive(radial_zip, tmp_path):
    """A traj sized itself with os.stat, which an archive member cannot answer.

    The failure was a bare TypeError, and a radial fid read from an archive
    silently lost its trajectory: the error was downgraded to a warning and
    dataset.traj then raised TrajNotLoaded as if none existed.
    """
    from_zip = Dataset(radial_zip / "23" / "traj")
    from_dir = Dataset(tmp_path / "radial" / "23" / "traj")

    assert from_zip.data.shape == (2, TRAJ_SAMPLES, TRAJ_PROJECTIONS)
    assert np.array_equal(from_zip.data, from_dir.data)
    assert np.array_equal(from_zip.data, TRAJ)
