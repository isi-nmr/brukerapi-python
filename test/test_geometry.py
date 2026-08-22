"""Image geometry: FILE_FORMAT.md 7.2 (VisuCore), 7.10 (slice packages), 12 (frames).

Every dataset here is synthetic, built by test/synthetic.py from the parameter
shapes of real ParaVision files, so the geometry rules are exercised without any
vendor data.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from brukerapi.dataset import LOAD_STAGES, Dataset
from brukerapi.exceptions import UnsupportedDatasetType
from test.synthetic import Verbatim, axial_orientation, stacked_positions, visu_pars_records, write_2dseq, write_binary, write_fid, write_jcampdx

SAGITTAL_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
PV360_NIFTI_ROOT = Path("test/test_data/PV360_StdData")
# Every reconstruction that ships a NIfTI export, rather than a fixed list, so
# a dataset added to the corpus is checked without editing this file.
PV360_NIFTI_EXPORTS = sorted(directory.parent.relative_to(PV360_NIFTI_ROOT).as_posix() for directory in PV360_NIFTI_ROOT.glob("*/pdata/*/nifti") if any(directory.glob("*.nii")))
CORPUS_ROOT = Path("test/test_data")
# Every raw acquisition of the test corpus that has a first reconstruction to
# compare against: the study/experiment layout, and the PV360 standard data's
# flat one (rawdata.job0 per scan).
ACQUISITIONS_WITH_RECONSTRUCTION = sorted(
    path.relative_to(CORPUS_ROOT).as_posix()
    for pattern in ("*/*/*/fid", "*/*/rawdata.job0")
    for path in CORPUS_ROOT.glob(pattern)
    if (path.parent / "pdata" / "1" / "2dseq").exists()
)

# Spec 5.4 geometry of real scans, used as fixtures: Zenodo 4048286 (PV5.1 0.2H2/13,
# 0.2H2/10), Zenodo 4522220 (PV6.0.1 lego phantom/3) and PV360_StdData T2_TurboRARE.
AXIAL = np.eye(3)
SAGITTAL = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
CORONAL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def nifti_affine(path):
    """Read the NIfTI-1 image affine without making nibabel a test dependency.

    The standard stores it two independent ways, and a file need only carry one.
    ParaVision writes the quaternion (`qform_code > 0`) and leaves the sform
    zeroed, so a reader that consults the sform alone finds nothing at all.
    """
    header = path.read_bytes()[:348]
    if header.startswith(b"version https://git-lfs.github.com/spec/"):
        pytest.skip("ParaVision NIfTI exports are Git LFS pointers; fetch the LFS assets to validate their affines")

    if len(header) != 348:
        pytest.fail(f"{path} does not contain a complete NIfTI-1 header")
    if struct.unpack_from("<I", header)[0] == 348:
        byte_order = "<"
    elif struct.unpack_from(">I", header)[0] == 348:
        byte_order = ">"
    else:
        pytest.fail(f"{path} is not a NIfTI-1 image")

    qform_code, sform_code = struct.unpack_from(f"{byte_order}2h", header, 252)
    if sform_code > 0:
        sform = np.asarray(struct.unpack_from(f"{byte_order}12f", header, 280)).reshape(3, 4)
        return np.vstack((sform, (0.0, 0.0, 0.0, 1.0)))
    if qform_code > 0:
        return nifti_qform_affine(header, byte_order)
    pytest.fail(f"{path} declares neither an sform nor a qform")


def nifti_qform_affine(header, byte_order):
    """NIfTI-1 "method 2": the affine as a rotation quaternion and a scaling.

    Only the vector part of the unit quaternion is stored; `pixdim[0]` is the
    handedness factor applied to the third column.
    """
    pixdim = struct.unpack_from(f"{byte_order}8f", header, 76)
    handedness = pixdim[0] if pixdim[0] else 1.0
    b, c, d, *origin = struct.unpack_from(f"{byte_order}6f", header, 256)
    a = np.sqrt(max(0.0, 1.0 - (b * b + c * c + d * d)))
    rotation = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )

    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag((pixdim[1], pixdim[2], handedness * pixdim[3]))
    affine[:3, 3] = origin
    return affine


def load(tmp_path, **kwargs):
    path = write_2dseq(tmp_path / "9" / "pdata" / "1", **kwargs)
    return Dataset(path, load=LOAD_STAGES["properties"])


def test_affine_maps_the_first_voxel_onto_visucoreposition(tmp_path):
    positions = stacked_positions((-20.0, -17.0, -3.0), (0.0, 0.0, 1.5), 5)
    dataset = load(tmp_path, frame_groups=(("FG_SLICE", 5),), positions=positions)

    origin = dataset.affine @ np.array([0.0, 0.0, 0.0, 1.0])

    assert np.allclose(origin[:3], positions[0])


def test_affine_maps_every_slice_index_onto_its_own_position(tmp_path):
    positions = stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), 5)
    dataset = load(tmp_path, frame_groups=(("FG_SLICE", 5),), positions=positions)

    for index, position in enumerate(positions):
        assert np.allclose((dataset.affine @ np.array([0.0, 0.0, float(index), 1.0]))[:3], position)


def test_affine_slice_column_keeps_the_direction_of_a_descending_stack(tmp_path):
    positions = stacked_positions((-20.0, -20.0, 5.0), (0.0, 0.0, -2.5), 4)
    dataset = load(tmp_path, frame_groups=(("FG_SLICE", 4),), positions=positions)

    # The stack advances against the third row of the orientation matrix, so a
    # magnitude-only spacing would silently reverse it.
    assert np.allclose(dataset.affine[:3, 2], [0.0, 0.0, -2.5])
    assert np.linalg.det(dataset.affine) != 0.0


def test_affine_is_the_same_whichever_way_the_subject_lies(tmp_path):
    positions = stacked_positions((-15.0, -17.0, -3.0), (0.0, 0.0, 1.0), 3)
    supine = load(tmp_path / "supine", frame_groups=(("FG_SLICE", 3),), positions=positions, subject_position="Head_Supine")
    prone = load(tmp_path / "prone", frame_groups=(("FG_SLICE", 3),), positions=positions, subject_position="Head_Prone")

    # VisuCoreOrientation/VisuCorePosition are already in the DICOM patient frame
    # (spec 7.2/12); re-applying VisuSubjectPosition mirrors x and y.
    assert np.allclose(supine.affine, prone.affine)
    assert np.allclose(np.diag(supine.affine)[:2], [10.0, 10.0])
    assert np.allclose(supine.affine[:3, 3], positions[0])


def test_affine_of_a_3d_volume_uses_the_third_extent(tmp_path):
    dataset = load(
        tmp_path,
        dim=3,
        dim_desc=("spatial", "spatial", "spatial"),
        size=(4, 4, 8),
        extent=(40.0, 40.0, 16.0),
        frame_groups=(),
        positions=np.array([[-20.0, -20.0, -8.0]]),
        orientations=axial_orientation(1),
    )

    assert np.allclose(np.diag(dataset.affine)[:3], [10.0, 10.0, 2.0])
    assert np.allclose(dataset.affine[:3, 3], [-20.0, -20.0, -8.0])


def test_affine_refuses_frames_that_are_not_purely_spatial(tmp_path):
    dataset = load(
        tmp_path,
        dim=3,
        dim_desc=("spectroscopic", "spatial", "spatial"),
        size=(64, 4, 4),
        extent=(1.0, 40.0, 40.0),
        frame_groups=(),
        positions=np.array([[-20.0, -20.0, 0.0]]),
        orientations=axial_orientation(1),
    )

    # Spec 7.2: such scans must be detected and skipped, not handed a plausible
    # identity affine at the scanner origin.
    with pytest.raises(UnsupportedDatasetType, match="rather than purely spatial"):
        _ = dataset.affine


def test_report_carries_computed_geometry_available_to_each_dataset(tmp_path):
    image = load(tmp_path / "image", frame_groups=(("FG_SLICE", 3),))
    spectroscopy = load(
        tmp_path / "spectroscopy",
        dim=1,
        dim_desc=("spectroscopic",),
        size=(64,),
        extent=(1.0,),
        frame_groups=(),
        positions=np.array([[0.0, 0.0, 0.0]]),
        orientations=axial_orientation(1),
    )

    image_report = image.to_dict()
    spectroscopy_report = spectroscopy.to_dict()

    assert np.allclose(image_report["affine"], image.affine)
    assert image_report["slice_distance"] == image.slice_distance
    assert "affine" not in spectroscopy_report
    assert spectroscopy_report["slice_distance"] == spectroscopy.slice_distance


@pytest.mark.parametrize("relative_path", PV360_NIFTI_EXPORTS)
def test_affine_agrees_with_the_paravision_nifti_export(relative_path):
    """PV's NIfTI export is an independent geometry oracle.

    VisuCore geometry is in the DICOM patient frame, whereas a NIfTI affine is
    in RAS.  Therefore the equivalent NIfTI affine negates the patient x/y
    axes (spec 12).

    Every volume of an export shares one geometry -- a diffusion series writes
    one file per direction -- so all of them are checked, which also pins the
    export as internally consistent.
    """
    reconstruction = PV360_NIFTI_ROOT / relative_path
    dataset = Dataset(reconstruction / "2dseq", load=LOAD_STAGES["properties"])

    expected = np.diag((-1.0, -1.0, 1.0, 1.0)) @ dataset.affine
    for nifti_path in sorted((reconstruction / "nifti").glob("*.nii")):
        assert np.allclose(nifti_affine(nifti_path), expected, atol=1e-5), nifti_path.name


def test_slice_spacing_is_centre_to_centre_not_thickness_plus_distance(tmp_path):
    dataset = load(
        tmp_path,
        frame_groups=(("FG_SLICE", 5),),
        positions=stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), 5),
        frame_thickness=1.5,
        slice_packs=(0, [(0, 5)]),
        slice_pack_distance=1.5,
    )

    # Spec 7.10: VisuCoreSlicePacksSliceDist is the inter-slice distance, so it is
    # not additive with VisuCoreFrameThickness.
    assert np.allclose(dataset.resolution, [10.0, 10.0, 1.5])
    assert np.isclose(np.linalg.norm(dataset.affine[:3, 2]), 1.5)


def test_slice_spacing_of_a_non_axial_stack_is_not_measured_along_z(tmp_path):
    positions = stacked_positions((0.0, -20.0, -20.0), (-1.0, 0.0, 0.0), 4)
    dataset = load(
        tmp_path,
        creator_version="5.1",
        frame_groups=(("FG_SLICE", 4),),
        positions=positions,
        orientations=np.tile(SAGITTAL_ORIENTATION, (4, 1)),
    )

    # A sagittal stack advances in x; taking the z component alone collapses the
    # volume and makes the affine singular.
    assert np.isclose(dataset.resolution[2], 1.0)
    assert np.linalg.det(dataset.affine) != 0.0
    assert np.allclose(dataset.affine[:3, 2], [-1.0, 0.0, 0.0])


def test_single_slice_spacing_falls_back_to_the_slice_distance(tmp_path):
    dataset = load(
        tmp_path,
        frame_groups=(("FG_SLICE", 1),),
        positions=np.array([[-20.0, -20.0, 0.0]]),
        orientations=axial_orientation(1),
        frame_thickness=0.8,
        slice_packs=(0, [(0, 1)]),
        slice_pack_distance=2.0,
    )

    assert np.isclose(np.linalg.norm(dataset.affine[:3, 2]), 2.0)
    assert np.isclose(dataset.resolution[2], 2.0)


def test_single_slice_spacing_falls_back_to_the_frame_thickness(tmp_path):
    dataset = load(
        tmp_path,
        frame_groups=(("FG_SLICE", 1),),
        positions=np.array([[-20.0, -20.0, 0.0]]),
        orientations=axial_orientation(1),
        frame_thickness=0.8,
    )

    assert np.isclose(np.linalg.norm(dataset.affine[:3, 2]), 0.8)


def test_each_slice_package_gets_its_own_affine(tmp_path):
    positions = np.vstack(
        [
            stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), 5),
            stacked_positions((0.0, -20.0, -20.0), (-1.0, 0.0, 0.0), 3),
        ]
    )
    orientations = np.vstack([axial_orientation(5), np.tile(SAGITTAL_ORIENTATION, (3, 1))])
    dataset = load(
        tmp_path,
        frame_groups=(("FG_SLICE", 8),),
        positions=positions,
        orientations=orientations,
        slice_packs=(0, [(0, 5), (5, 3)]),
        slice_pack_distance=[1.5, 1.0],
    )

    assert dataset.slice_packages_index() == [(0, 5), (5, 3)]
    assert np.allclose(dataset.affine_of_package(0)[:3, 3], positions[0])
    assert np.allclose(dataset.affine_of_package(0)[:3, 2], [0.0, 0.0, 1.5])
    assert np.allclose(dataset.affine_of_package(1)[:3, 3], positions[5])
    assert np.allclose(dataset.affine_of_package(1)[:3, 2], [-1.0, 0.0, 0.0])

    with pytest.warns(RuntimeWarning, match="multiple slice packages"):
        assert np.allclose(dataset.affine, dataset.affine_of_package(0))


def test_slice_packages_are_inferred_from_orientation_when_pv51_omits_them(tmp_path):
    positions = np.vstack(
        [
            stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), 4),
            stacked_positions((0.0, -20.0, -20.0), (-1.0, 0.0, 0.0), 2),
        ]
    )
    orientations = np.vstack([axial_orientation(4), np.tile(SAGITTAL_ORIENTATION, (2, 1))])
    dataset = load(
        tmp_path,
        creator_version="5.1",
        frame_groups=(("FG_SLICE", 6),),
        positions=positions,
        orientations=orientations,
    )

    # PV5.1 defines none of the slice-package parameters (spec 7.10).
    assert "VisuCoreSlicePacksSlices" not in dataset
    assert dataset.slice_packages_index() == [(0, 4), (4, 2)]


def test_geometry_follows_the_data_when_frames_are_stored_in_reverse(tmp_path):
    positions = stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), 4)
    dataset = load(
        tmp_path,
        frame_groups=(("FG_SLICE", 4),),
        positions=positions,
        extra={"VisuCoreDiskSliceOrder": "disk_reverse_slice_order"},
    )

    # Schema2dseq flips the slice axis for disk_reverse_slice_order, so index 0 of
    # the data array is the last frame on disk (spec 7.2/7.3).
    assert np.allclose(dataset.affine[:3, 3], positions[-1])
    assert np.allclose(dataset.affine[:3, 2], [0.0, 0.0, -1.5])


def test_slice_distance_is_the_step_the_affine_is_built_from(tmp_path):
    """A gapped stack: the spacing is not the slice thickness (spec 7.10).

    `VisuCoreFrameThickness` stays the source for the thickness; nothing
    reported the spacing, so a consumer had to re-derive it.
    """
    dataset = load(
        tmp_path,
        frame_groups=(("FG_SLICE", 4),),
        positions=stacked_positions((-20.0, -20.0, -7.5), (0.0, 0.0, 5.0), 4),
        frame_thickness=2.0,
    )

    assert dataset.slice_distance == [5.0]
    assert np.isclose(dataset["VisuCoreFrameThickness"].array[0], 2.0)


def test_slice_distance_of_a_3d_volume_is_the_plane_step_not_the_slab(tmp_path):
    """`VisuCoreFrameThickness` of a 3-D acquisition is the whole slab.

    Reporting it as a slice step overstates the spacing by the partition count
    -- 16 mm instead of 2 mm here.
    """
    dataset = load(
        tmp_path,
        dim=3,
        dim_desc=("spatial", "spatial", "spatial"),
        size=(4, 4, 8),
        extent=(40.0, 40.0, 16.0),
        frame_groups=(),
        positions=np.array([[-20.0, -20.0, -8.0]]),
        orientations=axial_orientation(1),
        frame_thickness=16.0,
    )

    assert dataset.slice_distance == [2.0]


def test_slice_distance_is_reported_per_package(tmp_path):
    """Packages can be spaced differently, and a single number cannot say so.

    `resolution[2]` measures across the whole stack, so for two packages that
    do not share an orientation it returns the diagonal between them rather
    than either spacing.
    """
    positions = np.vstack(
        [
            stacked_positions((-20.0, -20.0, -3.0), (0.0, 0.0, 1.5), 5),
            stacked_positions((0.0, -20.0, -20.0), (-1.0, 0.0, 0.0), 3),
        ]
    )
    orientations = np.vstack([axial_orientation(5), np.tile(SAGITTAL_ORIENTATION, (3, 1))])
    dataset = load(
        tmp_path,
        frame_groups=(("FG_SLICE", 8),),
        positions=positions,
        orientations=orientations,
        slice_packs=(0, [(0, 5), (5, 3)]),
        slice_pack_distance=[1.5, 1.0],
    )

    assert dataset.slice_distance == [1.5, 1.0]


def test_slice_distance_needs_geometry_and_nothing_else(tmp_path):
    """Availability is that of `affine_of_package`: geometry, not spatial frames.

    A parametric map whose third axis is a frame group still has a real slice
    spacing, and it is reported; a reconstruction with no VisuCorePosition has
    none, and says so.
    """
    parametric = load(
        tmp_path / "isa",
        frame_groups=(("FG_ISA", 2),),
        positions=stacked_positions((-20.0, -20.0, 0.0), (0.0, 0.0, 0.0), 2),
        frame_thickness=0.8,
        slice_packs=(0, [(0, 2)]),
        slice_pack_distance=0.8,
    )
    assert parametric.slice_distance == [0.8]

    flat = tmp_path / "flat" / "9" / "pdata" / "1"
    write_jcampdx(flat / "visu_pars", {key: value for key, value in visu_pars_records().items() if not key.startswith(("VisuCorePosition", "VisuCoreOrientation"))})
    write_binary(flat / "2dseq", np.zeros(4 * 4 * 3), np.dtype("<i2"))
    without_geometry = Dataset(flat / "2dseq", load=LOAD_STAGES["properties"])
    with pytest.raises(UnsupportedDatasetType, match="carries no VisuCorePosition"):
        _ = without_geometry.slice_distance


@pytest.mark.parametrize(
    ("normal", "step"),
    [
        ("axial", (0.0, 0.0, 1.0)),
        ("coronal", (0.0, -1.0, 0.0)),
        ("sagittal", (-1.0, 0.0, 0.0)),
        ("oblique", (-0.12, -0.993, 0.0)),
    ],
)
def test_extent_spans_the_slice_axis_whatever_direction_it_points(tmp_path, normal, step):
    """Spec 7.2/12: `VisuCorePosition` is a 3-vector in patient coordinates and
    the slice normal is the third row of `VisuCoreOrientation`.

    Taking only the patient-z component of the slice step made the slice extent
    exactly zero for every coronal, sagittal and oblique stack, and the step was
    counted once per slice plus one.
    """
    slices = 5
    path = write_2dseq(
        tmp_path / normal / "pdata" / "1",
        frame_groups=(("FG_SLICE", slices),),
        positions=stacked_positions((-20.0, -20.0, -3.0), step, slices),
    )

    dataset = Dataset(path)
    spacing = float(np.linalg.norm(step))

    assert dataset.extent[2] == pytest.approx(slices * spacing)
    assert dataset.resolution[2] == pytest.approx(spacing)
    assert dataset.slice_distance[0] == pytest.approx(spacing)


def test_repeated_positions_are_one_slice_not_a_stack(tmp_path):
    """Spec 7.4 cardinality: a parameter that is not frame-group dependent may
    carry no value, one value, or `VisuCoreFrameCount` values.

    An FG_ISA parameter-map PROCNO writes one position per map, all identical.
    Reading the leading size as a slice count made the step the difference
    between two copies of the same position -- exactly zero -- so the maps came
    out with a degenerate slice resolution.
    """
    maps = 6
    path = write_2dseq(
        tmp_path / "pdata" / "2",
        frame_groups=(("FG_ISA", maps),),
        positions=np.tile([9.8, 11.4, -1.7], (maps, 1)),
        frame_thickness=np.full(maps, 0.8),
        extent=(20.0, 20.0),
        size=(4, 4),
    )

    dataset = Dataset(path)

    assert dataset.is_single_slice is True
    assert dataset.resolution[2] == pytest.approx(0.8)
    assert dataset.extent[2] == pytest.approx(0.8)
    # the maps are frames of one slice, so the slice axis is a singleton
    assert dataset.shape == (4, 4, 1, maps)
    assert [str(label) for label in dataset.dim_type] == ["spatial", "spatial", "frame", "FG_ISA"]


def write_acquisition(
    tmp_path,
    *,
    version="PV 6.0.1",
    patient_pos="Head_Supine",
    grad_matrix,
    obj_order,
    slice_offsets,
    read_offset=0.0,
    phase_offset=0.0,
    fov=(40.0, 40.0),
    matrix=(8, 8),
    slice_thick=1.0,
    slice_sepn=None,
    dim=2,
    drop=(),
    method=None,
):
    """An experiment whose acqp carries the spec 5.4 geometry of `grad_matrix` (acquisition order), and its fid path."""
    slices = len(obj_order)
    acqp = {
        "ACQ_sw_version": [f"<{version}>"],
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "GO_block_size": "continuous",
        "BYTORDA": "little",
        "ACQ_dim": dim,
        "ACQ_dim_desc": Verbatim(f"( {dim} )\n" + " ".join(["Spatial"] * dim)),
        "ACQ_size": np.array([2 * matrix[0], *matrix[1:]]),
        "NI": slices,
        "NR": 1,
        "NSLICES": slices,
        "ACQ_phase_factor": 1,
        "PULPROG": ["<FLASH.ppg>"],
        "ACQ_patient_pos": patient_pos,
        "ACQ_obj_order": np.asarray(obj_order, dtype=int),
        "ACQ_grad_matrix": np.asarray(grad_matrix, dtype=float).reshape(-1, 3, 3),
        "ACQ_slice_offset": np.atleast_1d(np.asarray(slice_offsets, dtype=float)),
        "ACQ_read_offset": np.full(slices, read_offset),
        "ACQ_phase1_offset": np.full(slices, phase_offset),
        "ACQ_fov": np.asarray(fov, dtype=float) / 10.0,  # acqp keeps the field of view in cm (spec 5.1)
        "ACQ_slice_thick": slice_thick,
    }
    if slice_sepn is not None:
        acqp["ACQ_slice_sepn"] = slice_sepn
    for name in drop:
        del acqp[name]
    method = {"PVM_EncNReceivers": 1, "PVM_EncMatrix": np.asarray(matrix), "PVM_DigNp": matrix[0], **(method or {})}
    return write_fid(tmp_path / "1", acqp, method, blocks=slices * int(np.prod(matrix[1:])))


def test_acquisition_affine_places_every_slice_where_the_reconstruction_does(tmp_path):
    """Spec 5.4/12: the first voxel of slice k from acqp must be the 2dseq's VisuCorePosition[k].

    PV5.1 0.2H2/13 -- four interleaved axial slices, 50 mm field of view, offsets
    -7.5 .. 7.5 -- reconstructs to VisuCorePosition (-25, -25, offset) with an
    identity VisuCoreOrientation: index N/2 is the field-of-view centre, and the
    image axes run against the gradient directions.
    """
    fid = write_acquisition(
        tmp_path,
        version="PV 5.1",
        grad_matrix=[AXIAL] * 4,
        obj_order=[0, 2, 1, 3],
        slice_offsets=[-7.5, -2.5, 2.5, 7.5],
        fov=(50.0, 50.0),
        matrix=(64, 64),
        slice_thick=2.0,
        slice_sepn=5.0,
    )
    acquisition = Dataset(fid, load=LOAD_STAGES["properties"])

    affines = acquisition.acquisition_affines()

    assert len(affines) == 4
    for affine, offset in zip(affines, (-7.5, -2.5, 2.5, 7.5)):
        assert np.allclose(affine[:3, 3], (-25.0, -25.0, offset))
        assert np.allclose(affine[:3, :3], np.diag((50 / 64, 50 / 64, 5.0)))
    assert np.allclose(acquisition.acquisition_affine(3), affines[3])


def test_acquisition_affine_reads_the_gradient_matrix_in_acquisition_order(tmp_path):
    """ACQ_grad_matrix is one matrix per *acquisition position*, the offsets are per slice.

    Spec 5.4 says the matrix is built from the slice-pack orientations and the
    slice order, and the three-package scouts of the corpus confirm it: reading
    it per slice id puts the sagittal and coronal slices 5.7 mm off.  PV6.0.1
    lego phantom/3 acquires its axial, sagittal and coronal slices in the order
    0, 2, 1 and reconstructs them at (-20, -20, 0), (0, -20, 20) and
    (-20, 0, 20) with the orientations below.
    """
    fid = write_acquisition(tmp_path, grad_matrix=[AXIAL, CORONAL, SAGITTAL], obj_order=[0, 2, 1], slice_offsets=[0.0, 0.0, 0.0], matrix=(256, 256))

    axial, sagittal, coronal = Dataset(fid, load=LOAD_STAGES["properties"]).acquisition_affines()

    assert np.allclose(axial[:3, 3], (-20.0, -20.0, 0.0))
    assert np.allclose(sagittal[:3, 3], (0.0, -20.0, 20.0))
    assert np.allclose(coronal[:3, 3], (-20.0, 0.0, 20.0))
    # image read axis, phase axis and slice normal in the patient frame (the
    # reconstruction stores these two transposed: VisuCoreOrientation swaps the first two)
    assert np.allclose(sagittal[:3, :2] / (40 / 256), np.column_stack(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))))
    assert np.allclose(sagittal[:3, 2], (-1.0, 0.0, 0.0))
    assert np.allclose(coronal[:3, :2] / (40 / 256), np.column_stack(((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))))
    assert np.allclose(coronal[:3, 2], (0.0, -1.0, 0.0))  # the slice normal, along which the offsets of a package grow


def test_acquisition_affine_follows_the_slice_offsets_between_packages(tmp_path):
    """The slice column is the step to the neighbouring slice of the same package.

    Five sagittal slices 2 mm apart (PV5.1 0.2H2/1, package 2) sit at
    x = 4, 2, 0, -2, -4: the offsets run along the slice normal, which for that
    package points towards -x in the patient frame.
    """
    fid = write_acquisition(
        tmp_path, version="PV 5.1", grad_matrix=[SAGITTAL] * 5, obj_order=[0, 2, 4, 1, 3], slice_offsets=[-4.0, -2.0, 0.0, 2.0, 4.0], fov=(60.0, 60.0), slice_sepn=2.0
    )

    affines = Dataset(fid, load=LOAD_STAGES["properties"]).acquisition_affines()

    assert np.allclose([affine[:3, 3] for affine in affines], [(4.0, -30.0, 30.0), (2.0, -30.0, 30.0), (0.0, -30.0, 30.0), (-2.0, -30.0, 30.0), (-4.0, -30.0, 30.0)])
    assert all(np.allclose(affine[:3, 2], (-2.0, 0.0, 0.0)) for affine in affines)


def test_acquisition_affine_of_a_3d_volume_spans_the_slab(tmp_path):
    """A 3-D acquisition is one object whose third axis is the partition step.

    PV5.1 0.2H2/10 (50 mm cube, 128 partitions) reconstructs to
    VisuCorePosition (-25, -25, -25); from PV6 on the partition grid is
    centred between partitions, half a step in.
    """
    volume = write_acquisition(tmp_path / "pv5", version="PV 5.1", grad_matrix=[AXIAL], obj_order=[0], slice_offsets=[0.0], fov=(50.0, 50.0, 50.0), matrix=(8, 8, 8), dim=3)
    affine = Dataset(volume, load=LOAD_STAGES["properties"]).acquisition_affine()

    assert np.allclose(affine[:3, 3], (-25.0, -25.0, -25.0))
    assert np.allclose(affine[:3, :3], np.diag((50 / 8, 50 / 8, 50 / 8)))

    later = write_acquisition(tmp_path / "pv6", grad_matrix=[AXIAL], obj_order=[0], slice_offsets=[0.0], fov=(50.0, 50.0, 50.0), matrix=(8, 8, 8), dim=3)
    assert np.allclose(Dataset(later, load=LOAD_STAGES["properties"]).acquisition_affine()[:3, 3], (-25.0, -25.0, -25.0 + 50 / 16))


def test_acquisition_affine_applies_the_declared_position_on_pv360(tmp_path):
    """PV360 writes the gradient matrix in the magnet frame; ACQ_patient_pos maps it to the subject (spec 5.6, 12).

    PV360 3.6 T2_TurboRARE, Head_Prone: a 2 degree tilt about y, phase offset
    0.9375 mm and a first slice at -5.7578 mm reconstructs to VisuCorePosition
    (10.1949, 10.9375, -5.4053), read axis (-0.9994, 0, -0.0349), phase axis
    (0, -1, 0) and a 1 mm slice step along (-0.0349, 0, 0.9994).  Declared
    Head_Supine instead, the same acquisition comes out rotated by pi about the
    bore (x and y negated) -- the position is not folded into the PV360 matrix,
    unlike PV5.1/6/7, so a reader applies it once (spec 12).
    """
    tilted = np.array([[-0.9993908270190958, 0.0, -0.03489949670250097], [0.0, 1.0, 0.0], [0.03489949670250097, 0.0, -0.9993908270190958]])
    prone = write_acquisition(
        tmp_path / "prone",
        version="PV-360.3.6",
        patient_pos="Head_Prone",
        grad_matrix=[tilted] * 2,
        obj_order=[0, 1],
        slice_offsets=[-5.7578, -4.7578],
        phase_offset=0.9375,
        fov=(20.0, 20.0),
        matrix=(256, 256),
        slice_thick=0.7,
    )
    affine = Dataset(prone, load=LOAD_STAGES["properties"]).acquisition_affine()

    assert np.allclose(affine[:3, 3], (10.1949, 10.9375, -5.4053), atol=1e-4)
    assert np.allclose(affine[:3, :2] / (20 / 256), np.column_stack(((-0.99939, 0.0, -0.0349), (0.0, -1.0, 0.0))), atol=1e-4)
    assert np.allclose(affine[:3, 2], (-0.0349, 0.0, 0.99939), atol=1e-4)

    supine = write_acquisition(
        tmp_path / "supine",
        version="PV-360.3.6",
        patient_pos="Head_Supine",
        grad_matrix=[tilted] * 2,
        obj_order=[0, 1],
        slice_offsets=[-5.7578, -4.7578],
        phase_offset=0.9375,
        fov=(20.0, 20.0),
        matrix=(256, 256),
        slice_thick=0.7,
    )
    assert np.allclose(Dataset(supine, load=LOAD_STAGES["properties"]).acquisition_affine()[:3, 3], (-10.1949, -10.9375, -5.4053), atol=1e-4)

    with pytest.raises(UnsupportedDatasetType, match="not a subject position"):
        Dataset(
            write_acquisition(tmp_path / "unknown", version="PV-360.3.6", patient_pos="Sideways", grad_matrix=[tilted], obj_order=[0], slice_offsets=[0.0]),
            load=LOAD_STAGES["properties"],
        ).acquisition_affines()


def test_acquisition_affine_of_a_pv360_head_supine_volume(tmp_path):
    """The one PV360 Head_Supine acquisition of the corpus, a compressed-sensing 3-D FLASH (PV-360.3.4).

    Read, phase and slice along -z, -x and +y of the magnet, offsets (0.7055,
    0.1527, 1.7413) mm, 20 x 15 x 12.5 mm; VISU_DICOM_PV_MATRIX . M_Head_Supine
    reproduces its VisuCorePosition (-7.6527, -7.9132, 10.7055) -- the far end of
    the slab, since the reconstruction stores the partitions reversed (spec 7.2)
    -- where the Head_Prone map puts x on the wrong side.
    """
    fid = write_acquisition(
        tmp_path,
        version="PV-360.3.4",
        patient_pos="Head_Supine",
        grad_matrix=[[[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        obj_order=[0],
        slice_offsets=[1.74134],
        read_offset=0.705521,
        phase_offset=0.152749,
        fov=(20.0, 15.0, 12.5),
        matrix=(128, 96, 80),
        dim=3,
    )

    affine = Dataset(fid, load=LOAD_STAGES["properties"]).acquisition_affine()

    assert np.allclose(affine[:3, 3], (-7.652749, 4.430535, 10.705521), atol=1e-5)
    assert np.allclose(affine[:3, 3] + 79 * affine[:3, 2], (-7.652749, -7.913215, 10.705521), atol=1e-5)
    assert np.allclose(affine[:3, :3], np.column_stack(((0.0, 0.0, -20 / 128), (15 / 96, 0.0, 0.0), (0.0, -12.5 / 80, 0.0))))


def test_acquisition_affine_falls_back_to_the_method_slice_packages(tmp_path):
    """acqp is the primary source; a custom sequence may maintain only the GeoObject values.

    Without ACQ_grad_matrix and the offsets, the method's PVM_SPackArr* carry the
    same geometry per package -- the slice centre spread by the slice distance --
    and say so in a warning.
    """
    method = {
        "PVM_Fov": np.array([50.0, 50.0]),
        "PVM_SPackArrGradOrient": np.array([AXIAL]),
        "PVM_SPackArrNSlices": np.array([4]),
        "PVM_SPackArrSliceOffset": np.array([0.0]),
        "PVM_SPackArrReadOffset": np.array([0.0]),
        "PVM_SPackArrPhase1Offset": np.array([2.0]),
        "PVM_SPackArrSliceDistance": np.array([5.0]),
    }
    fid = write_acquisition(
        tmp_path,
        version="PV 5.1",
        grad_matrix=[AXIAL] * 4,
        obj_order=[0, 2, 1, 3],
        slice_offsets=[0.0] * 4,
        fov=(50.0, 50.0),
        matrix=(64, 64),
        drop=("ACQ_grad_matrix", "ACQ_slice_offset", "ACQ_read_offset", "ACQ_phase1_offset", "ACQ_fov"),
        method=method,
    )

    with pytest.warns(RuntimeWarning, match="acqp carries no ACQ_grad_matrix"):
        affines = Dataset(fid, load=LOAD_STAGES["properties"]).acquisition_affines()

    assert np.allclose([affine[:3, 3] for affine in affines], [(-25.0, -27.0, -7.5), (-25.0, -27.0, -2.5), (-25.0, -27.0, 2.5), (-25.0, -27.0, 7.5)])


def test_acquisition_affine_refuses_a_spectroscopic_acquisition(tmp_path):
    experiment = tmp_path / "1"
    experiment.mkdir()
    write_jcampdx(
        experiment / "acqp",
        {
            "ACQ_dim": 1,
            "ACQ_dim_desc": "Spectroscopic",
            "ACQ_grad_matrix": np.eye(3).reshape(1, 3, 3),
            "ACQ_sw_version": ["<PV 6.0.1>"],
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "GO_block_size": "continuous",
            "BYTORDA": "little",
            "ACQ_size": np.array([16]),
            "NI": 1,
            "NR": 1,
            "PULPROG": ["<PRESS.ppg>"],
        },
    )
    write_jcampdx(experiment / "method", {"PVM_EncNReceivers": 1})
    write_binary(experiment / "fid", np.zeros(16), np.dtype("int32"))

    with pytest.raises(UnsupportedDatasetType, match="rather than 2 or 3 spatial"):
        Dataset(experiment / "fid", load=LOAD_STAGES["parameters"]).acquisition_affines()


@pytest.mark.parametrize("relative_path", ACQUISITIONS_WITH_RECONSTRUCTION)
def test_acquisition_affine_agrees_with_the_reconstruction(relative_path):
    """ParaVision's own reconstruction is the oracle for the acquisition geometry (#166).

    Spec 12: both end in the Visu/DICOM patient frame, so every slice's first
    voxel must land on its VisuCorePosition and the image axes on
    VisuCoreOrientation -- read and phase exchanged where the reconstruction
    transposed them, since the acquisition affine describes the k-space image;
    the slice normal up to the sign, which for a lone slice Visu completes
    right-handedly.  A 3-D slab may be cropped (anti-aliasing) and stored
    reversed, so there the first partition is compared at either end, to
    within half a partition.
    """
    raw = Dataset(CORPUS_ROOT / relative_path, load=LOAD_STAGES["properties"])
    try:
        affines = raw.acquisition_affines()
    except UnsupportedDatasetType as reason:
        pytest.skip(str(reason))
    image = Dataset(CORPUS_ROOT / relative_path.rsplit("/", 1)[0] / "pdata" / "1" / "2dseq", load=LOAD_STAGES["properties"])
    positions = np.atleast_2d(np.asarray(image.get("VisuCorePosition", [[]]) if False else image._parameter_value("VisuCorePosition", np.empty((0, 3))), dtype=float))
    orientations = np.atleast_2d(np.asarray(image._parameter_value("VisuCoreOrientation", np.empty((0, 9))), dtype=float))
    if positions.size == 0 or orientations.size == 0 or positions.shape[0] not in (1, len(affines)):
        pytest.skip("the reconstruction carries no per-slice geometry to compare with")
    dimension = int(raw["ACQ_dim"].value)
    sizes = np.atleast_1d(np.asarray(image._parameter_value("VisuCoreSize"), dtype=float))
    extents = np.atleast_1d(np.asarray(image._parameter_value("VisuCoreExtent"), dtype=float))
    fov = np.atleast_1d(np.asarray(raw["ACQ_fov"].value, dtype=float)) * 10

    for index, affine in enumerate(affines):
        orientation = orientations[min(index, orientations.shape[0] - 1)].reshape(3, 3)
        axes = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        # RECO_transposition stores the image with read and phase exchanged (spec 6.9); the k-space image is not
        assert np.allclose(axes[:, :2].T, orientation[:2], atol=1e-6) or np.allclose(axes[:, [1, 0]].T, orientation[:2], atol=1e-6), relative_path
        assert np.isclose(abs(axes[:, 2] @ orientation[2]), 1.0, atol=1e-6), relative_path

        # the reconstruction crops an anti-aliased field of view (PVM_AntiAlias), so the
        # first voxel of the k-space image sits (ACQ_fov - VisuCoreExtent) / 2 further out
        swapped = not np.allclose(axes[:, :2].T, orientation[:2], atol=1e-6)
        corner = affine[:3, 3] + sum(axes[:, axis] * (fov[axis] - extents[1 - axis if swapped else axis]) / 2 for axis in (0, 1))
        expected = positions[min(index, positions.shape[0] - 1)]
        if dimension == 2:
            assert np.allclose(corner, expected, atol=1e-3), (relative_path, index)
        else:
            step = affine[:3, 2]
            acquired = round(fov[2] / np.linalg.norm(step))
            ends = (corner + step * (acquired - sizes[2]) / 2, corner + step * ((acquired + sizes[2]) / 2 - 1))
            assert min(np.linalg.norm(end - expected) for end in ends) <= 0.5 * np.linalg.norm(step) + 1e-6, relative_path
