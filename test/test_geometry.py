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
from test.synthetic import axial_orientation, stacked_positions, visu_pars_records, write_2dseq, write_binary, write_jcampdx

SAGITTAL_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
PV360_NIFTI_ROOT = Path("test/test_data/PV360_StdData")
# Every reconstruction that ships a NIfTI export, rather than a fixed list, so
# a dataset added to the corpus is checked without editing this file.
PV360_NIFTI_EXPORTS = sorted(directory.parent.relative_to(PV360_NIFTI_ROOT).as_posix() for directory in PV360_NIFTI_ROOT.glob("*/pdata/*/nifti") if any(directory.glob("*.nii")))


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
