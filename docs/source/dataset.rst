Dataset
========

Construction and load stages
----------------------------

``Dataset`` accepts a supported binary path or an experiment/processing
directory containing ``fid`` or ``2dseq``. Loading is eager by default. Use
``LOAD_STAGES`` for parameters-only or properties-only work, and ``mmap=True``
for random access.

Important options include ``scheme_id`` for acquisition-family overrides,
``scale`` for 2dseq pixel scaling, and ``combine_complex`` for complex 2dseq
frame assembly.

Reports exclude internal dataset typing fields and preserve property order.
Malformed query expressions raise ``FilterEvalFalse`` instead of leaking raw
``eval`` exceptions.

Raw acquisition and k-space views
---------------------------------

Use the explicit raw-data views for FIDs and supported PV360 jobs:

.. code-block:: python

   fid.raw                    # (sample, shot, receiver), acquisition order
   fid.kspace                 # ordered FID k-space
   rawdata.raw                # (sample, shot, receiver), acquisition order
   rawdata.kspace             # validated Cartesian PV360 k-space
   rawdata.to_kspace(bart=True)  # BART's 16-axis layout

``Dataset.data`` is retained for compatibility. It is ordered k-space for
FIDs, but the historical decoded job stream for PV360 ``rawdata.jobN`` files.
Accessing the latter emits ``FutureWarning``; use ``raw`` or ``kspace`` for
new code. EPI and non-Cartesian PV360 jobs are intentionally not reconstructed
by ``kspace``.

.. _raw-acquisition-geometry:

Raw acquisition geometry
------------------------

Spatial FID and PV360 rawdata datasets can generate affine matrices before an
image has been reconstructed. ``acquisition_affines()`` returns one matrix per
2-D slice; ``acquisition_affine(index)`` returns a selected matrix. A 3-D
acquisition has one matrix whose third column is the partition step.

.. code-block:: python

   acquisition = Dataset("path/to/fid")
   affines = acquisition.acquisition_affines()
   first_slice_affine = acquisition.acquisition_affine(0)

Each 4x4 matrix maps a voxel index in the image produced by Fourier transforming
the encoded k-space matrix to millimetres in the Visu/DICOM patient frame:

.. code-block:: python

   patient_position = first_slice_affine @ [i, j, 0, 1]

The first two columns contain the read and phase voxel steps, the third contains
the slice or partition step, and the last column locates voxel ``(0, 0, 0)``.
This is the same patient frame used by a reconstructed 2dseq
``Dataset.affine``, so acquisition and reconstruction geometry can be compared
directly. To produce a NIfTI RAS affine, negate the patient x and y axes:

.. code-block:: python

   import numpy as np

   nifti_affine = np.diag([-1, -1, 1, 1]) @ first_slice_affine

Geometry is derived primarily from ``ACQ_grad_matrix``, the ``ACQ_*_offset``
values and ``ACQ_fov``. Method slice-package parameters are used as fallbacks
when acquisition parameters are absent, with a ``RuntimeWarning`` because they
are not equivalent for every ParaVision version. Spectroscopy/CSI, non-spatial
or non-2-D/3-D acquisitions, missing orientation metadata, and PV360 data with
an unknown subject position raise ``UnsupportedDatasetType``.

The returned list is indexed by physical slice, not by every acquisition
object. If echoes, repetitions, or other dimensions make ``NI`` differ from
``NSLICES``, their object-to-slice nesting can only be recovered from the
reconstruction metadata.

Metadata views
--------------

``Dataset.frame_group_values`` aligns values declared by
``VisuGroupDepVals`` to the corresponding 2dseq axes, with singleton axes for
broadcasting. It supports, for example, per-echo echo times and per-diffusion
B matrices:

.. code-block:: python

   echo_times = dataset.frame_group_values["VisuAcqEchoTime"]
   b_matrices = dataset.frame_group_values["VisuAcqDiffusionBMatrix"]

``Dataset.metadata`` provides normalized grouped access to parsed Visu and
``SUBJECT_*`` parameters, including ``visu_subject``, ``visu_study``,
``visu_series``, ``visu_equipment``, ``visu_acq``, and ``subject``.

.. code-block:: python

   dataset.metadata["visu_study"]["uid"]
   dataset.metadata["visu_acq"]["sequence_name"]
   dataset.metadata["subject"]["id"]

.. automodule:: brukerapi.dataset
    :noindex:
    :members:
    :special-members:
