Compatibility
=============

Tested ParaVision releases
--------------------------

The committed and CI-managed corpora cover ParaVision 5.1, 6.0.1, 7.0.0, and
PV360 3.x. PV360 1.1 recipes are supported but the repository currently has no
PV360 1.1 binary corpus.

Dataset types
-------------

Supported primary binary names are:

* ``fid``
* ``fid_proc.64``
* ``2dseq``
* ``traj``
* ``rawdata.jobN``
* ``rawdata.Navigator``

Unknown suffixes such as ``fid.npz`` and ``2dseq.json`` are rejected instead
of being interpreted as scanner data. ``fid.spiral``, ``fid.navFid``, and
``fid.orig`` are auxiliary companions and are available through
``dataset.fid_companions`` after loading the parent ``fid``. TopSpin/NMR
``ser`` is intentionally not supported.

Acquisition schemes
-------------------

Dedicated layouts exist for common Cartesian 2D/3D, field-map, RARE, EPI,
diffusion EPI, radial/UTE, spiral, ZTE, spectroscopy, and CSI families.
Known pulse-program names are matched first. Custom names then use metadata
such as ``ACQ_dim``, ``ACQ_dim_desc``, ``ACQ_size``, ``PVM_EncMatrix``,
``NPro``, and trajectory parameters. If a custom sequence remains ambiguous,
pass a supported ``scheme_id`` explicitly:

.. code-block:: python

   dataset = Dataset("path/to/fid", scheme_id="RADIAL")

Data contract and limitations
-----------------------------

* FID data is returned as ordered raw k-space. ``AQ_mod=qf`` remains real;
  quadrature modes are assembled as complex data.
* RARE/EPI phase-line ordering and EPI odd-line mirroring are applied.
  Ramp-sampling regridding and ``RECO_qopts`` corrections are not full
  reconstruction steps and remain the caller's responsibility.
* Rawdata jobs are returned as complex ordered samples in their stored job
  layout through ``data``. Prefer ``raw`` for the normalized
  ``(sample, shot, receiver)`` acquisition stream and ``kspace`` for validated
  Cartesian PV360 layout conversion.
* 2dseq values are scaled as ``stored * slope + offset``. Visu slopes/offsets
  take precedence, with RECO values as fallback.
* PV7/PV360 2dseq geometry uses the version-independent Visu geometry fields.
  Reversed disk slice order is normalized on read.
* ``COMPLEX_IMAGE``/``FG_COMPLEX`` reconstructions are returned as complex
  arrays by default. Use ``combine_complex=False`` to retain the real frame
  axis.
* ``RECO_transposition`` records what the reconstruction already did and is
  not re-applied. Per-frame ``VisuCoreTransposition`` describes how a frame is
  stored, so a frame whose two exchanged dimensions differ in length is read in
  its stored shape and swapped back on read, and restored on write. FID
  ``ACQ_obj_order`` is normalized on read; the axis it orders is labelled
  ``object``, because ``NI`` counts acquisition objects (slices x echoes x
  movie frames), not slices.
* d3proc is an optional compatibility source for legacy/minimal 2dseq word
  type and image-size metadata after Visu and RECO metadata have been tried.
* ``Dataset.metadata`` groups the Visu parameters the way the format defines
  them -- administration, subject, study, series, equipment and acquisition --
  and the ``SUBJECT_*`` parameters of the study file, which ``subject`` in
  ``add_parameters=`` loads.
* Version-dependent behaviour is selected on a parsed ``pv_version``
  (``5.1``, ``6.0.1``, ``7.0.0``, ``360.3.7``, ...) rather than on an exact
  version string, so an unlisted point release is not silently unsupported.
* ``Dataset.affine`` is a voxel-index to patient-coordinate transform derived
  from ``VisuCorePosition``/``VisuCoreOrientation``: index ``(0, 0, 0)`` maps
  onto the centre of the first voxel transferred, and the slice column carries
  the measured direction and spacing between slice centres. It is expressed in
  the Visu/DICOM patient frame (R->L, A->P, F->H); a NIfTI writer converts with
  ``np.diag([-1, -1, 1, 1]) @ affine``, and the ParaVision user-interface frame
  needs both ends transformed. Frames that are not purely spatial
  (spectroscopy, CSI) have no image geometry and raise
  ``UnsupportedDatasetType`` rather than returning an identity matrix. A
  dataset with several slice packages cannot be described by one affine; it
  warns, and ``affine_of_package(i)`` or ``slice_packages`` gives the
  per-package transform.
