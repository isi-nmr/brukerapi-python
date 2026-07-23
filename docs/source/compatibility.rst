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
  layout; they are not reconstructed into image-space or generalized k-space.
* 2dseq values are scaled as ``stored * slope + offset``. Visu slopes/offsets
  take precedence, with RECO values as fallback.
* PV7/PV360 2dseq geometry uses the version-independent Visu geometry fields.
  Reversed disk slice order is normalized on read.
* ``COMPLEX_IMAGE``/``FG_COMPLEX`` reconstructions are returned as complex
  arrays by default. Use ``combine_complex=False`` to retain the real frame
  axis.
