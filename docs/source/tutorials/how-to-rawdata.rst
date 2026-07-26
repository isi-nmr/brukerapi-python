How to load a PV360 rawdata job?
=================================

ParaVision 360 experiments store raw acquisitions in ``rawdata.jobN`` rather
than ``fid``. Pass either the job file or its experiment directory; the latter
selects the lowest-numbered job:

.. code-block:: python

   from brukerapi.dataset import Dataset

   rawdata = Dataset('path/to/experiment/rawdata.job0')
   # or: rawdata = Dataset('path/to/experiment')

Use ``raw`` for a consistent decoded acquisition stream. Its axes are
``(sample, shot, receiver)``:

.. code-block:: python

   acquisitions = rawdata.raw

For validated Cartesian acquisition layouts, ``kspace`` provides the
metadata-reordered array. BART consumers can request its 16-axis layout:

.. code-block:: python

   kspace = rawdata.kspace
   bart_kspace = rawdata.to_kspace(bart=True)

``rawdata.data`` is retained for compatibility and exposes the historical
decoded job layout. It emits ``FutureWarning`` because that layout differs from
FID ``data``. Prefer ``raw`` or ``kspace`` in new applications. EPI and
non-Cartesian jobs are not reconstructed by ``kspace``.

Job discovery honours ``ACQ_ScanPipeJobSettings``. Jobs marked
``STORE_discard`` are deliberately omitted because ParaVision does not write a
corresponding ``rawdata.jobN`` file. For a present job, the reader sizes its
data from the settings record's ``nStoredScans`` and exposes the relevant
per-job metadata:

.. code-block:: python

   settings = rawdata.rawdata_job_settings
   scans = rawdata.rawdata_stored_scans
   receivers = rawdata.rawdata_channels

If the settings and ``ACQ_jobs`` records disagree about the stored scan count,
the settings value is used and a ``RuntimeWarning`` identifies the mismatch.
