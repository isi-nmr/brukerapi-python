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
