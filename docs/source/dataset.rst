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
