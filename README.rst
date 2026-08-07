brukerapi-python
======================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3831320.svg
   :target: https://doi.org/10.5281/zenodo.3831320

.. image:: https://github.com/isi-nmr/brukerapi-python/workflows/CI/badge.svg
   :target: https://doi.org/10.5281/zenodo.3831320

.. image:: https://readthedocs.org/projects/bruker-api/badge/?version=latest
    :target: https://bruker-api.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


A Python package providing I/O interface for Bruker data sets.

tl;dr
========

Install using pip:

.. code-block:: shell

    pip install brukerapi

Load any **data set**:

.. code-block:: python

    from brukerapi.dataset import Dataset
    dataset = Dataset('{path}/2dseq')    # also supports fid, fid_proc.64, traj, and rawdata.jobN
    dataset.data                         # access data array
    dataset.get_value('VisuCoreSize')    # get a parameter value

Raw acquisitions and k-space
==============================

Raw Bruker acquisitions have format-dependent historical ``.data`` semantics.
For explicit code, use the representation that matches the task:

.. code-block:: python

    fid = Dataset('{path}/fid')
    fid.raw                 # decoded (sample, shot, receiver) acquisitions
    fid.kspace              # ordered FID k-space

    job = Dataset('{path}/rawdata.job0')
    job.raw                 # decoded (sample, shot, receiver) acquisitions
    job.kspace              # ordered Cartesian PV360 k-space, when metadata proves the layout
    job.to_kspace(bart=True)  # optional 16-axis BART layout

``Dataset.data`` remains backward compatible: FIDs expose their historical
ordered k-space view, whereas PV360 ``rawdata.jobN`` exposes its historical
decoded stream and emits a ``FutureWarning``. ``.kspace`` is not a
reconstruction API; EPI and non-Cartesian jobs require acquisition-specific
handling.

Frame-group metadata
====================

For 2dseq data, ``frame_group_values`` aligns values named by
``VisuGroupDepVals`` to the corresponding array axes. Returned arrays use
singleton dimensions where needed and therefore broadcast directly against
``dataset.data``:

.. code-block:: python

    echoes = dataset.frame_group_values['VisuAcqEchoTime']
    b_matrices = dataset.frame_group_values['VisuAcqDiffusionBMatrix']

``metadata`` provides normalized, grouped access to parsed subject, study,
series, equipment, and acquisition fields:

.. code-block:: python

    dataset.metadata['visu_study']['uid']
    dataset.metadata['visu_acq']['sequence_name']
    dataset.metadata['subject']['id']

Load an entire **study**:

.. code-block:: python

    from brukerapi.folders import Study
    study = Study('{path_to_study_folder}')
    dataset = study.get_dataset(exp_id='1', proc_id='1')

    dataset.data                         # Study loads datasets by default

Load a parametric file:

.. code-block:: python

   from brukerapi.jcampdx import JCAMPDX

   parameters = JCAMPDX('path_to_scan/method')
   
   TR = parameters.params["PVM_RepetitionTime"].value
   TR = parameters.get_value("PVM_RepetitionTime")





Features
========

* **I/O** interface for **fid** data sets
* **I/O** interface for **2dseq** data sets
* **I/O** interface for **rawdata** data sets
* **Random access** for **fid** and **2dseq** data sets
* **Split** operation implemented over **2dseq** data sets
* **Filter** operation implemented over Bruker **folders** (allowing you to work with a subset of your study only)
* ParaVision 5.1, 6.0.1, 7.0.0, and 360 metadata and binary-layout support
* Metadata-based fallback inference for custom Cartesian, EPI, radial/UTE, spiral, ZTE, CSI, and spectroscopy sequences

Examples
========

* How to `read <examples/read_fid.ipynb>`_ a Bruker fid, 2dseq, or rawdata file
* How to `split slice packages <examples/split_sp_demo.ipynb>`_ of a 2dseq data set
* How to `split FG_ECHO <examples/split_fg_echo_demo.ipynb>`_ of a 2dseq data set
* How to `split FG_ISA <examples/examples/split_fg_isa_demo.ipynb>`_ of a 2dseq data set

Documentation
==============

Online `documentation <https://bruker-api.readthedocs.io/en/latest/>`_ of the API is available at Read The Docs.


Install
=======
Using pip:

.. code-block:: shell

    pip install brukerapi

From source:

.. code-block:: shell

    git clone https://github.com/isi-nmr/brukerapi-python.git
    cd brukerapi-python
    python -m pip install -e .[dev]

Testing
========
To ensure reliability, every commit to this repository is tested against the following, publicly available
data sets:

* `BrukerAPI test data set (Bruker ParaVision v5.1) <https://doi.org/10.5281/zenodo.3899268>`_
* `BrukerAPI test data set (Bruker ParaVision v6.0.1) <https://doi.org/10.5281/zenodo.3894651>`_
* BrukerAPI test data set for ParaVision v7.0.0 (Zenodo DOI collection ``10.5281/zenodo.4522220``)
* `PV360 standard data <https://github.com/cecilyen/PV360_StdData>`_

The corpus download is opt-in for local runs:

.. code-block:: shell

    python -m pytest test --download_test_data

Without that flag, pytest uses any corpus already present under ``test/test_data`` and
skips unavailable collections.

File format reference
=====================

`Bruker ParaVision Raw Data Format
<https://github.com/gdevenyi/brkraw-legacy/blob/main/FILE_FORMAT.md>`_ is the source of truth
for file-format parsing, binary layouts, dataset typing, and metadata-driven acquisition
scheme inference in this project.

Compatibility
=============

Tested releases are ParaVision 5.1, 6.0.1, 7.0.0, and PV360 3.x. Supported
primary binaries are ``fid``, ``fid_proc.64``, ``2dseq``, ``traj``,
``rawdata.jobN``, and named jobs declared by ``ACQ_jobs`` (for example,
``rawdata.Navigator`` or ``rawdata.echoNavigator``). Known ``fid.spiral``,
``fid.navFid``, and ``fid.orig`` files are exposed as auxiliary subdatasets of
their parent ``fid``; they are not accepted as standalone primary datasets.
TopSpin/NMR ``ser`` is intentionally unsupported.

Known pulse-program names use dedicated layouts. For custom sequences the
reader also infers common acquisition families from metadata; callers can pass
``scheme_id=`` when inference is ambiguous. Rawdata is returned as complex
ordered samples, not as reconstructed k-space. See the compatibility page in
the documentation for behavior and current reconstruction limitations.
