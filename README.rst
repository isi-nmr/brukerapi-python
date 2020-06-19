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
    dataset = Dataset('{path}/2dseq')    # create data set, works for fid, 2dseq, rawdata.x, ser
    dataset.data                         # access data array
    dataset.VisuCoreSize                 # get a value of a single parameter

Load an entire **study**:

.. code-block:: python

    from brukerapi.folders import Study
    study = Study('{path_to_study_folder}')
    dataset = study.get_dataset(exp_id='1', proc_id='1')

    # get_dataset returns an empty dataset
    # in order to load data into the data set, you can either use the context manager:

    with dataset as d:
        d.data                         # access data array
        d.VisuCoreSize                 # get a value of a parameter

    # or the load function
    dataset.load()
    dataset.data                       # access data array
    dataset.VisuCoreSize               # get a value of a single parameter




Features
========

* **I/O** interface for **fid** data sets
* **I/O** interface for **2dseq** data sets
* **I/O** interface for **ser** data sets
* **I/O** interface for **rawdata** data sets
* **Random access** for **fid** and **2dseq** data sets
* **Split** operation implemented over **2dseq** data sets
* **Filter** operation implemented over Bruker **folders** (allowing you to work with a subset of your study only)

Examples
========

* How to `read <examples/read_fid.ipynb>`_ a Bruker fid, 2dseq, rawdata, or ser file
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
    python setup.py build
    python setup.py install

Testing
========
To ensure reliability, every commit to this repository is tested against the following, publicly available
data sets:

* `BrukerAPI test data set (Bruker ParaVision v5.1) <https://doi.org/10.5281/zenodo.3899268>`_
* `BrukerAPI test data set (Bruker ParaVision v6.0.1) <https://doi.org/10.5281/zenodo.3894651>`_
* `bruker2nifti_qa data set <https://gitlab.com/naveau/bruker2nifti_qa>`_

Compatibility
=============

The API was tested using various data sets obtained by **ParaVision 5.1**, **6.0.1** and **360**. It it is compatible
with the following data set types from individual ParaVision versions.

ParaVision v5.1
"""""""""""""""
Compatible data set types:

* **fid**
* **2dseq**
* **rawdata.job0**
* **rawdata.Navigator**

Compatible pulse sequences for **fid** data sets:

* FLASH.ppg
* MGE.ppg
* MSME.ppg
* RARE.ppg
* FAIR_RARE.ppg
* RAREVTR.ppg
* RAREst.ppg
* MDEFT.ppg
* FISP.ppg
* FLOWMAP.ppg
* DtiStandard.ppg
* EPI.ppg
* FAIR_EPI.ppg
* DtiEpi.ppg
* T1_EPI.ppg
* T2_EPI.ppg
* T2S_EPI.ppg
* SPIRAL.ppg
* DtiSpiral.ppg
* UTE.ppg
* UTE3D.ppg
* ZTE.ppg
* CSI.ppg
* FieldMap.ppg
* NSPECT.ppg
* PRESS.ppg
* STEAM.ppg
* igFLASH.ppg

ParaVision v6.0.1
"""""""""""""""""
Compatible data set types:

* **fid**
* **2dseq**
* **rawdata.job0**
* **rawdata.Navigator**

Compatible pulse sequences for **fid** data sets:

* FLASH.ppg,
* FLASHAngio.ppg
* IgFLASH.ppg
* MGE.ppg
* MSME.ppg
* RARE.ppg
* FAIR_RARE.ppg
* RAREVTR.ppg
* RAREst.ppg
* MDEFT.ppg
* FISP.ppg
* FLOWMAP.ppg
* DtiStandard.ppg
* EPI.ppg
* FAIR_EPI.ppg
* CASL_EPI.ppg
* DtiEpi.ppg
* T1_EPI.ppg
* T2_EPI.ppg
* T2S_EPI.ppg
* SPIRAL.ppg
* DtiSpiral.ppg
* UTE.ppg
* UTE3D.ppg
* ZTE.ppg
* CSI.ppg
* FieldMap.ppg
* SINGLEPULSE.ppg
* NSPECT.ppg
* EPSI.ppg
* PRESS.ppg
* STEAM.ppg
* ISIS.ppg
* CPMG.ppg
* RfProfile.ppg

ParaVision v360
"""""""""""""""
Compatible data set types:

* **2dseq**
* **rawdata.job0**
* **rawdata.Navigator**

