How to work with Bruker study?
===============================

.. code-block:: python

   from brukerapi.study import Study

    study = Study('path_to_study')

    #get list of scans (fid data sets) contained in the study
    study.scans

    #get list of recos (2dseq data sets) contained in the study
    study.recos

    #get data set from the study hierarchy
    study.get_dataset(scan_id='2', reco_id='1')

Data set obtained from ``Study`` object are empty by default, to access its content, the data set needs to be loaded. Either using the load function.

.. code-block:: python

    dataset = study.get_dataset(scan_id='2', reco_id='1')

    dataset.load()
    dataset.data
    dataset.get_value('VisuCoreSize')

Or using context manager.

.. code-block:: python

    with study.get_dataset(scan_id='2', reco_id='1') as dataset:
        dataset.data
        dataset.get_value('VisuCoreSize')



