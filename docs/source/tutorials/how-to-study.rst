How to work with Bruker study?
===============================

.. code-block:: python

   from brukerapi.folders import Study

   study = Study('path_to_study')

   # get list of experiments contained in the study
   study.get_experiment_list()

   # get list of processing folders contained in the study
   study.get_processing_list()

   # get a data set from the study hierarchy
   study.get_dataset(exp_id='2', proc_id='1')

Data set obtained from ``Study`` object are empty by default, to access its content, the data set needs to be loaded. Either using the load function.

.. code-block:: python

    dataset = study.get_dataset(exp_id='2', proc_id='1')

    dataset.load()
    dataset.data
    dataset.get_value('VisuCoreSize')

Or using context manager.

.. code-block:: python

    with study.get_dataset(exp_id='2', proc_id='1') as dataset:
        dataset.data
        dataset.get_value('VisuCoreSize')


