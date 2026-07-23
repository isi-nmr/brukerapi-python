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

By default ``Study`` recursively constructs and loads its datasets, so the
returned dataset can be used immediately:

.. code-block:: python

    dataset = study.get_dataset(exp_id='2', proc_id='1')

    dataset.data
    dataset.get_value('VisuCoreSize')

For metadata-only traversal, provide a dataset state with the properties load
stage:

.. code-block:: python

    from brukerapi.dataset import LOAD_STAGES

    study = Study(
        'path_to_study',
        dataset_state={'load': LOAD_STAGES['properties']}
    )

