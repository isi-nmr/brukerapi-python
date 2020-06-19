*********************************
How to filter Bruker directories?
*********************************

In all tutorials we use our publicly available
`ParaVision v6.0.1 dataset <https://doi.org/10.5281/zenodo.3894651>`_.

First we creeate a `Study` object and print its structure.

.. code-block:: python

   from brukerapi.study import Study

   study = Study('20200612_094625_lego_phantom_3_1_2')

   #print structure of directory
   study.print()

.. code-block:: shell

    20200612_094625_lego_phantom_3_1_2 [Study]
       └-- 1 [Experiment]
         └-- fid [Dataset]
         └-- acqp [JCAMPDX]
         └-- uxnmr.par [JCAMPDX]
         └-- AdjStatePerScan [JCAMPDX]
         └-- configscan [JCAMPDX]
         └-- pdata [Folder]
           └-- 1 [Processing]
             └-- procs [JCAMPDX]
             └-- reco [JCAMPDX]
             └-- 2dseq [Dataset]
             └-- id [JCAMPDX]
             └-- visu_pars [JCAMPDX]
         └-- visu_pars [JCAMPDX]
         └-- specpar [JCAMPDX]
         └-- method [JCAMPDX]
       └-- 2 [Experiment]
         └-- fid [Dataset]
         └-- acqp [JCAMPDX]
         └-- uxnmr.par [JCAMPDX]
         └-- AdjStatePerScan [JCAMPDX]
         └-- configscan [JCAMPDX]
         └-- pdata [Folder]
           └-- 1 [Processing]
             └-- procs [JCAMPDX]
             └-- reco [JCAMPDX]
             └-- 2dseq [Dataset]
             └-- id [JCAMPDX]
             └-- visu_pars [JCAMPDX]
         └-- visu_pars [JCAMPDX]
         └-- specpar [JCAMPDX]
         └-- method [JCAMPDX]
    .
    .
    .

As we can see, there are all possible brukerapi object in the structure.
Now we can use the `ParameterFilter` to get `Datasets` measured
using the RARE pulse sequence only.

.. code-block:: python

    study.filter(parameter='PULPROG', operator='==', value='<RARE.ppg>')
    study.filter(type=Dataset)
    study.print()

The resulting folder structure matches our needs now:

.. code-block:: shell

     20200612_094625_lego_phantom_3_1_2 [Study]
       └-- 8 [Experiment]
         └-- fid [Dataset]
       └-- 47 [Experiment]
         └-- fid [Dataset]
