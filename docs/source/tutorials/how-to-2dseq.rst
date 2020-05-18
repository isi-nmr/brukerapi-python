How to load a 2dseq file?
===============================

The ``Dataset`` constructor accepts both a path to directory containing a fid file, or a path to the 2dseq file.

.. code-block:: python

   from brukerapi.dataset import Dataset

    dataset = Dataset('path_to_2dseq/')

    dataset = Dataset('path_to_2dseq/2dseq')

A `Dataset` object is primarily an interface to the data contained in the 2dseq file.

.. code-block:: python

   data = dataset.data

Data is typically and n-dimensional array, the physical meaning of individual dimensions is stored in ``dim_type`` property.

.. code-block:: python

   >> dataset.dim_type
   >> ['spatial', 'spatial', 'FG_SLICE']

It is possible to directly access some of the most wanted measurement parameters.

.. code-block:: python

   >> dataset.TE
   >> 3.0
   >> dataset.TR
   >> 15.0
   >> dataset.flip_angle
   >> 10.0

The visu_pars file is used to construct a 2dseq data set, it is possible to get value of any of hereby stored parameters.

.. code-block:: python

   >> dataset.get_value('VisuCoreSize')
   >> [192 192]
   >> dataset.get_value('VisuCoreDim')
   >> 2