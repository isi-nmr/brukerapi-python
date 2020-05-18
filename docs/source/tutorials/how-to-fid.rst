How to load a fid file?
===============================

The ``Dataset`` constructor accepts both a path to directory containing a fid file, or a path to the fid file.

.. code-block:: python

   from brukerapi.dataset import Dataset

    dataset = Dataset('path_to_fid/')

    dataset = Dataset('path_to_fid/fid')

A `Dataset` object is primarily an interface to the data contained in the fid file.

.. code-block:: python

   data = dataset.data

Data is typically and n-dimensional array, the physical meaning of individual dimensions is stored in ``dim_type`` property.

.. code-block:: python

   >> dataset.dim_type
   >> ['kspace_encode_step_0', 'kspace_encode_step_1', 'slice', 'repetition', 'channel']

It is possible to directly access some of the most wanted measurement parameters.

.. code-block:: python

   >> dataset.TE
   >> 3.0
   >> dataset.TR
   >> 15.0
   >> dataset.flip_angle
   >> 10.0

Both acqp and method files are used to construct a fid data set, it is possible to get value of any of hereby stored
parameters.

.. code-block:: python

   >> dataset.get_value('PVM_Matrix')
   >> [192 192]
   >> dataset.get_value('ACQ_dim_desc')
   >> ['Spatial' 'Spatial']
