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

Stored integer pixels are scaled using the Visu slope and offset (or RECO
fallbacks). Complex reconstructions are assembled from ``FG_COMPLEX`` frames
by default, and reversed on-disk slice order is normalized:

.. code-block:: python

   dataset = Dataset('path/to/2dseq')
   scaled_or_complex_data = dataset.data

   raw_frames = Dataset(
       'path/to/2dseq',
       scale=False,
       combine_complex=False,
   ).data

Multiple slice packages may have unequal depths. Access package-specific
in-memory datasets, including their own geometry, with:

.. code-block:: python

   for package in dataset.slice_packages:
       print(package.data.shape, package.affine, package.resolution)

Use memory-mapped random access when only a sub-array is needed:

.. code-block:: python

   dataset = Dataset('path/to/2dseq', mmap=True)
   frame = dataset.data[:, :, 0]

It is possible to directly access some of the most wanted measurement parameters.

.. code-block:: python

   >> dataset.TE
   >> 3.0
   >> dataset.TR
   >> 15.0
   >> dataset.flip_angle
   >> 10.0

The ``visu_pars`` file is used to construct a 2dseq dataset. Any parameter
stored in that file can also be accessed directly.

.. code-block:: python

   >> dataset.get_value('VisuCoreSize')
   >> [192 192]
   >> dataset.get_value('VisuCoreDim')
   >> 2
