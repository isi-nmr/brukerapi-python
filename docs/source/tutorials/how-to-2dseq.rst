How to load a 2dseq file?
===============================

The ``Dataset`` constructor accepts both a path to a directory containing a
``2dseq`` file, or a path to the ``2dseq`` file itself.

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

Frame-group-dependent metadata is available in data-axis order. For example,
per-echo times and diffusion B matrices can be broadcast with the image:

.. code-block:: python

   echo_times = dataset.frame_group_values['VisuAcqEchoTime']
   b_matrices = dataset.frame_group_values['VisuAcqDiffusionBMatrix']

Normalized study and acquisition metadata is grouped under ``metadata``:

.. code-block:: python

   study_uid = dataset.metadata['visu_study']['uid']
   sequence_name = dataset.metadata['visu_acq']['sequence_name']

Multiple slice packages may have unequal depths. Access package-specific
in-memory datasets, including their own geometry, with:

.. code-block:: python

   for package in dataset.slice_packages:
       print(package.data.shape, package.affine, package.resolution)

This also works for older PV5.1 datasets that do not have the optional
``VisuCoreSlicePacks*`` parameters. In that case packages are inferred from
contiguous frames with the same orientation. This inference cannot distinguish
two immediately adjacent packages that share an orientation; PV6 and later
datasets use the explicit package descriptors when available.

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

Which of these a dataset carries is ParaVision-version dependent, so a property
whose recipe does not resolve is simply not set. ``get`` reads one with a
default instead of an ``AttributeError``; a name that is not a property at all
still raises, so a misspelling does not quietly become the default.

.. code-block:: python

   >> dataset.get('TE')
   >> 3.0
   >> dataset.get('TE', 'n/a')      # where the scan records no echo time
   >> 'n/a'
   >> dataset.get('TEE')
   AttributeError: 'Dataset' object has no attribute 'TEE'

The ``visu_pars`` file is used to construct a 2dseq dataset. ``reco`` and
``d3proc`` are optional compatibility sources for legacy/minimal instances;
they supply reconstruction word type and image-size metadata only when Visu
metadata is absent. Any loaded parameter can be accessed directly.

.. code-block:: python

   >> dataset.get_value('VisuCoreSize')
   >> [192 192]
   >> dataset.get_value('VisuCoreDim')
   >> 2
