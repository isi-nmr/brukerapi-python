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
   >> ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'repetition', 'channel']

``Dataset.data`` contains ordered raw k-space, not a reconstructed image.
RARE/EPI line ordering is applied, while ramp-sampling regridding remains a
downstream reconstruction step. Real-only ``AQ_mod=qf`` data stays real;
quadrature acquisition modes are returned as complex arrays.

Use the explicit views in new code: ``dataset.raw`` is the decoded acquisition
stream with axes ``(sample, shot, receiver)``; ``dataset.kspace`` is the same
ordered k-space exposed through ``data`` for compatibility.

When an experiment has a reconstruction, its ``reco`` declaration is used to
determine EPI continuous-train handling and odd-line reversal
(``RECO_inp_order=REV_ALT_ROWS``). The conventional first reconstruction is
used by default. Select another reconstruction explicitly when its input-order
metadata is the one that corresponds to the data being read:

.. code-block:: python

   dataset = Dataset(
       'path/to/fid',
       reco_path='path/to/pdata/2/reco',
   )

For a FID without a reconstruction, the reader falls back to the acquisition
declaration and, where necessary, the pulse-program scheme inference. A
``RuntimeWarning`` is emitted if the selected ``reco`` and scheme inference
disagree; the declared reconstruction value is used.

For custom pulse-program names the scheme is inferred from acquisition
metadata. An explicit override is available when inference is ambiguous:

.. code-block:: python

   dataset = Dataset('path/to/fid', scheme_id='RADIAL')

For a 2-D or 3-D spatial acquisition, generate the geometry of the image that
a Fourier transform of the encoded k-space matrix would produce:

.. code-block:: python

   slice_affines = dataset.acquisition_affines()
   first_slice_affine = dataset.acquisition_affine(0)

The matrices map voxel indices to millimetres in the Visu/DICOM patient frame,
the same frame as a reconstructed 2dseq ``affine``. There is one matrix per
2-D slice and one for a 3-D volume. See :ref:`raw-acquisition-geometry` for the
coordinate convention, limitations, and NIfTI conversion.

Random-access ``mmap=True`` is currently supported for ``2dseq`` only. Load a
FID normally, then select the desired k-space array slice.

Known ``fid.spiral``, ``fid.navFid``, and ``fid.orig`` companions are loaded
under ``dataset.fid_companions``. They are auxiliary subdatasets rather than
standalone primary datasets. TopSpin/NMR ``ser`` is not supported.

It is possible to directly access some of the most wanted measurement parameters.

.. code-block:: python

   >> dataset.TE
   >> 3.0
   >> dataset.TR
   >> 15.0
   >> dataset.flip_angle
   >> 10.0

Both ``acqp`` and ``method`` files are used to construct a fid dataset. Any
parameter stored in those files can also be accessed directly.

.. code-block:: python

   >> dataset.get_value('PVM_Matrix')
   >> [192 192]
   >> dataset.get_value('ACQ_dim_desc')
   >> ['Spatial' 'Spatial']
