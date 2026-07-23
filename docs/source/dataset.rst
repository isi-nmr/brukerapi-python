Dataset
========

Construction and load stages
----------------------------

``Dataset`` accepts a supported binary path or an experiment/processing
directory containing ``fid`` or ``2dseq``. Loading is eager by default. Use
``LOAD_STAGES`` for parameters-only or properties-only work, and ``mmap=True``
for random access.

Important options include ``scheme_id`` for acquisition-family overrides,
``scale`` for 2dseq pixel scaling, and ``combine_complex`` for complex 2dseq
frame assembly.

Reports exclude internal dataset typing fields and preserve property order.
Malformed query expressions raise ``FilterEvalFalse`` instead of leaking raw
``eval`` exceptions.

.. automodule:: brukerapi.dataset
    :noindex:
    :members:
    :special-members:
