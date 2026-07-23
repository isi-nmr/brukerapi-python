Data for testing
================

The automated test harness knows about four public collections:

* ParaVision 5.1: ``0.2H2`` (Zenodo)
* ParaVision 6.0.1: ``20200612_094625_lego_phantom_3_1_2`` (Zenodo)
* ParaVision 7.0.0: ``20210128_122257_LEGO_PHANTOM_API_TEST_1_1`` (Zenodo)
* PV360 3.6 standard data: `cecilyen/PV360_StdData
  <https://github.com/cecilyen/PV360_StdData>`_

Downloads are opt-in:

.. code-block:: shell

   python -m pytest test --download_test_data

Select one collection with ``--test_data``, for example:

.. code-block:: shell

   python -m pytest test --test_data PV360_StdData --download_test_data

Without ``--download_test_data``, collection never performs a network fetch.
Tests use collections already present under ``test/test_data`` and omit
unavailable ones. GitHub CI caches both the Zenodo archives and the pinned
PV360 checkout. Property snapshots are stored in ``test/config`` for PV5,
PV6, PV7, and PV360.
