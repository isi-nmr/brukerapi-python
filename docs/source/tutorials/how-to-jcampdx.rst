How to read parametric files
===============================


Simply by importing the JCAMPDX class we can read e.g. the method file.

.. code-block:: python

   from brukerapi.jcampdx import JCAMPDX

   parameters = JCAMPDX('path_to_scan/method')
   
   TR = data.params["PVM_RepetitionTime"].value # This way
   TR = data.get_value("PVM_RepetitionTime") # Or this way


The loaded parameters can be simply cast into dictionary for easier manipulation

.. code-block:: python

   parameters_dict = parameters.to_dict()
   TR = parameters_dict["PVM_RepetitionTime"]["value"]

