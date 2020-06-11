from brukerapi.jcampdx import JCAMPDX
import numpy as np


def test_jcampdx(test_jcampdx_data, data_path):

    j = JCAMPDX(data_path / test_jcampdx_data['path'])
    for key, ref in test_jcampdx_data['parameters'].items():
        parameter_test  = j.get_parameter(key)
        size_test= parameter_test.size
        value_test= parameter_test.value
        type_test = value_test.__class__

        value_ref = ref['value']
        size_ref = ref['size']
        type_ref = ref['type']

        #test SIZE
        if size_ref == 'None':
            size_ref = None
        if isinstance(size_ref, list):
            size_ref = tuple(size_ref)
        elif isinstance(size_ref, int):
            size_ref = (size_ref,)
        assert size_ref == size_test

        #test TYPE
        assert type_ref == type_test.__name__

        #test VALUE
        if isinstance(value_test, np.ndarray):
            value_ref = np.array(value_ref)
            assert np.array_equal(value_ref, value_test)
        elif isinstance(value_test, list):
            assert value_test == value_ref
        else:
            assert value_ref == value_test

