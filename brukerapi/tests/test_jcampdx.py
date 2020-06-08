from brukerapi.jcampdx import JCAMPDX
import unittest
import numpy as np
import json
from pathlib import Path

data_path = Path('/home/tomas/data/bruker2nifti_qa')
config_path = Path('test_jcampdx_qa.json')

with open(config_path) as json_file:
    reference = json.load(json_file)['test_parameter']

class TestJcampdx(unittest.TestCase):

    def test_value(self):

        for r in reference.values():
            j = JCAMPDX(data_path / r['path'])
            print("TestJCAMPDX/test_value:{}".format(r['path']))
            for key, ref in r['parameters'].items():
                print("TestJCAMPDX/test_value:{}/{}".format(r['path'],key))
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
                self.assertEqual(size_ref, size_test)

                #test TYPE
                self.assertEqual(type_ref, type_test.__name__)

                #test VALUE
                if isinstance(value_test, np.ndarray):
                    value_ref = np.array(value_ref)
                    self.assertTrue(np.array_equal(value_ref, value_test))
                elif isinstance(value_test, list):
                    self.assertListEqual(value_test, value_ref)
                else:
                    self.assertEqual(value_ref, value_test)


if __name__ == '__main__':
    unittest.main()

# context manager
# with self.assertRaises(ValueError):
#   calc.divide(10, 0)

