from bruker.jcampdx import JCAMPDX
import unittest
import numpy as np
import json

data_path = 'C:/data/' # ['C:/data/', '/home/tomas/data/']
with open('test_jcampdx.json') as json_file:
    reference = json.load(json_file)['test_parameter']

class TestJcampdx(unittest.TestCase):

    def test_value(self):
        for r in reference.values():
            j = JCAMPDX(data_path + r['path'])

            for key, ref in r['parameters'].items():
                parameter_test  = j.get_parameter(key)
                size_test= parameter_test.size
                value_test= parameter_test.value
                type_test = value_test.__class__

                value_ref = ref['value']
                size_ref = ref['size']
                type_ref = ref['type']

                #test size
                if isinstance(size_ref, list):
                    size_ref = tuple(size_ref)
                    self.assertEqual(size_ref, value_test.shape)
                elif isinstance(size_ref, int):
                    size_ref = (size_ref,)

                if size_ref == 'None':
                    size_ref = None
                # else:
                #     size_ref = tuple(size_ref)

                self.assertEqual(size_ref, size_test)

                #test type
                self.assertEqual(type_ref, type_test.__name__)

                #test value
                if isinstance(value_test, np.ndarray):
                    value_test = list(value_test.flatten())
                    self.assertEqual(value_ref, value_test)
                else:
                    self.assertEqual(value_ref, value_test)

    def test_exception(self):
        pass

    """
    def test_strip_bracket(self):
        to_test = [
            "( 16 ) <>",
            "( 4 ) Yes Yes Yes Yes",
            "( 2 ) (Shim_Z, 25, 1, 3, 0, 93.5) (Shim_Y, 25, 1, 3, 0, 93.5)",
            "(0..7) 0 0 0 0 0 0 0 0",
            "(XY..XY) 1.036651e+00, 0.000000e+00",
            "( 1, 3, 3 ) -11.53 -10.07 -16.82 -11.53 -10.07 -14.82 -11.53 -10.07 -12.82",
            "-1 0 0 0 1 0 0 0 -1",
            "32BIT_SGN_INT",
        ]

        control = [
            ["<>", 16],
            ["Yes Yes Yes Yes", 4],
            ["(Shim_Z, 25, 1, 3, 0, 93.5) (Shim_Y, 25, 1, 3, 0, 93.5)",2],
            ["0 0 0 0 0 0 0 0",range(0,7)],
            ["1.036651e+00, 0.000000e+00", "XY..XY"],
            ["-11.53 -10.07 -16.82 -11.53 -10.07 -14.82 -11.53 -10.07 -12.82", (1, 3, 3)],
            ["-1 0 0 0 1 0 0 0 -1", None],
            ["32BIT_SGN_INT",None]
        ]

        for i in range(len(to_test)):
            result = JCAMPDX.strip_size_bracket(to_test[i])
            self.assertEqual(result[0], control[i][0])
            self.assertEqual(result[1],control[i][1])

    def test_split_parallel_lists(self):
        to_test = [
            "(<1H>, 4.7) (<none>, 0) (<none>, 0)",
            "(0.982, 0.172) (4.223, 0.17) (RPS_KS, 0)",
            "(5, <FG_ISA>, <T2: y=A+C*exp(-t/T2)>, 0) (5, <FG_ISA>, <T2: y=A+C*exp(-t/T2)>, 0)",
            "(((-1 0, -0 -1.8), 19.8 13.27 20, <+R;read> <+P;phase> <+S;slice>, 0), 40, No)",
            "(1.9, 14, 90, No, 3, 2740, 0.41, 0.29, 0, 50, 7.75, <gauss512.exc>)",
            "-1 0 0 0 1 0 0 0 -1",
            "32BIT_SGN_INT",
        ]

        control = [
            ["(<1H>, 4.7)", "(<none>, 0)", "(<none>, 0)"],
            ["(0.982, 0.172)", "(4.223, 0.17)", "(RPS_KS, 0)"],
            ["(5, <FG_ISA>, <T2: y=A+C*exp(-t/T2)>, 0)", "(5, <FG_ISA>, <T2: y=A+C*exp(-t/T2)>, 0)"],
            "(((-1 0, -0 -1.8), 19.8 13.27 20, <+R;read> <+P;phase> <+S;slice>, 0), 40, No)",
            "(1.9, 14, 90, No, 3, 2740, 0.41, 0.29, 0, 50, 7.75, <gauss512.exc>)",
            "-1 0 0 0 1 0 0 0 -1",
            "32BIT_SGN_INT",
            "2200",
            "3.4234 23634.34534 345345.345"
        ]

        for i in range(len(to_test)):
            result = JCAMPDX.split_parallel_lists(to_test[i])
            self.assertEqual(result, control[i])

    def test_parse_simple_list(self):
        to_test = [
            "1.9, 14, No, 2740, <gauss512.exc>"
        ]

        control = [
            [1.9, 14, "No", 2740, "<gauss512.exc>"]
        ]

    def test_parse_element(self):
        to_test = [
            '<PVM_TrajKx> <PVM_TrajBx> <PVM_TrajKy> <PVM_TrajBy> <PVM_TrajKz> <PVM_TrajBz>',
            '<Mapshim disabled: No map acquired>',
            '26.0022222851143',
            '128 128 128'
        ]
        control = [
            ['PVM_TrajKx', 'PVM_TrajBx', 'PVM_TrajKy', 'PVM_TrajBy', 'PVM_TrajKz', 'PVM_TrajBz'],
             '<Mapshim disabled: No map acquired>',
             26.0022222851143,
             np.array([128, 128, 128])
        ]

        for i in range(len(to_test)):
            result = JCAMPDX.parse_element(to_test[i])
            self.assertEqual(result,control[i])

    """

if __name__ == '__main__':
    unittest.main()

# context manager
# with self.assertRaises(ValueError):
#   calc.divide(10, 0)

