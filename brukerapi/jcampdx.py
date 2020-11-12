import numpy as np
from .exceptions import *
from pathlib import Path
import re
import ast
from collections import OrderedDict

SUPPORTED_VERSIONS = ['4.24', '5.0', '5.00 Bruker JCAMP library', '5.00 BRUKER JCAMP library', '5.01']
GRAMMAR = {
        'COMMENT_LINE' : '\$\$[^\n]*\n',
        'PARAMETER': '##',
        'USER_DEFINED' : '\$',
        'TRAILING_EOL' : '\n$',
        'DATA_LABEL' : '\(XY..XY\)',
        'DATA_DELIMETERS':', |\n',
        'SIZE_BRACKET': '^\([^\(\)<>]*\)(?!$)',
        'LIST_DELIMETER': ', ',
        'EQUAL_SIGN': '=',
        'SINGLE_NUMBER':'-?[\d.]+(?:e[+-]?\d+)?',
        'PARALLEL_BRACKET': '\) ',
        'GEO_OBJ': '\(\(\([\s\S]*\)[\s\S]*\)[\s\S]*\)',
        'HEADER':'TITLE|JCAMPDX|JCAMP-DX|DATA TYPE|DATATYPE|ORIGIN|OWNER',
        'VERSION_TITLE':'JCAMPDX|JCAMP-DX'
    }
MAX_LINE_LEN = 78


class Parameter(object):
    """
    Data model of a single jcamp-dx parameter.

    It consists of three main parts:
    - key_str
    - size_str
    - val_str

    For instance, the following data entry:

    ##$VisuCoreSize=( 2 )
    192 192

    Is stored as:

    - key_str: ##$VisuCoreSize
    - size_str: ( 2 )
    - val_str: 192 192

    The value is parsed once it is requested. Parse methods are different for individual subclasses.
    """
    def __init__(self, key_str, size_str, val_str, version):
        """
        :param key_str: key part of the parameter e.g. ##$ACQ_ReceiverSelect
        :param size_str: size part of the parameter e.g. ( 8 )
        :param val_str: value part of the parameter Yes Yes Yes Yes No No No No
        :param version: version of the parent jcamp-dx file 4.24
        """
        self.key_str = key_str
        self.size_str = size_str
        self.val_str = val_str
        self.version = version

    def __str__(self):

        str_ = '{}'.format(self.key_str)

        if self.version == '4.24':
            str_ += '='
        else:
            str_ += '= '

        if self.size_str != '':
            str_ += '{}\n'.format(self.size_str)

        str_ += '{}'.format(self.val_str)

        return str_

    def __repr__(self):
        return self.key_str


    @property
    def key(self):
        return re.sub('##', '', re.sub('\$', '', self.key_str)).rstrip()

    @key.setter
    def key(self, key):
        #Throw error
        pass

    @property
    def user_defined(self):
        if re.search(GRAMMAR['USER_DEFINED'], self.key_str):
            return True
        else:
            return False

    @property
    def tuple(self):
        value = self.value
        if isinstance(value, int) or isinstance(value, float):
            return (value,)
        else:
            return tuple(value)

    @property
    def list(self):
        value = self.value
        if isinstance(value, list):
            return value
        elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return [value]
        else:
            return list(value)

    @property
    def nested(self):
        """Return value as nested list

        For example the following entry is nested by default:
        ##$VisuFGOrderDesc=( 2 )
        (6, <FG_ECHO>, <>, 0, 1) (5, <FG_SLICE>, <>, 1, 2)

        But this one not:
        ##$VisuFGOrderDesc=( 2 )
        (5, <FG_SLICE>, <>, 1, 2)

        This proprerty allows treat the parameter as nested list in all cases.

        """
        value = self.list
        if isinstance(value[0], list):
            return value
        else:
            return [value]


    @property
    def array(self):
        return np.atleast_1d(self.value)

    @property
    def shape(self):
        value = self.value
        if isinstance(value, np.ndarray):
            return value.shape
        else:
            raise AttributeError



    @classmethod
    def pack_key(cls, value, usr_defined):
        assert isinstance(value, str)

        val_str = value

        if usr_defined:
            val_str = '$' + val_str

        return '##' + val_str


class GenericParameter(Parameter):
    def __init__(self, version, key, size_bracket, value):
        super(GenericParameter, self).__init__(version, key, size_bracket, value)

    @classmethod
    def from_values(cls, version, key, size, value, user_defined):

        key_str = key
        size_str = size
        value_str = value

        super(GenericParameter, cls).__init__(version, key_str, size_str, value_str)

    @property
    def value(self, **kwargs):

        val_str = re.sub('\n', '', self.val_str)

        # unwrap wrapped list
        if re.match('@[0-9]*\*',val_str) is not None:
            val_str = self._unwrap_list(val_str)

        val_str_list = GenericParameter.split_parallel_lists(val_str)

        if isinstance(val_str_list, str):
            value = GenericParameter.parse_value(val_str_list)
        elif isinstance(val_str_list, list):
            value = []
            for val_str in val_str_list:
                value.append(GenericParameter.parse_value(val_str))

        if isinstance(value, np.ndarray) and self.size:
            if not 'str' in value.dtype.name:
                return np.reshape(value, self.size, order='C')
            else:
                return value
        else:
            return value

    @value.setter
    def value(self, value):
        size = self.size

        if isinstance(value, float):
            val_str = self.serialize_float(value, self.version)
        elif isinstance(value, int):
            val_str = str(value)
        elif isinstance(value, list):
            if isinstance(value[0], list):
                val_str = self.serialize_nested_list(value)
                size = (len(value),)
            else:
                val_str = self.serialize_list(value)
        elif isinstance(value, np.ndarray):
            val_str = self.serialize_ndarray(value)
        else:
            val_str = value

        self.size = size
        self.val_str= val_str

    def primed_dict(self, index):
        nested_list = self.nested
        primed_dict = OrderedDict()

        for list in nested_list:
            primed_dict[list[index]] = list

        return primed_dict

    def sub_list(self, index):
        nested_list = self.nested
        sub_list = []
        for list in nested_list:
            sub_list.append(list[index])

        return sub_list




    @property
    def size(self):
        size_str = self.size_str[1:-2]

        if size_str == '':
            return None

        #"(3,3)\n" -> 3,3
        if ".." in size_str:
            try:
                size_str = np.array(size_str.split(".."), dtype='int32')
                size = range(size_str[0], size_str[1])
            except ValueError:
                # size bracket is returned as string
                # catches (XY..XY) etc.
                pass

        elif "," in size_str:
            size_str = size_str.split(",")
            size = tuple(np.array(size_str, dtype='int32'))
        else:
            size = (np.int(size_str),)

        return size

    @size.setter
    def size(self, size):
        if size is None:
            self.size_str = ''
            return

        if isinstance(size, tuple):
            # (1,3,3) -> "( 1,3,3 )"
            if len(size) > 1:
                size_str = '( {} )'.format(str(size)[1:-1])
            #(1,) -> "( 1 )"
            else:
                size_str = '( {} )'.format(str(size)[1:-2])
        elif isinstance(size, range):
            size_str = '({size.start}..{size.stop})'.format(size.start, size.stop)
        elif isinstance(size, int):
            size_str = '( {} )'.format(str(size))
        else:
            size_str = '({})'.format(size)

        self.size_str = size_str

    @classmethod
    def parse_value(cls, val_str, size_bracket=None):
        # remove \n
        val_str = re.sub('\n','', val_str)

        # sharp string
        if val_str.startswith('<') and val_str.endswith('>'):

            val_strs = re.findall('<[^<>]*>', val_str)

            if len(val_strs) == 1:
                return val_strs[0]
            else:
                return np.array(val_strs)


        # int/float
        if len(re.findall(GRAMMAR['SINGLE_NUMBER'],val_str))==1:
            try:
                value = ast.literal_eval(val_str)

                # if value is int, or float, return, tuple will be parsed as list later on
                if isinstance(value, float) or isinstance(value, int):
                    return value
            except (ValueError, SyntaxError):
                pass

        # list
        if val_str.startswith('(') and val_str.endswith(''):
            val_strs = re.split(GRAMMAR['LIST_DELIMETER'], val_str[1:-1])
            value = []

            for val_str in val_strs:
                value.append(cls.parse_value(val_str))

            return value

        val_strs = re.split(' ', val_str)

        if len(val_strs) > 1:
            # try casting into int, or float array, if both of casts fail, it should be string array
            try:
                return np.array(val_strs).astype('int')
            except ValueError:
                pass

            try:
                return np.array(val_strs).astype('float')
            except ValueError:
                pass

            return np.array(val_strs)
        else:
            return val_strs[0]

    @classmethod
    def serialize_value(cls, value):

        if isinstance(value, float):
            val_str = cls.serialize_float(value)
        elif isinstance(value, int):
            val_str = str(value)
        elif isinstance(value, list):
            val_str = cls.serialize_float(value)
        elif isinstance(value, np.ndarray):
            val_str = cls.serialize_list(value)
        else:
            val_str = value
        return val_str

    @classmethod
    def serialize_float(cls, value, version):
        if version == 4.24:
            return "{:.6e}".format(value)
        else:
            return str(value)

    @classmethod
    def serialize_list(cls, value):

        if isinstance(value[0], list):

            val_str = ''

            for value_ in value:
                val_str += cls.serialize_list(value_)
                val_str += ' '

            return val_str


        val_str = '('

        for item in value:
            val_str += cls.serialize_value(item)
            val_str += ', '

        return val_str[:-2] + ')'

    @classmethod
    def serialize_nested_list(cls, values):
        val_str = ''

        for value in values:
            val_str += GenericParameter.serialize_list(value)
            val_str += ' '

        return val_str[0:-1]

    @classmethod
    def serialize_ndarray(cls, value):
        val_str = ''

        for value_ in value:
            val_str_ = str(value_)
            val_str += val_str_
            val_str += ' '

        return val_str[:-1]

    @classmethod
    def split_parallel_lists(cls, val_str):
        lst = re.split(GRAMMAR['PARALLEL_BRACKET'], val_str)

        if len(lst) == 1:
            return lst[0]

        def restore_right_bra(string):
            if string.endswith(')'):
                return string
            else:
                return string + ')'

        for i in range(len(lst)):
            lst[i] = restore_right_bra(lst[i])

        return lst

    def _unwrap_list(self, val_str):

        while re.search('@[0-9]*\*\(\d*\.?\d*\)', val_str):
            match = re.search('@[0-9]*\*\(\d*\.?\d*\)', val_str)
            left = val_str[0:match.start()]
            right = val_str[match.end():]
            sub = val_str[match.start():match.end()]
            size, value = re.split('\*', sub)
            size = int(size[1:])
            middle = ''
            for i in range(size):
                middle += '{} '.format(value[1:-1])
            val_str = left + middle[0:-1] + right

        return val_str


class HeaderParameter(Parameter):
    def __init__(self, key_str, size_str, val_str, version):
        super(HeaderParameter, self).__init__(key_str, size_str, val_str, version)

    @property
    def value(self):
        return self.val_str

    @value.setter
    def value(self, val_str):
        self.val_str = val_str

    @property
    def size(self):
        return None


class GeometryParameter(Parameter):
    def __init__(self, key_str, size_str, val_str, version):
        super(GeometryParameter, self).__init__(key_str, size_str, val_str, version)

    @property
    def value(self):
        return None

    @value.setter
    def value(self):
        pass

    @property
    def affine(self):
        """

        :return: 4x4 3D Affine Transformation Matrix
        """
        # TODO support for multiple slice packages
        match = re.match('\(\(\(.*\)', self.val_str)
        affine_str = self.val_str[match.start() + 3: match.end() - 1]
        orient, shift = affine_str.split(', ')

        orient = GenericParameter.parse_value(orient)
        shift = GenericParameter.parse_value(shift)
        affine = np.zeros(shape=(4,4))
        affine[0:3, 0:3] = np.reshape(orient, (3,3))
        affine[0:3, 3] = shift

        return affine


class DataParameter(Parameter):
    def __init__(self, version, key, size_bracket, value):
        super(DataParameter, self).__init__(version, key, size_bracket, value)

    @property
    def value(self):
        val_list = re.split(GRAMMAR['DATA_DELIMETERS'], self.val_str)
        data = [GenericParameter.parse_value(x) for x in val_list]
        return np.reshape(data, (2, -1))

    @value.setter
    def value(self, value):
        val_str = ""

        for i in range(len(value)):
            val_str += "{:.6e}".format(value[i])
            if np.mod(i, 2) == 0:
                val_str += ', '
            else:
                val_str += '\n'

        self.value = val_str

    @property
    def size(self):
        return self.size_str[1:-1]

    @size.setter
    def size(self, value):
        self.size_str = '({})'.format(value)


class JCAMPDX(object):
    """Representation of a single jcamp-dx file.

    It's main component is a dictionary of parameters.

    **Example:**

    .. highlight:: python
    .. code-block:: python

        from bruker.jcampdx import JCAMPDX

        visu_pars = JCAMPDX('path/visu_pars')
        size = visu_pars.get_value('VisuCoreSize')

    """

    def __init__(self, path, load=True, **kwargs):
        """JCAMPDX constructor

        JCAMPDX object is constructed by passing a path to a valid jcamp-dx file. It is possible to construct an
        empty object.
        """

        # If path is directory
        self.path = Path(path)

        if self.path.is_dir():
            raise InvalidJcampdxFile(path)

        self.params = {}

        self.unload()

        if load:
            self.load()
        else:
            JCAMPDX.verify_version(self.version)

    @property
    def type(self):
        return self.path.name

    def __str__(self, file=None):


        if self.params == {}:
            return self.type

        jcampdx_serial = ''

        for param in self.params.values():

            param_str = str(param)

            if len(param_str) > 78:
                param_str = JCAMPDX.wrap_lines(param_str)

            jcampdx_serial += '{}\n'.format(param_str)

        return jcampdx_serial[0:-1] + "\n##END= "

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    def __add__(self, other):
        self.params.update(other.params)
        return self

    def __getitem__(self, key):
        return self.params[key]

    def __delitem__(self, key):
        del self.params[key]

    def load(self):
        self.load_parameters()

    def load_parameters(self):
        self.params = JCAMPDX.read_jcampdx(self.path)

    def unload(self):
        self.params = {}

    @property
    def version(self):
        try:
            return self.params['JCAMPDX'].value
        except KeyError:
            pass

        try:
            self.params['JCAMP-DX'].value
        except KeyError:
            pass

        try:
            _, version = JCAMPDX.load_parameter(self.path, 'JCAMPDX')
            return version.value
        except (InvalidJcampdxFile, ParameterNotFound):
            pass

        try:
            _, version = JCAMPDX.load_parameter(self.path, 'JCAMP-DX')
            return version.value
        except (InvalidJcampdxFile, ParameterNotFound):
            pass

        raise InvalidJcampdxFile(self.path)

    @version.setter
    def version(self, value):
        self.version = value

    def keys(self):
        return self.params.keys()

    """
    PUBLIC INTERFACE
    """
    def get_parameters(self):
        return self.params

    def get_parameter(self, key):
        return self.params[key]

    def set_parameter(self, key, value):
        self.params[key] = value

    def get_value(self, key):
        return self.params[key].value

    def get_list(self, key):
        """Idea is to ensure, that a parameter will be a list even if parameter only contains one entry
        """
        value = self.get_value(key)
        if isinstance(value, list):
            return value
        elif isinstance(value, np.ndarray):
            return list(value)
        else:
            return [value, ]

    def get_nested_list(self, key):
        value = self.get_value(key)
        if not isinstance(value, list):
            value =[value,]

        if not isinstance(value[0], list):
            value = [value, ]

        return value

    def set_nested_list(self,key, value):
        self.params[key].value = value

    def get_int(self, key):
        return int(self.get_value(key))

    def set_int(self,key, value):
        self.params[key].value = value

    def get_float(self, key):
        return float(self.get_value(key))

    def get_tuple(self, key):
        value = self.get_value(key)

        if isinstance(value, int) or isinstance(value, float):
            return (value,)
        else:
            return tuple(value)

    def get_array(self, key, dtype=None, shape=(-1,), order='C'):
        parameter=self.get_parameter(key)
        value = parameter.value
        size = parameter.size

        value = np.atleast_1d(value)

        if dtype is not None:
            value = value.astype(dtype)

        # user did not specify shape, try to get the size_bracket
        if shape == (-1,) and isinstance(size, tuple):
            shape = size

        return np.reshape(value, shape, order=order)

    def set_array(self, key, value, file=None , order='C'):

        parameter = self.get_parameter(key, file)

        value = np.reshape(value,(-1,), order=order)
        self.__setattr__(key, value.tolist())

    def get_str(self, key, strip_sharp=True):
        value = str(self.get_value(key))

        if strip_sharp and value.startswith('<') and value.endswith('>'):
            value = value[1:-1]

        return value

    @classmethod
    def verify_version(cls, version):
        if version not in SUPPORTED_VERSIONS:
            raise JcampdxVersionError(version)

    @classmethod
    def load_parameter(cls, path, key):
        with open(path) as f:
            try:
                content = f.read()
            except:
                raise InvalidJcampdxFile(path)

        match = re.search('##{}[^\#\$]+|##\${}[^\#\$]+'.format(key,key), content)

        if match == None:
            raise ParameterNotFound(key, path)

        line = content[match.start():match.end()-1] # strip trailing EOL
        key, parameter = JCAMPDX.handle_jcampdx_line(line, None)

        return key, parameter

    @classmethod
    def read_jcampdx(cls, path):

        path = Path(path)

        params = {}

        with path.open() as f:
            try:
                content = f.read()
            except:
                raise JcampdxFileError('file {} is not a text file'.format(path))

        # remove all comments
        content = re.sub(GRAMMAR['COMMENT_LINE'], '', content)

        # split into individual entries
        content = re.split(GRAMMAR['PARAMETER'], content)[1:-1]

        # strip trailing EOL
        content = [re.sub(GRAMMAR['TRAILING_EOL'],'',x) for x in content]

        # ASSUMPTION the jcampdx version string is in the second row
        try:
            version_line = content[1]
            if re.search(GRAMMAR['VERSION_TITLE'], version_line) is None:
                raise JcampdxFileError('file {} is not a JCAMP-DX file'.format(path))
        except:
            raise JcampdxFileError('file {} is not a text file'.format(path))

        _, _, version = JCAMPDX.divide_jcampdx_line(version_line)

        if version not in SUPPORTED_VERSIONS:
            raise JcampdxVersionError(version)

        for line in content:
            # Restore the ##
            key, parameter = JCAMPDX.handle_jcampdx_line('##{}'.format(line), version)
            params[key] = parameter
        return params

    @classmethod
    def handle_jcampdx_line(cls, line, version):
        key_str, size_str, val_str = cls.divide_jcampdx_line(line)
        if re.search(GRAMMAR['GEO_OBJ'], line) is not None:
            parameter = GeometryParameter(key_str, size_str, val_str, version)
        elif re.search(GRAMMAR['DATA_LABEL'], line):
            parameter = DataParameter(key_str, size_str, val_str, version)
        elif re.search(GRAMMAR['HEADER'],key_str):
            parameter = HeaderParameter(key_str, size_str, val_str, version)
        else:
            parameter = GenericParameter(key_str, size_str, val_str, version)

        return parameter.key, parameter

    @classmethod
    def divide_jcampdx_line(cls, line):
        key_str, val_str = cls.split_key_value_pair(line)
        val_str, siz_str = cls.strip_size_bracket(val_str)
        return key_str, siz_str, val_str

    @classmethod
    def split_key_value_pair(cls, line):
        # ASSUMPTION the first occurence of = in jcampdx line divides key and value pair
        # example:
        match = re.search(GRAMMAR['EQUAL_SIGN'], line)
        key = line[0:match.start()]
        val_str = line[match.end():].lstrip()
        return key, val_str

    @classmethod
    def strip_size_bracket(cls, val_str):
        """Strip size bracket if it's found in value string.

        Examples of strings with size bracket:
        '( 2 ) Spatial Spatial'
        '( 1, 20 ) <>'

        :param val_str: value string
        :return value: value string without bracket in case, size bracket is found, otherwise returns unmodified val_str
        :return size: size bracket str
        """
        match = re.search(GRAMMAR['SIZE_BRACKET'], val_str)

        if match is None:
            return val_str, ''
        else:
            size_bracket = val_str[match.start():match.end()]
            val_str = val_str[match.end():].lstrip()

        return val_str, size_bracket

    @classmethod
    def wrap_lines(cls, line):
        line_wraps = re.split('\n', line)
        tail = line_wraps[-1]

        tail_bits = re.split('\s', tail)

        lines = 1
        tail = ''

        for tail_bit in tail_bits:
            if len(tail + tail_bit) > lines * MAX_LINE_LEN:
                tail += '\n'
                lines += 1
            tail += tail_bit
            tail += ' '

        line_wraps[-1] = tail[:-1]

        return '\n'.join(line_wraps)

    def write(self, path):
        """
        Write JCAMP-DX object into file
        :param path:
        :return:
        """
        with Path(path).open('w') as f:
            f.write(str(self))
