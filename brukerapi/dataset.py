from .exceptions import *
from .schemas import *
from .data import *

from pathlib import Path
import json
import numpy as np
import os
import os.path
import yaml
import datetime

# Dict of supported data sets. In order to read type of dataset specified by key value, all files listed in value
# need to be present in the same directory.
SUPPORTED = {
    'fid': ['acqp', 'method'],
    'traj': ['acqp','method'],
    '2dseq': ['visu_pars'],
    'ser': ['acqp', 'method'],
    'rawdata': ['acqp','method'],
    '1r': [],
    '1i': []
}


class Dataset:
    """
    Base class of the API. It contains information obtained from a pair of Bruker binary data file (fid, 2dseq,
    ser, rawdata, 1r, 1i) and one, or more :py:class:`bruker.jcampdx.JCAMPDX` files (acqp, method, visu_pars).

    **Main components:**

    - **parameters**: :py:class:`dict`

    Meta data essential for construction of schema and manipulation with the binary data file.

    - **schema**: one of :py:class:`~bruker.schemas.SchemeFid`:py:class:`~bruker.schemas.Scheme2dseq`:py:class:`~bruker.schemas.SchemeSer`:py:class:`~bruker.schemas.SchemeRawdata`

    An object encapsulating all functionality dependent on metadata. It provides method to reshape data.

    - **data**: :py:class:`numpy.ndarray`

    Array containing the data read from any of the supported binary files.

    **Example:**

    .. highlight:: python
    .. code-block:: python

        from bruker.dataset import Dataset

        dataset = Dataset('path/2dseq')

    """

    def __init__(self, path, load=True, random_access=False, **kwargs):
        """Constructor of Dataset

        Dataset can be constructed either by passing a path to one of the SUPPORTED binary files, or to a directory
        containing it. It is possible, to create an empty object using the load switch.

        :param path: **str** path to dataset
        :param load: **bool** when false, empty Dataset is created
        """
        self.path = Path(path)
        self.random_access = random_access
        if not self.path.exists() and load:
            raise FileNotFoundError(self.path)

        # directory constructor
        if self.path.is_dir():
            content = os.listdir(self.path)
            if 'fid' in content:
                self.path = self.path / 'fid'
            elif '2dseq' in content:
                self.path = self.path / '2dseq'
            else:
                raise NotADatasetDir(self.path)

        # validate path
        print(self.path)
        print(str(self.path.name))
        print(str(self.type))
        # self.validate(load)

        # save kwargs
        self._kwargs = kwargs

        # create an empty data set
        self.unload()

        # load data
        if load:
            self.load()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    def __str__(self):
        """String representation"""
        return str(self.path)

    def __getitem__(self, item):
        return self.get_item(item)

    def __getattr__(self, item):
        return self.get_value(item)

    def __call__(self, **kwargs):
        self._kwargs.update(kwargs)
        return self

    def validate(self, load):
        """Validate Dataset

        Check whether the dataset type is supported and complete. Dataset is allowed to be incomplete when load is
        set to `False`.

        :param load: **bool** binary switch which determines, whether dataset will load data, or remain empty
        :raise: :UnsuportedDatasetType: In case `Dataset.type` is not in SUPPORTED
        :raise: :IncompleteDataset: If some jcamp-dx file, necessary to create a Dataset is missing.
        """

        # Check whether dataset file is supported
        if self.type not in SUPPORTED:
            raise UnsuportedDatasetType(self.type)

        # Check whether all necessary JCAMP-DX files are present
        if load:
            if not (set(SUPPORTED[self.type]) <= set(os.listdir(str(self.path.parent)))):
                raise IncompleteDataset

    @property
    def type(self):
        """Type of dataset (fid, 2dseq,...)

        :return: **str** type of dataset
        """
        if 'rawdata' in self.path.name:
            return 'rawdata'
        else:
            return self.path.name

    @property
    def subtype(self):
        """Subtype of rawdata (job0, job1, Navigator)

        :return: **str** subtype
        """
        if 'rawdata' in self.path.name:
            return self.path.name.split('.')[-1]
        else:
            return None

    """
    LOADERS/UNLOADERS
    """
    def load(self):
        """
        Load parameters, schema and data. In case, there is a traj file related to a fid file, traj is loaded as well.
        """
        self.load_parameters()
        self.load_properties()
        self.load_schema()
        self.load_data()
        # self.load_traj()

    def load_parameters(self, parameters=None):
        """
        Load all parameters essential for reading of given dataset type. For instance, type `fid` data set loads acqp and method file, from parent directory in which the fid file is contained.
        """
        if parameters:
            self._parameters = parameters
        else:
            self._parameters = self._read_parameters()

    def add_parameters(self, file):
        """
        Load additional jcamp-dx file and add it to Dataset parameter space. It is later available via getters,
        or using the dot notation.
        :param file: file
        :return:
        """
        try:
            parameters = JCAMPDX(self.path.parent / file)
        except FileNotFoundError:
            if self.type in ['2dseq','1r','1i']:
                try:
                    parameters = JCAMPDX(self.path.parents[2] / file)
                except FileNotFoundError:
                    raise FileNotFoundError(file)
            if self.type in ['fid','ser','rawdata','traj']:
                try:
                    parameters = JCAMPDX(self.path.parent / Path('pdata/1') / file)
                except FileNotFoundError:
                    raise FileNotFoundError(file)

        self.parameters[file] = parameters

    def load_properties(self):
        self.proc_property_file('{}/schema_{}_core.json'.format(str(config_paths['core']),self.type))
        if self._kwargs.get('schema_custom_config'):
            self.proc_property_file('{}/{}_custom.json'.format(self._kwargs.get('custom_config'), self.type))
        else:
            self.proc_property_file(
                '{}/schema_{}_custom.json'.format(str(config_paths['core']), self.type))

    def proc_property_file(self, path):
        with open(path) as f:
            for property in json.load(f).items():
                self.add_property(property)

    def add_property(self, property):
        """ Add property to the dataset and schema

        * Evaluate the condition for a given command if these are fulfilled, the next step follows, otherwise,
        the next command is processed.
        * Get the value of the property using command and parameters of the dataset.
        * Set the property of a dataset
        * Set the property of a schema

        :param property: tuple containing the name of the property and a list of possible commands to create it
        """
        for desc in property[1]:
            try:
                self.eval_conditions(desc['conditions'])
                value = self.make_element(desc['cmd'])
                self.__setattr__(property[0], value)
                break
            except (PropertyConditionNotMet, AttributeError):
                pass

    def make_element(self, cmd):
        if isinstance(cmd, str):
            return eval(self.sub_parameters(cmd))
        elif isinstance(cmd, int) or isinstance(cmd, float):
            return cmd
        elif isinstance(cmd, list):
            element = []
            for cmd_ in cmd:
                element.append(self.make_element(cmd_))
            return element

    def eval_conditions(self, conditions):
        """

        Condition can be in the following forms:

        - a string
        - a list with two elements

        :param conditions:
        :return:
        """
        for condition in conditions:
            try:
                if isinstance(condition, str):
                    if not self.make_element(condition):
                        raise PropertyConditionNotMet
                elif isinstance(condition, list):
                    if not self.make_element(condition[0]) in condition[1]:
                        raise PropertyConditionNotMet
            except KeyError:
                raise PropertyConditionNotMet

    def sub_parameters(self, recipe):
        # entries with property e.g. VisuFGOrderDesc.nested to self._dataset['VisuFGOrderDesc'].nested
        for match in re.finditer('#[a-zA-Z0-9_]+\.[a-zA-Z]+', recipe):
            m = re.match('#[a-zA-Z0-9_]+', match.group())
            recipe = recipe.replace(m.group(),"self['{}']".format(m.group()[1:]))
        # entries without property e.g. VisuFGOrderDesc to self._dataset['VisuFGOrderDesc'].value
        for match in re.finditer('@[a-zA-Z0-9_]+', recipe):
            recipe = recipe.replace(match.group(),"self.{}".format(match.group()[1:]))
        for match in re.finditer('#[a-zA-Z0-9_]+', recipe):
            recipe = recipe.replace(match.group(),"self['{}'].value".format(match.group()[1:]))
        return recipe

    def load_schema(self):
        """
        Load the schema for given data set.
        """
        if self.type == 'fid':
            self._schema = SchemaFid(self)
        elif self.type == '2dseq':
            self._schema = Schema2dseq(self)
        elif self.type == 'rawdata':
            self._schema = SchemaRawdata(self)
        elif self.type == 'ser':
            self._schema = SchemaSer(self)
        elif self.type == 'traj':
            self._schema = SchemaTraj(self)

    def load_data(self):
        """
        Load the data binary file.
        """
        if self.random_access:
            self._data = DataRandomAccess(self)
        else:
            self._data = self._read_data()

    def load_traj(self, **kwargs):
        if Path(self.path.parent / 'traj').exists() and self.type != 'traj':
            self._traj = Dataset(self.path.parent / 'traj', load=False, random_access=self.random_access)
            self._traj._parameters = self.parameters
            self._traj._schema = SchemaTraj(self._traj, meta=self.schema._meta, sub_params=self.schema._sub_params,
                                           fid=self)
            self._traj.load_data()
        else:
            self._traj = None

    def unload(self):
        self.unload_parameters()
        self.unload_schema()
        self.unload_data()
        self.unload_traj()

    def unload_parameters(self):
         self._parameters = None

    def unload_schema(self):
        self._schema = None

    def unload_data(self):
        self._data = None

    def unload_traj(self):
        self._traj = None

    """
    READERS/WRITERS
    """
    def _read_parameters(self, **kwargs):
        """Read parameters form required JCAMPDX files in the directory

        Parameters
        ----------
        paths - dictionary containing paths to all files and folders in Bruker directory

        Returns
        -------
        JCAMPDX object containing parameters from all JCAMP-DX files in root folder
        """
        if kwargs.get('parameter_scope') is None:
            parameter_scope = SUPPORTED[self.type]
        else:
            parameter_scope = kwargs.get('parameter_scope')

        parameters = {}

        for file_type in parameter_scope:
            parameters[file_type] = JCAMPDX(self.path.parent / file_type)

        return parameters

    def _read_data(self):
        data = self._read_binary_file(self.path, self.numpy_dtype, self.shape_storage)
        return self._schema.deserialize(data, self._schema.layouts)

    def _read_binary_file(self, path, dtype, shape):
        """Read Bruker binary file

        Parameters
        ----------
        path
        dtype - numpy dtype obtained using GO_raw_data_format and BYTORDA, or VisuCoreWordType and VisuCoreByteOrder parameters

        Returns
        -------
        1D ndarray containing the full data vector
        """
        # TODO debug with this
        # try:
        #     assert os.stat(str(path)).st_size == np.prod(shape) * dtype.itemsize
        # except AssertionError:
        #     raise ValueError('Dimension missmatch')

        return np.array(np.memmap(path, dtype=dtype, shape=shape, order='F')[:])

    def write(self, path=None, **kwargs):
        """
        Write a Dataset into a file given by file.

        :param path: Path to one of the supported Datasets
        :param kwargs:
        :return:
        """

        if path:
            path = Path(path)
        else:
            path = self.path

        parent = path.parent

        if not parent.exists():
            os.mkdir(parent)

        if self.type not in SUPPORTED:
            raise UnsuportedDatasetType

        # try:
        #     os.mkdir(parent)
        # except IOError as e:
        #     if not kwargs.get('force'):
        #         raise e

        self._write_parameters(parent)
        self._write_data(path)

    def _write_parameters(self, parent):
        for type_, jcampdx in self.parameters.items():
            jcampdx.write(parent / type_)

    def _write_data(self, path):
        data = self.data.copy()
        data = self._schema.serialize(data, self._schema.layouts)
        self._write_binary_file(path, data, self.shape_storage, self.numpy_dtype)

    def _write_binary_file(self, path, data, storage_layout, dtype):
        fp = np.memmap(path, mode='w+', dtype=dtype, shape=storage_layout, order='F')
        fp[:] = data

    """
    REPORTING INTERFACE
    """
    def report(self, path, abs_path=None, names=None):
        """
        Reporting function, allows to save data set properties to a json, or yaml file
        :param path:
        :param abs_path:
        :param names:
        :return:
        """
        path = Path(path)
        if path.suffix == '.json':
            self.to_json(path, abs_path=abs_path, names=names)
        elif path.suffix == '.yml':
            self.to_yaml(path, abs_path=abs_path, names=names)

    def to_json(self, path=None, abs_path=None, names=None):
        if path:
            with open(path, 'w') as json_file:
                    json.dump(self.to_dict(abs_path=abs_path, names=names), json_file, indent=4)
        else:
            return json.dumps(self.to_dict(abs_path=abs_path, names=names), indent=4)

    def to_yaml(self, path=None, abs_path=None, names=None):
        if path:
            with open(path, 'w') as yaml_file:
                    yaml.dump(self.to_dict(abs_path=abs_path, names=names), yaml_file, default_flow_style=False)
        else:
            return yaml.dump(self.to_dict(abs_path=abs_path, names=names), default_flow_style=False)

    def to_dict(self, abs_path=None, names=None):
        """

        :param abs_path:
        :param names:
        :return:
        """
        path = self.path.relative_to(abs_path)

        if not names:
            names = list(vars(self).keys())

        # list of Dataset properties to be excluded from the export
        reserved = ['_parameters', 'path', '_data', '_traj', '_kwargs', '_schema','random_access']
        names = list(set(names) - set(reserved))

        properties = {}

        for var in names:
            properties[var] = self.encode_property(self.__getattribute__(var))

        return {"path": path.as_posix(), "properties": properties}

    def encode_property(self, var):
        """
        Encoder subclassing was not used due to complicated debugging
        :param var:
        :return:
        """
        if isinstance(var, Path):
            return str(var)
        elif isinstance(var, np.integer) or isinstance(var, np.int32):
            return int(var)
        elif isinstance(var, np.floating):
            return float(var)
        elif isinstance(var, np.ndarray):
            return var.tolist()
        elif isinstance(var, np.dtype):
            return var.name
        elif isinstance(var, list):
            return [self.encode_property(var_) for var_ in var]
        elif isinstance(var, tuple):
            return self.encode_property(list(var))
        elif isinstance(var, datetime.datetime):
            return str(datetime.datetime)
        else:
            return var

    """
    GETTERS
    """
    def get_parameter(self,key):
        for param_file in self.parameters.values():
            try:
                return param_file.get_parameter(key)
            except:
                pass
        raise KeyError

    def get_value(self, key):
        for param_file in self.parameters.values():
            try:
                return param_file.get_value(key)
            except:
                pass
        raise AttributeError('Dataset object has no attribute {}'.format(key))

    def get_item(self, key):
        for param_file in self.parameters.values():
            try:
                return param_file[key]
            except:
                pass
        raise AttributeError('Dataset object has no attribute {}'.format(key))

    def get_str(self, key, strip_sharp=True):
        for param_file in self.parameters.values():
            try:
                return param_file.get_str(key, strip_sharp)
            except:
                pass
        raise KeyError

    def get_int(self, key):
        for param_file in self.parameters.values():
            try:
                return param_file.get_int(key)
            except:
                pass
        raise KeyError

    def get_float(self, key):
        for param_file in self.parameters.values():
            try:
                return param_file.get_float(key)
            except:
                pass
        raise KeyError

    def get_tuple(self, key):
        for param_file in self.parameters.values():
            try:
                return param_file.get_tuple(key)
            except:
                pass
        raise KeyError

    def get_array(self, key, dtype=None, shape=None, order='C'):
        for param_file in self.parameters.values():
            try:
                return param_file.get_array(key, dtype=dtype, shape=shape, order=order)
            except:
                pass
        raise KeyError

    def get_list(self, key):
        for param_file in self.parameters.values():
            try:
                return param_file.get_list(key)
            except:
                pass
        raise KeyError

    def get_nested_list(self, key):
        for param_file in self.parameters.values():
            try:
                return param_file.get_nested_list(key)
            except:
                pass
        raise KeyError

    """
    PROPERTIES
    """
    @property
    def data(self):
        """Data array.

        :type: numpyp.ndarray
        """
        if self._data is not None:
            return self._data
        else:
            raise DataNotLoaded

    @data.setter
    def data(self,value):
        self._data = value

    @property
    def traj(self):
        """Trajectory array loaded from a `traj` file

        :type: numpy.ndarray
        """
        if self._traj is not None:
            return self._traj.data
        else:
            raise TrajNotLoaded

    @property
    def parameters(self):
        if self._parameters is not None:
            return self._parameters
        else:
            raise ParametersNotLoaded

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def schema(self):
        if self._schema is not None:
            return self._schema
        else:
            raise SchemeNotLoaded

    @property
    def dim(self):
        """number of dimensions of the data array

        :type: int
        """
        return self.data.ndim

    @property
    def shape(self):
        """shape of data array

        :type: tuple
        """
        return self.data.shape

