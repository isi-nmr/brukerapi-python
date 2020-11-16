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

# Dict of supported data set types with a list of JCAMP-DX files essential for the creation of given data set type.
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
    Data set is created using one binary file {fid, 2dseq, rawdata, ser, 1r, 1i} and several JCAMP-DX
    files (method, acqp, visu_pars,...). The JCAMP-DX files necessary for a creation of a data set are denoted as
    **essential**. Each of the binary data files (fid, 2dseq,...) has slightly different data layout, i.e. the . The
    data in the binary files is stored Since the individual types of b some features We distinguish By he name of the
    binary file  we determine the **type** of data set.

    **Main components of a data set:**

    - **parameters**:

    Meta data essential for construction of schema and manipulation with the binary data file.

    - **properties**

    Derived from parameters.

    - **schema**:
        - :py:class:`~brukerapi.schemas.SchemaFid`
        - :py:class:`~brukerapi.schemas.Schema2dseq`
        - :py:class:`~brukerapi.schemas.SchemaSer`
        - :py:class:`~brukerapi.schemas.SchemaRawdata`

    An object encapsulating all functionality dependent on metadata. It provides method to reshape data.

    - **data**:
        - :py:class:`numpy.ndarray`

    Array containing the data read from any of the supported binary files.

    **Example:**

    .. highlight:: python
    .. code-block:: python

        from bruker.dataset import Dataset

        dataset = Dataset('path/2dseq')

    """

    def __init__(self, path, **kwargs):
        """Constructor of Dataset

        Dataset can be constructed either by passing a path to one of the SUPPORTED binary files, or to a directory
        containing it. It is possible, to create an empty object using the load switch.

        :param path: **str** path to dataset
        :raise: :UnsuportedDatasetType: In case `Dataset.type` is not in SUPPORTED
        :raise: :IncompleteDataset: If any of the JCAMP-DX files, necessary to create a Dataset instance is missing

        """
        self.path = Path(path)

        if not 'load' in kwargs:
            kwargs['load'] = True
        self._kwargs = kwargs

        if not self.path.exists() and kwargs['load']:
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
        self._validate()

        # load data if the load kwarg is true
        if self._kwargs['load']:
            self.load()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    def __str__(self):
        """
        String representation is a path to the data set.
        """
        return str(self.path)

    def __getitem__(self, item):
        for parameter_file in self._parameters.values():
            try:
                return parameter_file[item]
            except KeyError:
                pass

        raise KeyError(item)

    def __call__(self, **kwargs):
        self._kwargs.update(kwargs)
        return self

    def _validate(self):
        """Validate Dataset

        Check whether the dataset type is supported and complete. Dataset is allowed to be incomplete when load is
        set to `False`.

        :raise: :UnsuportedDatasetType: In case `Dataset.type` is not in SUPPORTED
        :raise: :IncompleteDataset: If any of the JCAMP-DX files, necessary to create a Dataset instance is missing
        """

        # Check whether dataset file is supported
        if self.type not in SUPPORTED:
            raise UnsuportedDatasetType(self.type)

        # Check whether all necessary JCAMP-DX files are present
        if self._kwargs['load']:
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

    def load(self):
        """
        Load parameters, properties, schema and data. In case, there is a traj file related to a fid file,
        traj is loaded as well.
        """
        self.load_parameters()
        self.load_properties()
        self.load_schema()
        self.load_data()
        # self.load_traj()

    def unload(self):
        """
        Unload parameters, properties, schema and data. In case, there is a traj file related to a fid file,
        traj is unloaded as well.
        """
        self.unload_parameters()
        self.unload_properties()
        self.unload_schema()
        self.unload_data()
        self.unload_traj()

    """
    PARAMETERS
    """

    def load_parameters(self):
        """
        Load all parameters essential for reading of given dataset type. For instance, type `fid` data set loads acqp and method file, from parent directory in which the fid file is contained.
        """
        self._read_parameters()

    def unload_parameters(self):
        self._parameters = None

    def add_parameter_file(self, file_type):
        """
        Load additional jcamp-dx file and add it to Dataset parameter space. It is later available via getters,
        or using the dot notation.
        :param file_type: JCAMP-DX file to add to the data set. Must be located in the same folder, or the first proc subfolder.

        **Example:**

        .. highlight:: python
        .. code-block:: python

            from bruker.dataset import Dataset

            dataset = Dataset('.../2dseq')
            dataset.add_parameter_file('method')
            dataset['PVM_DwDir'].value

        """
        try:
            jcampdx = JCAMPDX(self.path.parent / file_type)
        except FileNotFoundError:
            if self.type in ['2dseq','1r','1i']:
                try:
                    jcampdx = JCAMPDX(self.path.parents[2] / file_type)
                except FileNotFoundError:
                    raise FileNotFoundError(file_type)
            if self.type in ['fid','ser','rawdata','traj']:
                try:
                    jcampdx = JCAMPDX(self.path.parent / Path('pdata/1') / file_type)
                except FileNotFoundError:
                    try:
                        jcampdx = JCAMPDX(self.path.parents[1] / file_type)
                    except:
                        raise FileNotFoundError(file_type)

        if not hasattr(self, '_parameters') or self._parameters is None:
            self._parameters = {file_type:jcampdx}
        else:
            self._parameters[file_type] = jcampdx

    def _read_parameters(self):
        """
        Read parameters form the essential JCAMP-DX files.

        :return:
        """
        parameter_files = deepcopy(SUPPORTED[self.type])
        if self._kwargs.get('add_parameters'):
            parameter_files += list(self._kwargs.get('add_parameters'))
        for file_type in parameter_files:
            try:
                self.add_parameter_file(file_type)
            except FileNotFoundError as e:
                # if jcampdx file is required but not found raise ERror
                if file_type in SUPPORTED[self.type]:
                    raise e
                # if jcampdx file is not found, but not required, pass
                else:
                    pass

    def _write_parameters(self, parent):
        for type_, jcampdx in self._parameters.items():
            jcampdx.write(parent / type_)

    """
    PROPERTIES
    """

    def load_properties(self):
        """
        Load properties from two default configuration files. First configuration file contains core properties -
        properties essential for data loading, second contains custom properties - to provide more information about
        given data set, such as the date of measurement, the echo time, etc.

        Some properties depend on values of parameters from JCAMP-DX files which are not essential for creating the
        dataset. For instance, the date property of the fid dataset type is dependent on the AdjStatePerScan. Such
        JCAMP-DX file can be added using the `add_parameter_file` function, then the `load_properties` function can
        be called to reevaluate values of properties, so that the properties dependent on parameters stored in
        non-essential JCAMP-DX files are loaded.

        **Example:**

        .. highlight:: python
        .. code-block:: python

            from bruker.dataset import Dataset

            dataset = Dataset('.../fid')
            dataset.add_parameter_file('AdjStatePerScan')
            dataset.load_properties()
            dataset.date

        """
        self.add_property_file('{}/schema_{}_core.json'.format(str(config_paths['core']),self.type))
        if self._kwargs.get('schema_custom_config'):
            self.add_property_file('{}/{}_custom.json'.format(self._kwargs.get('custom_config'), self.type))
        else:
            self.add_property_file(
                '{}/schema_{}_custom.json'.format(str(config_paths['core']), self.type))

    def unload_properties(self):
        pass

    def add_property_file(self, path):
        with open(path) as f:
            for property in json.load(f).items():
                self._add_property(property)

    def _add_property(self, property):
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
                self._eval_conditions(desc['conditions'])
                try:
                    value = self._make_element(desc['cmd'])
                    self.__setattr__(property[0], value)
                    break
                # if some of the parameters needed for evaluation is missing
                except (KeyError, ValueError):
                    pass

            except (PropertyConditionNotMet, AttributeError):
                pass

    def _make_element(self, cmd):
        """
        Calculate value of a property using the command string.

        :param cmd: command string, or list of command strings
        :return: value of property, or list of values of properties
        """
        if isinstance(cmd, str):
            return eval(self._sub_parameters(cmd))
        elif isinstance(cmd, int) or isinstance(cmd, float):
            return cmd
        elif isinstance(cmd, list):
            element = []
            for cmd_ in cmd:
                element.append(self._make_element(cmd_))
            return element

    def _eval_conditions(self, conditions):
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
                    if not self._make_element(condition):
                        raise PropertyConditionNotMet
                elif isinstance(condition, list):
                    if not self._make_element(condition[0]) in condition[1]:
                        raise PropertyConditionNotMet
            except KeyError:
                raise PropertyConditionNotMet

    def _sub_parameters(self, recipe):
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

    """
    SCHEMA
    """

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

    def unload_schema(self):
        self._schema = None

    """
    DATA
    """

    def load_data(self):
        """

        Load the data file. The data is first read from the binary file to a data vector. Then the data vector is
        deserialized into a data array. The process of deserialization is different for each data set type and is
        implemented in the individual subclasses of the :class:`brukerapi.schemas.Schema`,
        i.e. :class:`brukerapi.schemas.SchemaFid`, :class:`brukerapi.schemas.Schema2dseq`,
        :class:`brukerapi.schemas.SchemaRawdata`, :class:`brukerapi.schemas.SchemaSer`.

        If the object was created with random_access=True, the data is not read,
        instead it can be accessed using sub-arrays.

        **called in the class constructor.**
        """
        if self._kwargs.get('random_access'):
            self._data = DataRandomAccess(self)
        else:
            self._data = self._read_data()

    def unload_data(self):
        """
        Remove the data array from the data set.
        """
        self._data = None

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

    def _write_data(self, path):
        data = self.data.copy()
        data = self._schema.serialize(data, self._schema.layouts)
        self._write_binary_file(path, data, self.shape_storage, self.numpy_dtype)

    def _write_binary_file(self, path, data, storage_layout, dtype):
        fp = np.memmap(path, mode='w+', dtype=dtype, shape=storage_layout, order='F')
        fp[:] = data

    """
    TRAJECTORY
    """

    def load_traj(self, **kwargs):
        if Path(self.path.parent / 'traj').exists() and self.type != 'traj':
            self._traj = Dataset(self.path.parent / 'traj', load=False, random_access=self.random_access)
            self._traj._parameters = self.parameters
            self._traj._schema = SchemaTraj(self._traj, meta=self.schema._meta, sub_params=self.schema._sub_params,
                                           fid=self)
            self._traj.load_data()
        else:
            self._traj = None

    def unload_traj(self):
        self._traj = None

    """
    EXPORT INTERFACE
    """
    def write(self, path, **kwargs):
        """
        Write the Dataset instance to the disk. This consists of writing the binary data file {fid, rawdata, 2dseq,
        ser,...} and respective JCAMP-DX files {method, acqp, visu_pars, reco}.

        :param path: *str* Path to one of the supported data set types.
        :param kwargs:
        :return:
        """

        path = Path(path)

        if path.name != self.type:
            raise DatasetTypeMissmatch

        parent = path.parent

        if not parent.exists():
            os.mkdir(parent)

        self._write_parameters(parent)
        self._write_data(path)

    def report(self, path=None, props=None, verbose=None):
        """
        Save properties to JSON, or YAML file.

        if path is None then save report in-place as path / self.id + '.json'
        if path is a path path to a folder then save report to path / self.id + '.json'
        if path is a json, or yml file save report to path

        :param path: *str* path to a resulting report file
        :param names: *list* names of properties to be exported
        """

        if path is None:
            path = self.path.parent / self.id + '.json'
        elif path.is_dir():
            path = Path(path) / self.id + '.json'

        if verbose:
            print("bruker report: {} -> {}".format(str(self.path), str(path)))

        if path.suffix == '.json':
            self.to_json(path, props=props)
        elif path.suffix == '.yml':
            self.to_yaml(path, props=props)

    def to_json(self, path=None, props=None):
        """
        Save properties to JSON file.

        :param path: *str* path to a resulting report file
        :param names: *list* names of properties to be exported
        """
        if path:
            with open(path, 'w') as json_file:
                    json.dump(self.to_dict(props=props), json_file, indent=4)
        else:
            return json.dumps(self.to_dict(props=props), indent=4)

    def to_yaml(self, path=None, props=None):
        """
        Save properties to YAML file.

        :param path: *str* path to a resulting report file
        :param names: *list* names of properties to be exported
        """
        if path:
            with open(path, 'w') as yaml_file:
                    yaml.dump(self.to_dict(props=props), yaml_file, default_flow_style=False)
        else:
            return yaml.dump(self.to_dict(props=props), default_flow_style=False)

    def to_dict(self, props=None):
        """
        Export properties as dict.

        :param path: *str* path to a resulting report file
        :param names: *list* names of properties to be exported
        """

        if not props:
            props = list(vars(self).keys())

        # list of Dataset properties to be excluded from the export
        reserved = ['_parameters', 'path', '_data', '_traj', '_kwargs', '_schema', 'random_access', 'id', 'study_id', 'exp_id', 'proc_id', 'subj_id']
        props = list(set(props) - set(reserved))

        properties = {}

        for var in props:
            properties[var] = self._encode_property(self.__getattribute__(var))

        return properties

    def _encode_property(self, var):
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
            return [self._encode_property(var_) for var_ in var]
        elif isinstance(var, tuple):
            return self._encode_property(list(var))
        elif isinstance(var, datetime.datetime):
            return str(datetime.datetime)
        else:
            return var

    def query(self, query):
        try:
            if not eval(self._sub_parameters(query)):
                raise FilterEvalFalse
        except (KeyError, AttributeError) as e:
            raise FilterEvalFalse

    """
    PROPERTIES
    """
    @property
    def data(self):
        """Data array.

        :type: *numpy.ndarray*
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

        :type: *numpy.ndarray*
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

        :type: *int*
        """
        return self.data.ndim

    @property
    def shape(self):
        """shape of data array

        :type: *tuple*
        """
        return self.data.shape

