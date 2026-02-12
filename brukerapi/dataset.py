import datetime
import json
import os
import os.path
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .data import DataRandomAccess
from .exceptions import (
    DataNotLoaded,
    DatasetTypeMissmatch,
    FilterEvalFalse,
    IncompleteDataset,
    NotADatasetDir,
    ParametersNotLoaded,
    PropertyConditionNotMet,
    SchemeNotLoaded,
    TrajNotLoaded,
    UnsuportedDatasetType,
)
from .jcampdx import JCAMPDX
from .schemas import Schema2dseq, SchemaFid, SchemaRawdata, SchemaSer, SchemaTraj

LOAD_STAGES = {
    "empty": 0,
    "parameters": 1,
    "properties": 2,
    "all": 3,
}

# Dict of default dataset states
DEFAULT_STATES = {
    "fid": {
        "parameter_files": ["acqp", "method"],
        "property_files": [Path(__file__).parents[0] / "config/properties_fid_core.json", Path(__file__).parents[0] / "config/properties_fid_custom.json"],
        "load": LOAD_STAGES["all"],
        "mmap": False,
    },
    'fid_proc': {
        "parameter_files" : ['acqp', 'method'],
        "property_files": [Path(__file__).parents[0] / 'config/properties_fid_core.json', Path(__file__).parents[0] / 'config/properties_fid_custom.json'],
        "load": LOAD_STAGES['all'],
        "mmap": False
    },
    '2dseq': {
        "parameter_files": ['visu_pars'],
        "property_files": [Path(__file__).parents[0] / 'config/properties_2dseq_core.json', Path(__file__).parents[0] / 'config/properties_2dseq_custom.json'],
        "load": LOAD_STAGES['all'],
        "scale": True,
        "mmap": False,
    },
    "traj": {
        "parameter_files": ["acqp", "method"],
        "property_files": [Path(__file__).parents[0] / "config/properties_traj_core.json", Path(__file__).parents[0] / "config/properties_traj_custom.json"],
        "load": LOAD_STAGES["all"],
        "mmap": False,
    },
    "ser": {
        "parameter_files": ["acqp", "method"],
        "property_files": [Path(__file__).parents[0] / "config/properties_ser_core.json", Path(__file__).parents[0] / "config/properties_ser_custom.json"],
        "load": LOAD_STAGES["all"],
        "mmap": False,
    },
    "rawdata": {
        "parameter_files": ["acqp", "method"],
        "property_files": [Path(__file__).parents[0] / "config/properties_rawdata_core.json", Path(__file__).parents[0] / "config/properties_rawdata_custom.json"],
        "load": LOAD_STAGES["all"],
        "mmap": False,
    },
}

RELATIVE_PATHS = {
    "fid": {
        "method": "./method",
        "acqp": "./acqp",
        "subject": "../subject",
        "reco": "./pdata/1/reco",
        "visu_pars": "./pdata/1/visu_pars",
        "AdjStatePerScan": "./AdjStatePerScan",
        "AdjStatePerStudy": "../AdjStatePerStudy",
    },
    "fid_proc": {
        "method": "../../method",
        "acqp": "../../acqp",
        "subject": "../../../subject",
        "reco": "./reco",
        "visu_pars": "./visu_pars",
        "AdjStatePerScan": "../../AdjStatePerScan",
        "AdjStatePerStudy": "../../../AdjStatePerStudy",
    },
    "2dseq": {
        "method": "../../method",
        "acqp": "../../acqp",
        "subject": "../../../subject",
        "reco": "./reco",
        "visu_pars": "./visu_pars",
        "AdjStatePerScan": "../../AdjStatePerScan",
        "AdjStatePerStudy": "../../../AdjStatePerStudy",
    },
    "traj": {
        "method": "./method",
        "acqp": "./acqp",
        "subject": "../subject",
        "reco": "./pdata/1/reco",
        "visu_pars": "./pdata/1/visu_pars",
        "AdjStatePerScan": "./AdjStatePerScan",
        "AdjStatePerStudy": "../AdjStatePerStudy",
    },
    "rawdata": {
        "method": "./method",
        "acqp": "./acqp",
        "subject": "../subject",
        "reco": "./pdata/1/reco",
        "visu_pars": "./pdata/1/visu_pars",
        "AdjStatePerScan": "./AdjStatePerScan",
        "AdjStatePerStudy": "../AdjStatePerStudy",
    },
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

        dataset = Dataset("path/2dseq")

    """

    def __init__(self, path, **state):
        """Constructor of Dataset

        Dataset can be constructed either by passing a path to one of the SUPPORTED binary files, or to a directory
        containing it. It is possible, to create an empty object using the load switch.

        :param path: **str** path to dataset
        :raise: :UnsupportedDatasetType: In case `Dataset.type` is not in SUPPORTED
        :raise: :IncompleteDataset: If any of the JCAMP-DX files, necessary to create a Dataset instance is missing

        """
        self.path = Path(path)

        if not self.path.exists() and state.get("load") is not LOAD_STAGES["empty"]:
            raise FileNotFoundError(self.path)

        # directory constructor
        if self.path.is_dir() and state.get("load"):
            content = os.listdir(self.path)
            if "fid" in content:
                self.path = self.path / "fid"
            elif "2dseq" in content:
                self.path = self.path / "2dseq"
            else:
                raise NotADatasetDir(self.path)

        self.type = self.path.stem
        self.subtype = self.path.suffix
        if self.subtype:
            self.subtype = self.subtype[1:]  # remove the dot from the suffix
        self._properties = []

        # set
        self._set_state(state)

        # validate path
        self._validate()

        # load data if the load kwarg is true
        self.load()

    def __enter__(self):
        self._state["load"] = LOAD_STAGES["all"]
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        self._state["load"] = LOAD_STAGES["empty"]

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

    def __contains__(self, item):
        return any(item in parameter_file for parameter_file in self._parameters.values())

    def __call__(self, **kwargs):
        self._set_state(kwargs)
        return self

    def _set_state(self, passed):
        result = deepcopy(DEFAULT_STATES[self.type])

        if "parameter_files" in passed:
            passed["parameter_files"] = result["parameter_files"] + passed["parameter_files"]

        if "property_files" in passed:
            passed["property_files"] = result["property_files"] + passed["property_files"]

        result.update(passed)
        self._state = result

    def _validate(self):
        """Validate Dataset

        Check whether the dataset type is supported and complete. Dataset is allowed to be incomplete when load is
        set to `False`.

        :raise: :UnsuportedDatasetType: In case `Dataset.type` is not in SUPPORTED
        :raise: :IncompleteDataset: If any of the JCAMP-DX files, necessary to create a Dataset instance is missing
        """

        # Check whether dataset file is supported
        if self.type not in DEFAULT_STATES:
            raise UnsuportedDatasetType(self.type)

        # Check whether all necessary JCAMP-DX files are present
        if self._state.get('load') >= LOAD_STAGES['parameters']:
            for i in DEFAULT_STATES[self.type]['parameter_files']:
                param_path = self.path.parent / RELATIVE_PATHS[self.type][i]
                if i not in set(os.listdir(str(param_path.parent))):
                    raise IncompleteDataset


    def load(self):
        """
        Load parameters, properties, schema and data. In case, there is a traj file related to a fid file,
        traj is loaded as well.
        """

        if self._state["load"] is LOAD_STAGES["empty"]:
            return

        self.load_parameters()

        if self._state["load"] is LOAD_STAGES["parameters"]:
            return

        self.load_properties()

        if self._state["load"] is LOAD_STAGES["properties"]:
            return

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
        Load all parameters essential for reading of given dataset type.
        For instance, type `fid` data set loads acqp and method file, from parent directory in which the fid file is contained.
        """
        self._read_parameters()

    def unload_parameters(self):
        self._parameters = None

    def add_parameter_file(self, file):
        """
        Load additional jcamp-dx file and add it to Dataset parameter space. It is later available via getters,
        or using the dot notation.
        :param file_type: JCAMP-DX file to add to the data set. Must be located in the same folder, or the first proc subfolder.

        **Example:**

        .. highlight:: python
        .. code-block:: python

            from bruker.dataset import Dataset

            dataset = Dataset(".../2dseq")
            dataset.add_parameter_file("method")
            dataset["PVM_DwDir"].value

        """
        path = self.path.parent / RELATIVE_PATHS[self.type][file]

        if not hasattr(self, "_parameters") or self._parameters is None:
            self._parameters = {path.name: JCAMPDX(path)}
        else:
            self._parameters[path.name] = JCAMPDX(path)

    def _read_parameters(self):
        """
        Read parameters form the essential JCAMP-DX files.

        :return:
        """
        parameter_files = self._state["parameter_files"]
        for file in parameter_files:
            try:
                self.add_parameter_file(file)
            except FileNotFoundError as e:
                # if jcampdx file is required but not found raise Error
                if file in DEFAULT_STATES[self.type]["parameter_files"]:
                    raise e
                # if jcampdx file is not found, but not required, pass
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

            dataset = Dataset(".../fid")
            dataset.add_parameter_file("AdjStatePerScan")
            dataset.load_properties()
            dataset.date

        """
        for file in self._state["property_files"]:
            self.add_property_file(file)

        self._state["load_properties"] = True

    def unload_properties(self):
        for property in self._properties:
            if hasattr(self, property):
                delattr(self,property)
        self._properties = []
        self._state["load_properties"] = False

    def reload_properties(self):
        self.unload_properties()
        self.load_properties()

    def add_property_file(self, path):
        with open(path) as f:
            for property in json.load(f).items():
                self._add_property(property)

    def _add_property(self, property):
        """Add property to the dataset and schema

        * Evaluate the condition for a given command if these are fulfilled, the next step follows, otherwise,
            the next command is processed.
        * Get the value of the property using command and parameters of the dataset.
        * Set the property of a dataset
        * Set the property of a schema

        :param property: tuple containing the name of the property and a list of possible commands to create it
        """
        for desc in property[1]:
            try:
                self._eval_conditions(desc["conditions"])
                try:
                    value = self._make_element(desc["cmd"])
                    self.__setattr__(property[0], value)

                    if not hasattr(self, "_properties"):
                        self._properties = [
                            property[0],
                        ]
                    else:
                        self._properties.append(property[0])

                    break
                # if some of the parameters needed for evaluation is missing
                except (KeyError, ValueError, IndexError):
                    pass

            except (PropertyConditionNotMet, AttributeError, IndexError):
                pass

    def _make_element(self, cmd):
        """
        Calculate value of a property using the command string.

        :param cmd: command string, or list of command strings
        :return: value of property, or list of values of properties
        """
        if isinstance(cmd, str):
            return eval(self._sub_parameters(cmd))
        if isinstance(cmd, (int, float)):
            return cmd
        if isinstance(cmd, list):
            element = []
            for cmd_ in cmd:
                element.append(self._make_element(cmd_))
            return element
        return None

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
                elif isinstance(condition, list) and self._make_element(condition[0]) not in condition[1]:
                    raise PropertyConditionNotMet
            except KeyError:
                raise PropertyConditionNotMet from KeyError

    def _sub_parameters(self, recipe):
        # entries with property e.g. VisuFGOrderDesc.nested to self._dataset['VisuFGOrderDesc'].nested
        for match in re.finditer(r"#[a-zA-Z0-9_]+\.[a-zA-Z]+", recipe):
            m = re.match("#[a-zA-Z0-9_]+", match.group())
            recipe = recipe.replace(m.group(), f"self['{m.group()[1:]}']")
        # entries without property e.g. VisuFGOrderDesc to self._dataset['VisuFGOrderDesc'].value
        for match in re.finditer("@[a-zA-Z0-9_]+", recipe):
            recipe = recipe.replace(match.group(), f"self.{match.group()[1:]}")
        for match in re.finditer("#[a-zA-Z0-9_]+", recipe):
            recipe = recipe.replace(match.group(), f"self['{match.group()[1:]}'].value")
        return recipe

    """
    SCHEMA
    """

    def load_schema(self):
        """
        Load the schema for given data set.
        """
        if self.type in ["fid", "fid_proc"]:
            self._schema = SchemaFid(self)
        elif self.type == "2dseq":
            self._schema = Schema2dseq(self)
        elif self.type == "rawdata":
            self._schema = SchemaRawdata(self)
        elif self.type == "ser":
            self._schema = SchemaSer(self)
        elif self.type == "traj":
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
        if self._state["mmap"]:
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
        try:
            assert os.stat(str(path)).st_size >= np.prod(shape) * dtype.itemsize
        except AssertionError:
            raise ValueError("Dimension mismatch") from AssertionError
        return np.array(np.memmap(path, dtype=dtype, shape=shape, order="F")[:])

    def _write_data(self, path):
        data = self.data.copy()
        data = self._schema.serialize(data, self._schema.layouts)
        self._write_binary_file(path, data, self.shape_storage, self.numpy_dtype)

    def _write_binary_file(self, path, data, storage_layout, dtype):
        fp = np.memmap(path, mode="w+", dtype=dtype, shape=storage_layout, order="F")
        fp[:] = data

    """
    TRAJECTORY
    """

    def load_traj(self, **kwargs):
        if Path(self.path.parent / "traj").exists() and self.type != "traj":
            self._traj = Dataset(self.path.parent / "traj", load=False, random_access=self.random_access)
            self._traj._parameters = self.parameters
            self._traj._schema = SchemaTraj(self._traj, meta=self.schema._meta, sub_params=self.schema._sub_params, fid=self)
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

        if path.name.split(".")[0] != self.type:
            raise DatasetTypeMissmatch

        parent = path.parent

        if not parent.exists():
            os.makedirs(parent, exist_ok=True)

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
            path = self.path.parent / self.id + ".json"
        elif path.is_dir():
            path = Path(path) / self.id + ".json"

        if verbose:
            print(f"bruker report: {self.path!s} -> {path!s}")

        if path.suffix == ".json":
            self.to_json(path, props=props)
        elif path.suffix == ".yml":
            self.to_yaml(path, props=props)

    def to_json(self, path=None, props=None):
        """
        Save properties to JSON file.

        :param path: *str* path to a resulting report file
        :param names: *list* names of properties to be exported
        """
        if path:
            with open(path, "w") as json_file:
                json.dump(self.to_dict(props=props), json_file, indent=4)
        else:
            return json.dumps(self.to_dict(props=props), indent=4)
        return None

    def to_yaml(self, path=None, props=None):
        """
        Save properties to YAML file.

        :param path: *str* path to a resulting report file
        :param names: *list* names of properties to be exported
        """
        if path:
            with open(path, "w") as yaml_file:
                yaml.dump(self.to_dict(props=props), yaml_file, default_flow_style=False)
        else:
            return yaml.dump(self.to_dict(props=props), default_flow_style=False)
        return None

    def to_dict(self, props=None):
        """
        Export properties as dict.

        :param path: *str* path to a resulting report file
        :param names: *list* names of properties to be exported
        """

        if not props:
            props = list(vars(self).keys())

        # list of Dataset properties to be excluded from the export
        reserved = ["_parameters", "path", "_data", "_traj", "_state", "_schema", "random_access", "study_id", "exp_id", "proc_id", "subj_id", "_properties"]
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
        if isinstance(var, (np.integer, np.int32)):
            return int(var)
        if isinstance(var, np.floating):
            return float(var)
        if isinstance(var, np.ndarray):
            return var.tolist()
        if isinstance(var, np.dtype):
            return var.name
        if isinstance(var, list):
            return [self._encode_property(var_) for var_ in var]
        if isinstance(var, tuple):
            return self._encode_property(list(var))
        if isinstance(var, (datetime.datetime, str)):
            return str(var)
        return var

    def query(self, query):
        if isinstance(query, str):
            query = [query]

        for q in query:
            try:
                if not eval(self._sub_parameters(q)):
                    raise FilterEvalFalse
            except (KeyError, AttributeError) as e:
                raise FilterEvalFalse from e

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
        raise DataNotLoaded

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def traj(self):
        """Trajectory array loaded from a `traj` file

        :type: *numpy.ndarray*
        """
        if self._traj is not None:
            return self._traj.data
        raise TrajNotLoaded

    @property
    def parameters(self):
        if self._parameters is not None:
            return self._parameters
        raise ParametersNotLoaded

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def schema(self):
        if self._schema is not None:
            return self._schema
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
