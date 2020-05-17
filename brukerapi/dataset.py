from .exceptions import *
from .schemes import *

from pathlib2 import Path
import json
import numpy as np
import os
import os.path

FID_SCHEMES_PATH = str(Path(__file__).parents[0]  / 'fid_schemes.json')
SUPPORTED = {
    'fid': ['acqp', 'method'],
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

    Meta data essential for construction of scheme and manipulation with the binary data file.

    - **scheme**: one of :py:class:`~bruker.schemes.SchemeFid`:py:class:`~bruker.schemes.Scheme2dseq`:py:class:`~bruker.schemes.SchemeSer`:py:class:`~bruker.schemes.SchemeRawdata`

    An object encapsulating all functionality dependent on metadata. It provides method to reshape data.

    - **data**: :py:class:`numpy.ndarray`

    Array containing the data read from any of the supported binary files.

    **Example:**

    .. highlight:: python
    .. code-block:: python

        from bruker.dataset import Dataset

        dataset = Dataset('path/2dseq')

    """

    def __init__(self, path, load=True, **kwargs):
        """Constructor of Dataset

        Dataset can be constructed either by passing a path to one of the SUPPORTED binary files, or to a directory
        containing it. It is possible, to create an empty object using the load switch.

        :param path: **str** path to dataset
        :param load: **bool** when false, empty Dataset is created
        """
        self.path = Path(path)

        if not self.path.exists():
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
        self.validate(load)

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
        return self.type

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
        Load parameters, scheme and data. In case, there is a traj file related to a fid file, traj is loaded as well.
        """

        self.load_parameters()
        self.load_scheme()
        self.load_data()

        if 'traj_shape' in self.scheme.layouts:
            self.load_traj()

    def load_parameters(self):
        """
        Load all parameters essential for reading of given dataset type. For instance, type `fid` data set loads acqp and method file, from parent directory in which the fid file is contained.
        """
        self._parameters = self._read_parameters()

    def load_scheme(self):
        """
        Load the scheme for given data set.
        """
        if self.type == 'fid':
            with open(FID_SCHEMES_PATH) as json_file:
                # Search trough individual acquisition schemes.
                for scheme in json.load(json_file).values():
                    try:
                        self._scheme = SchemeFid(self, scheme)
                        return
                    except (SequenceNotMet, ConditionNotMet, PvVersionNotMet):
                        continue
            raise UnknownAcqSchemeException
        elif self.type == '2dseq':
            self._scheme = Scheme2dseq(self)
        elif self.type == 'rawdata':
            self._scheme = SchemeRawdata(self)
        elif self.type == 'ser':
            self._scheme = SchemeSer(self)

    def load_data(self, **kwargs):
        """

        :return:
        """
        if kwargs.get('READ_DATA') is False:
            return None
        else:
            self._data = self._read_data()

    def load_traj(self, **kwargs):
        self._traj = self._read_traj(self.path.parent/ 'traj', self.scheme)

    def unload(self):
        self.unload_parameters()
        self.unload_scheme()
        self.unload_data()
        self.unload_traj()

    def unload_parameters(self):
         self._parameters = None

    def unload_scheme(self):
        self._scheme = None

    def unload_data(self):
        self._data = None

    def unload_traj(self):
        self._traj = None

    """
    READERS/WRITERS
    """
    def _read_parameters(self, **kwargs):
        """Read parameters form all JCAMPDX files in the directory

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
        data = self._read_binary_file(self.path, self._scheme.numpy_dtype)
        return self._scheme.reshape(data, dir='FW')

    def _read_binary_file(self, path, dtype):
        """Read Bruker binary file

        Parameters
        ----------
        path
        dtype - numpy dtype obtained using GO_raw_data_format and BYTORDA, or VisuCoreWordType and VisuCoreByteOrder parameters

        Returns
        -------
        1D ndarray containing the full data vector
        """
        with path.open("rb") as f:
            return np.fromfile(f, dtype=dtype)

    def _read_traj(self, path, scheme, **kwargs):
        """Read trajectory data

        Parameters
        ----------
        path
        scheme
        kwargs

        Returns
        -------

        """
        # If user specifies not to read traj. Trajectory can be loaded later using Dataset.load_traj()
        if kwargs.get('READ_TRAJ') is False:
            return None

        traj = self._read_binary_file(path, 'float64')
        return self.scheme.reshape_traj_fw(traj)

    def write(self, path, **kwargs):
        """
        Write a Dataset into a file given by file.

        :param path: Path to one of the supported Datasets
        :param kwargs:
        :return:
        """

        path = Path(path)
        parent = path.parent

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
        data = self.scheme.reshape(data, dir="BW")
        data = data.astype(self.scheme.numpy_dtype)
        self._write_binary_file(path, self.scheme.numpy_dtype, data)

    def _write_binary_file(self, path, dtype, data):
        with path.open("wb") as f:
            return data.tofile(f)

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
        raise KeyError

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
    def data(self, slc=None):
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
            return self._traj
        else:
            raise TrajNotLoaded

    @traj.setter
    def traj(self,value):
        self._traj = value

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
    def scheme(self):
        if self._scheme is not None:
            return self._scheme
        else:
            raise SchemeNotLoaded

    @property
    def dim(self):
        """number of dimensions of the data array

        :type: int
        """
        return self.data.ndim


    @property
    def shape(self,index=None):
        """shape of data array

        :type: tuple
        """
        if index is not None:
            return self.data.shape[index]
        else:
            return self.data.shape


    @property
    def encoded_dim(self):
        """dimensionality of acquisition

        :type: int
        """
        return self.scheme.encoded_dim

    @property
    def rotation_matrix(self):
        return self.scheme.rotation_matrix


    @property
    def axes(self):
        return self.scheme.axes

    @property
    def dim_type(self):
        """description of each dimension

        :type: numpy.ndarray
        """
        return self.scheme.dim_type

    @property
    def dim_size(self):
        """number of samples in each dimension

        :type: numpy.ndarray
        """
        return self.scheme.dim_size

    @property
    def dim_extent(self):
        """extent of each dimension

        :type: numpy.ndarray
        """
        return self.scheme.dim_extent

    @property
    def dim_origin(self):
        """origin of each dimension

        :type: numpy.ndarray
        """
        return self._scheme.dim_origin

    @property
    def pv_version(self):
        """Version of ParaVision software

        :type: str
        """
        return self.scheme.pv_version

    @property
    def sw(self):
        """Sweep width [s]

        :type: float
        """
        return self.scheme.sw

    @property
    def transmitter_freq(self):
        """transmitter frequency [Hz]

        :type: float
        """
        return self.scheme.transmitter_freq

    @property
    def flip_angle(self):
        """flip angle [°]

        :type: float
        """
        return self.scheme.flip_angle

    @property
    def TR(self):
        """repetition time [s]

        :type: float
        """
        return self.scheme.TR

    @property
    def TE(self):
        """echo time [s]

        :type: float
        """
        return self.scheme.TE
