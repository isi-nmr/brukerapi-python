from .jcampdx import JCAMPDX
from .exceptions import *
from .ontology import *

import numpy as np
import re

class Scheme():
    """Base class for all schemes

    """
    def reshape(self, data, dir='FW', **kwargs):
        """ Reshape data according to the content of layouts

        :param data:
        :param dir:
        :param kwargs:
        :return:
        """
        if dir == 'FW':
            return self.reshape_fw(data, **kwargs)
        elif dir == 'BW':
            return self.reshape_bw(data, **kwargs)

    def reload(self):
        """Update contents of scheme, typically after change of parameters

        :return:
        """
        self.unload()
        self.load()

    @property
    def sw(self):
        """Sweep width

        :return: Sweep width [s]
        """
        if self.dataset.type in ['fid','rawdata','ser']:
            return self.dataset.get_float('SW_h')
        elif self.dataset.type in ['2dseq',]:
            return self.dataset.get_array('VisuCoreSize', dtype='float')[0] * self.get_parameter('VisuAcqPixelBandwidth')
        else:
            return None

    @property
    def transmitter_freq(self):
        """Transmitter frequency

        :return: Transmitter frequency [Hz]
        """
        if self.dataset.type in ['fid','rawdata','ser']:
            return self.dataset.get_float('BF1')
        elif self.dataset.type in ['2dseq', ]:
            return self.dataset.get_float('VisuAcqImagingFrequency')
        else:
            return None

    @property
    def flip_angle(self):
        """Repetition time extracted either from acqp (`ACQ_flip_angle`), or visu_pars (`VisuAcqFlipAngle`) files.

        :return: flip angle [°]
        """


        if self.dataset.type in ['fid','rawdata','ser']:
            return self.dataset.get_value('ACQ_flip_angle')
        elif self.dataset.type in ['2dseq', ]:
            return self.dataset.get_float('VisuAcqFlipAngle')
        else:
            return None

    @property
    def TR(self):
        """Repetition time extracted either from acqp (`ACQ_echo_time`), or visu_pars (`VisuAcqEchoTime`) files.

        :return: epetition time [s]
        """
        if self.dataset.type in ['fid','rawdata','ser']:
            return self.dataset.get_value('PVM_RepetitionTime')
        elif self.dataset.type in ['2dseq', ]:
            return self.dataset.get_float('VisuAcqRepetitionTime')
        else:
            return None

    @property
    def TE(self):
        """Echo time extracted either from acqp (`ACQ_echo_time`), or visu_pars (`VisuAcqEchoTime`) files.

        :return: echo time [s]
        """
        if self.dataset.type in ['fid','rawdata','ser']:
            return self.dataset.get_array('ACQ_echo_time')[0]
        elif self.dataset.type in ['2dseq', ]:
            return self.dataset.get_float('VisuAcqEchoTime')
        else:
            return None

    def pv_version_acqp(self):
        """Get version of ParaVision software  from acqp file using ACQ_sw_version parameter.

        :return: ParaVision version: str
        """
        ACQ_sw_version = self.dataset.get_str('ACQ_sw_version')
        if '6.0.1' in ACQ_sw_version:
            return '6.0.1'
        elif '6.0' in ACQ_sw_version:
            return '6.0'
        elif '5.1' in ACQ_sw_version:
            return '5.1'
        elif '360' in ACQ_sw_version:
            return '360'

        return self.dataset.get_str('ACQ_sw_version')

    def pv_version_visu(self):
        """Get version of ParaVision software  from visu_pars file using VisuCreatorVersion parameter.

        :return: ParaVision version: str
        """
        VisuCreatorVersion = self.dataset.get_str('VisuCreatorVersion')
        if '6.0.1' in VisuCreatorVersion:
            return '6.0.1'
        elif '6.0' in VisuCreatorVersion:
            return '6.0'
        elif '5.1' in VisuCreatorVersion:
            return '5.1'
        elif '360' in VisuCreatorVersion:
            return '360'

    def permutation_inverse(self, permutation):
        """Get permutation inverse to the input permutation

        :param inverse permutation: list
        :return:
        """
        inverse = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse[p] = i
        return inverse

class SchemeFid(Scheme):
    """
    AcquisitionScheme class
    """

    def __init__(self, dataset, meta, load=True):
        """

        :param dataset: Dataset
        :param meta: on of the dicts fid_schemes.json
        :param load: bool
        """
        self.dataset = dataset
        self.meta = meta
        self.unload()

        # rises SequenceNotMet exception
        self.validate_sequence()

        # get values of parameters
        self.load_parameters()

        # validate pv version
        self.validate_pv()

        # rises ConditionNotMet exception
        self.validate_conditions()

        if load:
            self.load()

    def load(self):
        """Create private copies of properties

        :return:
        """
        self._pv_version = self.pv_version
        self._numpy_dtype = self.numpy_dtype
        self._block_size = self.block_size
        self._single_acq_length = self.single_acq_length
        self._encoded_dim = self.encoded_dim
        self._layouts = self.layouts

    def unload(self):
        """Delete private copies of properties

        :return:
        """
        self._pv_version = None
        self._numpy_dtype = None
        self._block_size = None
        self._single_acq_length = None
        self._encoded_dim = None
        self._layouts = None

    @property
    def pv_version(self):
        """Version of ParaVision software

        :return: version: str
        """
        if self._pv_version is not None:
            return self._pv_version
        return self.pv_version_acqp()

    @property
    def layouts(self):
        """Dictionary of possible logical layouts of data

        - encoding_space
        - permute
        - k_space

        :return: layouts: dict
        """

        if self._layouts is not None:
            return self._layouts

        layouts = {}

        layouts['encoding_space'] = self.proc_shape_list(self.meta['encoding_space_shape'])
        layouts['permute'] = tuple(self.meta['permute_scheme'])
        layouts['k_space'] = self.proc_shape_list(self.meta['k_space_shape'])

        if "raw_traj_shape" in self.meta:
            layouts['raw_traj'] = self.proc_shape_list(self.meta['raw_traj_shape'])

        if "traj_shape" in self.meta:
            layouts['traj'] = self.proc_shape_list(self.meta['traj_shape'])

        if "traj_permute_scheme" in self.meta:
            layouts['traj_permute'] = self.meta['traj_permute_scheme']

        return layouts

    @property
    def numpy_dtype(self):
        """dtype for reading the binary data

        :return: dtype
        """
        if self._numpy_dtype is not None:
            return self._numpy_dtype

        if self.pv_version == '360':
            return self._numpy_dtype_pv_360()
        else :
            return self._numpy_dtype_pv_5_6()

    def _numpy_dtype_pv_360(self):
        data_format = self.dataset.get_str('ACQ_word_size')
        byte_order = self.dataset.get_str('BYTORDA')

        if data_format == '_32_BIT' and byte_order == 'little':
            return np.dtype('i4').newbyteorder('<')
        else:
            raise NotImplemented('Bruker to numpy data type conversion not implemented for ACQ_word_size '.format(data_format))

    def _numpy_dtype_pv_5_6(self):
        data_format = self.dataset.get_str('GO_raw_data_format')
        byte_order = self.dataset.get_str('BYTORDA')

        if data_format == 'GO_32BIT_SGN_INT' and byte_order == 'little':
            return np.dtype('i4').newbyteorder('<')
        elif data_format == 'GO_16BIT_SGN_INT' and byte_order == 'little':
            return np.dtype('i').newbyteorder('<')
        elif data_format == 'GO_32BIT_FLOAT' and byte_order == 'little':
            return np.dtype('f4').newbyteorder('<')
        elif data_format == 'GO_32BIT_SGN_INT' and byte_order == 'big':
            return np.dtype('i4').newbyteorder('>')
        elif data_format == 'GO_16BIT_SGN_INT' and byte_order == 'big':
            return np.dtype('i').newbyteorder('>')
        elif data_format == 'GO_32BIT_FLOAT' and byte_order == 'big':
            return np.dtype('f4').newbyteorder('>')
        else:
            return np.dtype('i4').newbyteorder('<')

    @property
    def block_size(self):
        """Size of acquisition block

        :return: block_size: int
        """

        if self._block_size is not None:
            return self._block_size

        if self.pv_version == '360':
            return self._block_size_pv_360()
        else:
            return self._block_size_pv_5_6()

    def _block_size_pv_360(self):
        return int(self.dataset.get_value('ACQ_jobs')[0][0] / 2)

    def _block_size_pv_5_6(self):
        ACQ_size = self.dataset.get_array('ACQ_size')
        PVM_EncNReceivers = self.dataset.get_int('PVM_EncNReceivers')
        GO_block_size = self.dataset.get_str('GO_block_size')
        ACQ_dim_desc = self.dataset.get_array('ACQ_dim_desc')

        # TODO ASSUMPTION - spectroscopic data have averaged channel dimension.
        if ACQ_dim_desc[0] == 'Spectroscopic':
            PVM_EncNReceivers = 1

        single_acq = ACQ_size[0] * PVM_EncNReceivers

        if GO_block_size == 'Standard_KBlock_Format':
            return int((np.ceil(single_acq * self.numpy_dtype.itemsize / 1024.) * 1024. / self.numpy_dtype.itemsize) / 2)
        else:
            return int(single_acq / 2)

    @property
    def single_acq_length(self):
        """ Length of single acquisition

        :return:
        """

        if self._single_acq_length is not None:
            return self._single_acq_length

        if self.pv_version == '360':
            return self._single_acq_length_pv_360()
        else:
            return self._single_acq_length_pv_5_6()

    def _single_acq_length_pv_360(self):
        return int(self.dataset.get_value('ACQ_jobs')[0][0] / 2)

    def _single_acq_length_pv_5_6(self):
        ACQ_size = self.dataset.get_array('ACQ_size')
        PVM_EncNReceivers = self.dataset.get_int('PVM_EncNReceivers')
        ACQ_dim_desc = self.dataset.get_list('ACQ_dim_desc')

        if ACQ_dim_desc[0] == 'Spectroscopic':
            PVM_EncNReceivers = 1

        return ACQ_size[0] * PVM_EncNReceivers // 2

    @property
    def encoded_dim(self):
        if self._encoded_dim is not None:
            return self._encoded_dim

        return self.dataset.get_int('ACQ_dim')

    @property
    def dim_size(self):
        return None

    @property
    def dim_type(self):
        return self.meta['k_space_dim_desc']

    @property
    def dim_extent(self):

        extent = []

        for dim in self.dim_type:
            if dim == "kspace_encode_step_0":
                if self.dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spatial':
                    value = 10./self.dataset.get_parameter('ACQ_fov').value[0]
                elif self.dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spectral':
                    value = 1./self.dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_1":
                if self.dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spatial':
                    value = 10./self.dataset.get_parameter('ACQ_fov').value[1]
                elif self.dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spectral':
                    value = 1./self.dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_2":
                value = 10./self.dataset.get_parameter('ACQ_fov').value[2]
            elif dim == "slice":
                value = self.dim_size[dim] * self.dataset.get_value("PVM_SliceThick")\
                        + (self.dim_size[dim] - 1) * self.dataset.get_value("PVM_SPackArrSliceGap")
            elif dim == "repetition":
                value = self.dataset.get_value("NR") * self.dataset.get_value("PVM_RepetitionTime") / 1000.
            elif dim == "channel":
                value = self.dim_size[dim]
            else:
                value = None

            extent.append(value)

        return extent

    @property
    def dim_origin(self):

        origin = []

        for dim in self.dim_type:
            if dim == "kspace_encode_step_0":
                if self.dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spatial':
                    value = 10./self.dataset.get_parameter('ACQ_fov').value[0]
                elif self.dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spectral':
                    value = 1./self.dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_1":
                if self.dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spatial':
                    value = 10./self.dataset.get_parameter('ACQ_fov').value[1]
                elif self.dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spectral':
                    value = 1./self.dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_2":
                value = 10./self.dataset.get_parameter('ACQ_fov').value[2]
            elif dim == "slice":
                value = self.dim_size[dim] * self.dataset.get_value("PVM_SliceThick")\
                        + (self.dim_size[dim] - 1) * self.dataset.get_value("PVM_SPackArrSliceGap")
            elif dim == "repetition":
                value = self.dataset.get_value("NR") * self.dataset.get_value("PVM_RepetitionTime") / 1000.
            elif dim == "channel":
                value = self.dim_size[dim]
            else:
                value = None

            origin.append(value)

        return origin

    @property
    def dim_units(self):
        return None

    @property
    def axes(self):

        axes = []

        for dim in range(len(self.dim_size)):
            axis = np.linspace(self.dim_origin, self.dim_origin + self.dim_extent, self.dim_size)
            axes.append(axis)

        return axes

    def validate_sequence(self):
        PULPROG = self.dataset.get_str('PULPROG', strip_sharp=True)
        for sequence in self.meta['sequences']:
            if sequence == PULPROG:
                return
        raise SequenceNotMet

    def validate_pv(self):
        if self.pv_version not in self.meta['pv_version']:
            raise PvVersionNotMet

    def validate_conditions(self):
        for condition in self.meta['conditions']:
            # substitute parameters in expression string
            for parameter in self.parameters:
                condition = condition.replace(parameter,
                                            "self.parameters[\'%s\']" %
                                   parameter)
            if not eval(condition):
                raise ConditionNotMet(condition)

    def load_parameters(self):
        self.parameters = {}
        for parameter in self.meta['parameters']:
            self.parameters[parameter] = self.dataset.get_value(parameter)

    def proc_shape_list(self, shape_list):

        shape = ()

        # this is necessary to allow to calculate
        for shape_entry in shape_list:
            for parameter in self.parameters:
                shape_entry = re.sub(re.compile(r'\b%s\b' % parameter, re.I),
                                              "self.parameters[\'%s\']" % parameter,
                                              shape_entry)
            shape += (int(eval(shape_entry)),)

        return shape

    def reshape_fw(self, data, **kwargs):

        # Form complex data entries
        data = self._vector_to_vectorcplx(data)

        # Form blocks
        data = self._vectorcplx_to_blocks(data)

        # Trim blocks to acquisitions
        data = self._blocks_to_acquisitions(data)

        # Form encoding space
        data = self._acquisitions_to_encode(data)

        # Permute acquisition dimensions
        data = self._encode_to_permute(data)

        # Form k-space
        data = self._permute_to_kspace(data)

        # Typically for RARE, or EPI
        data = self._reorder_fid_lines(data, dir='FW')

        if self.meta['id'] == 'EPI':
            data = self._mirror_odd_lines(data)

        return data

    def _vector_to_vectorcplx(self, data):
        return data[0::2] + 1j * data[1::2]

    def _vectorcplx_to_blocks(self, data):
        return np.reshape(data, (self.block_size,-1), order='F')

    def _blocks_to_acquisitions(self, data):

        if self.block_size != self.single_acq_length:
            return data[0:self.single_acq_length,:]
        else:
            return data

    def _acquisitions_to_encode(self, data):
        return np.reshape(data, self.layouts['encoding_space'], order='F')

    def _encode_to_permute(self, data):
        return np.transpose(data, self.layouts['permute'])

    def _permute_to_kspace(self, data):
        return np.reshape(data, self.layouts['k_space'], order='F')

    def _reorder_fid_lines(self,data, dir):
        """
        Function to sort phase encoding lines using PVM_EncSteps1
        :param data ndarray in k-space layout:
        :return:
        """
        # TODO when to use?

        # Create local copies of variables
        try:
            PVM_EncSteps1 = self.dataset.get_parameter('PVM_EncSteps1').value
        except KeyError:
            return data

        # Order encoding steps for sorting
        PVM_EncSteps1_sorted = np.argsort(PVM_EncSteps1)

        if dir == 'BW':
            PVM_EncSteps1_sorted = self.permutation_inverse(PVM_EncSteps1_sorted)


        if np.array_equal(PVM_EncSteps1_sorted,PVM_EncSteps1):
            return data

        for index in np.ndindex(data.shape[2:]):
            index = list(index)
            index.insert(0,slice(0,data.shape[1]))
            index.insert(0,slice(0, data.shape[0]))
            index = tuple(index)
            tmp = data[index]
            data[index] = tmp[:,PVM_EncSteps1_sorted]

        return data

    def _mirror_odd_lines(self, data):
        # Both FW and BW run are the same
        # Order encoding steps for sorting

        for index in np.ndindex(data.shape[2:]):
            index_odd = list(index)
            index_odd.insert(0,slice(1,data.shape[1],2))
            index_odd.insert(0,slice(0, data.shape[0]))
            index_odd = tuple(index_odd)
            tmp = data[index_odd]
            data[index_odd] = tmp[::-1,:]
        return data

    def reshape_bw(self, data, **kwargs):

        if self.meta['id'] == 'EPI':
            data = self._mirror_odd_lines(data)

        data = self._reorder_fid_lines(data, dir='BW')



        p = np.array(self.layouts['permute'])
        e = np.array(self.layouts['encoding_space'])
        k = np.array(self.layouts['k_space'])
        ip = self.permutation_inverse(self.layouts['permute'])

        # 4
        data = np.reshape(data, e[p], order='F')

        # 3
        data = np.transpose(data, ip)

        # 2
        data = np.reshape(data, (self.single_acq_length, -1), order='F')

        # 1
        if self.single_acq_length != self.block_size:
            data_ = np.zeros((self.block_size, data.shape[1]), dtype=data.dtype)
            data_[0:self.single_acq_length,:] = data
            data = data_

        data = data.flatten(order='F')

        data_ = np.zeros(2*len(data), dtype=self.numpy_dtype)
        data_[0::2] = np.real(data)
        data_[1::2] = np.imag(data)

        return data_

    def reshape_traj_fw(self, traj):
        traj = np.reshape(traj, self.layouts['raw_traj'], order='F')
        traj = np.transpose(traj, self.layouts['traj_permute'])
        return np.reshape(traj, self.layouts['traj'], order='F')

    def reshape_traj_bw(self, traj):
        traj = np.reshape(traj, self.layouts['raw_traj'], order='F')
        traj = np.transpose(traj, self.layouts['traj_permute'])
        return np.reshape(traj, self.layouts['traj'], order='F').copy()


class SchemeRawdata(Scheme):
    def __init__(self, dataset, load=True):
        self.dataset = dataset

        self.unload()

        if load:
            self.load()

    def unload(self):
        self._pv_version = None
        self._job = None
        self._layouts = None
        self._numpy_dtype = None

    def load(self):
        self._pv_version = self.pv_version
        self._job = self.job
        self._layouts = self.layouts
        self._numpy_dtype = self.numpy_dtype

    @property
    def pv_version(self):
        if self._pv_version is not None:
            return self._pv_version

        return self.pv_version_acqp()

    @property
    def job(self):
        if self._job is not None:
            return self._job

        ACQ_jobs = self.dataset.get_nested_list('ACQ_jobs')

        if self.pv_version in ['5.1','6.0','6.0.1']:
            jobid = self.dataset.subtype.replace('job','')
            jobid = int(jobid)
            return ACQ_jobs[jobid]
        else:
            for job in ACQ_jobs:
                if '<{}>'.format(self.dataset.subtype) == job[-1].lower():
                    return job


    @property
    def layouts(self):
        if self._layouts is not None:
            return self._layouts

        PVM_EncNReceivers = self.dataset.get_value('PVM_EncNReceivers')

        layouts={}
        layouts['raw']=(int(self.job[0]/2),PVM_EncNReceivers , int(self.job[3]))

        return layouts

    @property
    def numpy_dtype(self):
        if self._numpy_dtype is not None:
            return self._numpy_dtype

        return np.dtype('i4')

    @property
    def dim_type(self):
        return ["kspace_encode_step_0","channel", "kspace_encode_step_1"]

    def reshape_fw(self, data, **kwargs):
        data = data[0::2] + 1j * data[1::2]
        data = np.reshape(data, self.layouts['raw'], order='F')
        return data

    def reshape_bw(self, data, **kwargs):
        data = data.copy()
        data = data.flatten(order='F')
        re = np.real(data)
        im = np.imag(data)
        data = np.zeros((2*len(data)), dtype=self.numpy_dtype)
        data[0::2] = re
        data[1::2] = im
        return data

class SchemeSer(Scheme):
    def __init__(self, dataset, load=True):
        self.dataset = dataset
        self.unload()

        if load:
            self.load()

    def unload(self):
        self._pv_version = None
        self._layouts = None
        self._numpy_dtype = None

    def load(self):
        self._pv_version = self.pv_version
        self._layouts = self.layouts
        self._numpy_dtype = self.numpy_dtype

    @property
    def pv_version(self):

        if self._pv_version is not None:
            return self._pv_version

        return self.pv_version_acqp()

    @property
    def layouts(self):
        if self._layouts is not None:
            return self._layouts

        PVM_SpecMatrix = self.dataset.get_value('PVM_SpecMatrix')
        PVM_Matrix = self.dataset.get_value('PVM_Matrix')
        PVM_EncNReceivers = self.dataset.get_value('PVM_EncNReceivers')

        layouts={}
        layouts['raw']=(PVM_SpecMatrix,PVM_EncNReceivers , PVM_Matrix[0], PVM_Matrix[1])

        return layouts

    @property
    def numpy_dtype(self):
        if self._numpy_dtype is not None:
            return self._numpy_dtype

        return np.dtype('i4')

    def reshape_fw(self, data, **kwargs):
        data = data[0::2] + 1j * data[1::2]
        data = np.reshape(data, self.layouts['raw'], order='F')
        return data

    def reshape_bw(self, data, **kwargs):
        raise NotImplemented


class Scheme2dseq(Scheme):
    """
    FrameGroupScheme class

    - vector: data vector as obtained from binary file
    - frames: individual frames combined in
    - framegroups: aldasdasd

    """
    def __init__(self, dataset, load=True):
        self.dataset = dataset
        self.unload()  # sets local copies of properties to None
        if load:
            self.load()

    def load(self):
        """Create local copies of all properties"""
        self._pv_version = self.pv_version
        self._numpy_dtype = self.numpy_dtype
        self._core_size = self.core_size
        self._encoded_dim = self.encoded_dim
        self._is_single_slice = self.is_single_slice
        self._frame_count = self.frame_count
        self._layouts = self.layouts
        self._dim_type = self.dim_type
        self._dim_extent = self.dim_extent
        self._axes = self.axes

    def unload(self):
        """Delete copies of properties"""
        self._pv_version = None
        self._numpy_dtype = None
        self._core_size = None
        self._encoded_dim = None
        self._is_single_slice = None
        self._frame_count = None
        self._layouts = None
        self._dim_type = None
        self._dim_extent = None
        self._axes = None

    def reload(self):
        """Update local copies of properties in case self.parameters have changed"""
        self.unload()
        self.load()

    @property
    def pv_version(self):
        """
        :return: paravision version
        """
        if self._pv_version is not None:
            return self._pv_version

        return self.pv_version_visu()

    @property
    def layouts(self):
        """
        - vector: data vector as obtained from binary file
        - frames: individual frames combined in
        - framegroups: aldasdasd

        :param data: data
        :return layouts: `dict` containing
        """
        if self._layouts is not None:
            return self._layouts

        shapes = {}

        try:
            VisuFGOrderDesc = self.dataset.get_nested_list('VisuFGOrderDesc')
        except:
            VisuFGOrderDesc = []

        dim_size = []

        for i in range(len(self.core_size)):
            dim_size.append(self.core_size[i])

        for i in range(len(VisuFGOrderDesc)):
            dim_size.append(VisuFGOrderDesc[i][0])

        # insert a dummy spatial dimension for single frame acquisitions
        if self.is_single_slice:
            dim_size.insert(2, 1)

        shapes['frame_groups'] = tuple(dim_size)
        shapes['frames'] = self.core_size + (self.frame_count,)

        return shapes

    @property
    def dim_size(self):
        return self.layouts['frame_groups']

    @property
    def dim_type(self):
        if self._dim_type is not None:
            return self._dim_type

        VisuCoreDimDesc = self.dataset.get_array('VisuCoreDimDesc')
        VisuCoreDim = self.dataset.get_int('VisuCoreDim')
        try:
            VisuFGOrderDesc = self.dataset.get_nested_list('VisuFGOrderDesc')
        except:
            VisuFGOrderDesc = []

        dim_type = []

        for i in range(len(VisuCoreDimDesc)):
            dim_type.append(VisuCoreDimDesc[i])

        for i in range(len(VisuFGOrderDesc)):
            dim_type.append(VisuFGOrderDesc[i][1][1:-1])

        # insert a dummy spatial dimension for single frame acquisitions
        if dim_type[0] == 'spatial' and VisuCoreDim < 3 and 'FG_SLICE' not in dim_type:
            dim_type.insert(2, 'spatial')

        return dim_type

    @property
    def dim_origin(self):
        return None

    @property
    def dim_units(self):
        if self._dim_units is not None:
            return self._dim_units
        return None

    @property
    def dim_extent(self):
        if self._dim_extent is not None:
            return self._dim_extent
        return None

    @property
    def axes(self):
        pass

    @property
    def frame_count(self):
        if self._frame_count is not None:
            return self._frame_count
        return self.dataset.get_int('VisuCoreFrameCount')

    @property
    def numpy_dtype(self):
        if self._numpy_dtype is not None:
            return self._numpy_dtype

        VisuCoreWordType = self.dataset.get_str('VisuCoreWordType')
        VisuCoreByteOrder = self.dataset.get_str('VisuCoreByteOrder')

        if VisuCoreWordType == '_32BIT_SGN_INT' and VisuCoreByteOrder == 'littleEndian':
            return np.dtype('int32').newbyteorder('<')
        elif VisuCoreWordType == '_16BIT_SGN_INT' and VisuCoreByteOrder == 'littleEndian':
            return np.dtype('int16').newbyteorder('<')
        elif VisuCoreWordType == '_32BIT_FLOAT' and VisuCoreByteOrder == 'littleEndian':
            return np.dtype('float32').newbyteorder('<')
        elif VisuCoreWordType == '_8BIT_USGN_INT' and VisuCoreByteOrder == 'littleEndian':
            return np.dtype('uint8').newbyteorder('<')
        elif VisuCoreWordType == '_32BIT_SGN_INT' and VisuCoreByteOrder == 'bigEndian':
            return np.dtype('int32').newbyteorder('>')
        elif VisuCoreWordType == '_16BIT_SGN_INT' and VisuCoreByteOrder == 'bigEndian':
            return np.dtype('int16').newbyteorder('>')
        elif VisuCoreWordType == '_32BIT_FLOAT' and VisuCoreByteOrder == 'bigEndian':
            return np.dtype('float32').newbyteorder('>')
        elif VisuCoreWordType == '_8BIT_USGN_INT' and VisuCoreByteOrder == 'bigEndian':
            return np.dtype('uint8').newbyteorder('>')
        else:
            print('Data format not specified correctly!')

    @property
    def core_size(self):
        if self._core_size is not None:
            return self._core_size
        return tuple(self.dataset.get_array('VisuCoreSize'))

    @property
    def encoded_dim(self):
        if self._encoded_dim is not None:
            return self._encoded_dim
        return len(self.core_size)

    @property
    def is_single_slice(self):
        if self._is_single_slice is not None:
            return self._is_single_slice

        VisuCoreDim = self.dataset.get_int('VisuCoreDim')

        if 'FG_SLICE' in self.dim_type:
            is_fg_size = True
        else:
            is_fg_size = False

        if is_fg_size or VisuCoreDim > 2:
            return False
        else:
            return True

    @property
    def rot_matrix(self):
        pass

    def get_rel_fg_index(self, fg_type):
        try:
            return self.fg_list.index(fg_type)
        except:
            raise KeyError('Framegroup {} not found in fg_list'.format(fg_type))

    def reshape_fw(self, data, **kwargs):

        data = self._vector_to_frames(data, **kwargs)

        # scale
        data = self._scale_frame_groups(data, 'FW', **kwargs)

        # frames -> frame_groups
        data = self._frames_to_framegroups(data, **kwargs)

        return data

    def _vector_to_frames(self, data, **kwargs):

        return np.reshape(data, self.layouts['frames'], order='F')

    def _scale_frame_groups(self, data, dir, **kwargs):

        data = data.astype(np.float)
        VisuCoreDataSlope = self.dataset.get_array('VisuCoreDataSlope', dtype='f4')
        VisuCoreDataOffs = self.dataset.get_array('VisuCoreDataOffs', dtype='f4')

        for frame in range(self.frame_count):
            if dir == 'FW':
                data[..., frame] *= VisuCoreDataSlope[frame]
                data[..., frame] += VisuCoreDataOffs[frame]
            elif dir == 'BW':
                data[..., frame] -= VisuCoreDataOffs[frame]
                data[..., frame] /= VisuCoreDataSlope[frame]

        return data

    def _frames_to_framegroups(self, data, **kwargs):
        return np.reshape(data, self.layouts['frame_groups'], order='F')

    def _frames_to_framegroups(self, data, **kwargs):
        return np.reshape(data, self.layouts['frame_groups'], order='F')

    def reshape_bw(self, data, **kwargs):
        data = self._framegroups_to_frames(data, **kwargs)
        data = self._scale_frame_groups(data, 'BW', **kwargs)
        data = self._frames_to_vector(data)
        return data

    def _frames_to_vector(self, data):
        return data.flatten(order='F')

    def _framegroups_to_frames(self, data, **kwargs):
        return np.reshape(data, self.layouts['frames'], order='F')
