from .jcampdx import JCAMPDX
from .exceptions import *

import numpy as np
import re
from copy import deepcopy

class Scheme():
    """Base class for all schemes

    """
    def reshape(self, data, dir='FW', layouts=None, **kwargs):
        """ Reshape data according to the content of layouts

        :param data:
        :param dir:
        :param kwargs:
        :return:
        """

        if layouts is None:
            layouts = self.layouts

        if dir == 'FW':
            return self.reshape_fw(data, layouts, **kwargs)
        elif dir == 'BW':
            return self.reshape_bw(data, layouts, **kwargs)

    def reload(self):
        """Update contents of scheme, typically after change of parameters

        :return:
        """
        self.unload()
        self.load()

    @property
    def sw(self):
        """Sweep width

        :return: Sweep width [Hz]
        """
        if self._dataset.type in ['fid','rawdata','ser']:
            return self._dataset.get_float('SW_h')
        elif self._dataset.type in ['2dseq',]:
            try:
                return self._dataset.get_float('VisuAcqPixelBandwidth')
            except KeyError:
                return None
        else:
            return None

    @property
    def transmitter_freq(self):
        """Transmitter frequency

        :return: Transmitter frequency [Hz]
        """
        if self._dataset.type in ['fid','rawdata','ser']:
            return self._dataset.get_float('BF1')
        elif self._dataset.type in ['2dseq', ]:
            try:
                return self._dataset.get_float('VisuAcqImagingFrequency')
            except KeyError:
                return None
        else:
            return None

    @property
    def flip_angle(self):
        """Repetition time extracted either from acqp (`ACQ_flip_angle`), or visu_pars (`VisuAcqFlipAngle`) files.

        :return: flip angle [°]
        """


        if self._dataset.type in ['fid','rawdata','ser']:
            return self._dataset.get_value('ACQ_flip_angle')
        elif self._dataset.type in ['2dseq', ]:
            try:
                return self._dataset.get_float('VisuAcqFlipAngle')
            except KeyError:
                return None
        else:
            return None

    @property
    def TR(self):
        """Repetition time extracted either from acqp (`ACQ_echo_time`), or visu_pars (`VisuAcqEchoTime`) files.

        :return: epetition time [s]
        """
        if self._dataset.type in ['fid','rawdata','ser']:
            return self._dataset.get_value('PVM_RepetitionTime')
        elif self._dataset.type in ['2dseq', ]:
            try:
                return self._dataset.get_float('VisuAcqRepetitionTime')
            except KeyError:
                return None
        else:
            return None

    @property
    def TE(self):
        """Echo time extracted either from acqp (`ACQ_echo_time`), or visu_pars (`VisuAcqEchoTime`) files.

        :return: echo time [s]
        """
        if self._dataset.type in ['fid','rawdata','ser']:
            return self._dataset.get_array('ACQ_echo_time')[0]
        elif self._dataset.type in ['2dseq', ]:
            try:
                return self._dataset.get_float('VisuAcqEchoTime')
            except KeyError:
                return None
        else:
            return None

    def pv_version_acqp(self):
        """Get version of ParaVision software  from acqp file using ACQ_sw_version parameter.

        :return: ParaVision version: str
        """
        ACQ_sw_version = self._dataset.get_str('ACQ_sw_version')
        if '6.0.1' in ACQ_sw_version:
            return '6.0.1'
        elif '6.0' in ACQ_sw_version:
            return '6.0'
        elif '5.1' in ACQ_sw_version:
            return '5.1'
        elif '360' in ACQ_sw_version:
            return '360'

        return self._dataset.get_str('ACQ_sw_version')

    def pv_version_visu(self):
        """Get version of ParaVision software  from visu_pars file using VisuCreatorVersion parameter.

        :return: ParaVision version: str
        """
        VisuCreatorVersion = self._dataset.get_str('VisuCreatorVersion')
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

    def load_sub_params(self):
        sub_params = {}
        for sub_param in self._meta['sub_params']:
            sub_params[sub_param] = self.value_filter(self._dataset.get_value(sub_param))

        return sub_params

    def value_filter(self, value):
        if isinstance(value, str):
            if value=='Yes':
                return True
            elif value == 'No':
                return False
            else:
                return value
        else:
            return value

    def proc_shape_list(self, shape_list):

        shape = ()

        # this is necessary to allow to calculate
        for shape_entry in shape_list:
            for parameter in self._sub_params:
                shape_entry = re.sub(re.compile(r'\b%s\b' % parameter, re.I),
                                              "self._sub_params[\'%s\']" % parameter,
                                              shape_entry)
            shape += (int(eval(shape_entry)),)

        return shape

    def validate_sequence(self):
        PULPROG = self._dataset.get_str('PULPROG', strip_sharp=True)
        for sequence in self._meta['sequences']:
            if sequence == PULPROG:
                return
        raise SequenceNotMet

    def validate_pv(self):
        if self.pv_version not in self._meta['pv_version']:
            raise PvVersionNotMet

    def validate_conditions(self):
        for condition in self._meta['conditions']:
            # substitute parameters in expression string
            for sub_params in self._sub_params:
                condition = condition.replace(sub_params,
                                            "self._sub_params[\'%s\']" %
                                   sub_params)
            if not eval(condition):
                raise ConditionNotMet(condition)

    def _get_ra_k_space_info(self, layouts, slice_full):

        k_space = []
        k_space_offset = []

        for slc_, size_ in zip(slice_full, layouts['k_space']):
            if isinstance(slc_, slice):
                start = slc_.start if slc_.start else 0
                stop = slc_.stop if slc_.stop else size_
            elif isinstance(slc_, int):
                start = slc_
                stop = slc_ + 1
            k_space.append(stop-start)
            k_space_offset.append(start)
        return tuple(k_space), np.array(k_space_offset)


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
        self._dataset = dataset
        self._meta = meta
        self.unload()

        # rises SequenceNotMet exception
        self.validate_sequence()

        # get values of parameters
        self._sub_params = self.load_sub_params()

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
        self._encoded_dim = self.encoded_dim
        self._layouts = self.layouts


    def unload(self):
        """Delete private copies of properties

        :return:
        """
        self._pv_version = None
        self._numpy_dtype = None
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

        acquisition_length = self.get_acquisition_length()
        block_size = self.get_block_size()

        layouts = {'storage': (block_size,) + self.proc_shape_list(self._meta['block_count'])}
        layouts['encoding_space'] = self.proc_shape_list(self._meta['encoding_space_shape'])
        layouts['permute'] = tuple(self._meta['permute_scheme'])
        layouts['encoding_permuted'] = tuple(np.array(layouts['encoding_space'])[np.array(layouts['permute'])])
        layouts['inverse_permute'] = self.permutation_inverse(layouts['permute'])

        layouts['k_space'] = self.proc_shape_list(self._meta['k_space_shape'])


        if "EPI" in self._meta['id']:
            layouts['acquisition_position'] = (block_size - acquisition_length, acquisition_length)
        else:
            layouts['acquisition_position'] = (0,acquisition_length)

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
        data_format = self._dataset.get_str('ACQ_word_size')
        byte_order = self._dataset.get_str('BYTORDA')

        if data_format == '_32_BIT' and byte_order == 'little':
            return np.dtype('i4').newbyteorder('<')
        else:
            raise NotImplemented('Bruker to numpy data type conversion not implemented for ACQ_word_size '.format(data_format))

    def _numpy_dtype_pv_5_6(self):
        data_format = self._dataset.get_str('GO_raw_data_format')
        byte_order = self._dataset.get_str('BYTORDA')

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

    def get_block_size(self, channels=None):
        """Size of acquisition block

        :return: block_size: int
        """
        if self.pv_version == '360':
            return self._block_size_pv_360(channels)
        else:
            return self._block_size_pv_5_6(channels)

    def _block_size_pv_360(self, channels=None):
        return int(self._dataset.get_value('ACQ_jobs')[0][0] / 2)

    def _block_size_pv_5_6(self, channels=None):
        ACQ_size = self._dataset.get_array('ACQ_size')
        if channels:
            PVM_EncNReceivers = channels
        else:
            PVM_EncNReceivers = self._dataset.get_int('PVM_EncNReceivers')
        GO_block_size = self._dataset.get_str('GO_block_size')
        ACQ_dim_desc = self._dataset.get_array('ACQ_dim_desc')

        # TODO ASSUMPTION - spectroscopic data have averaged channel dimension.
        if ACQ_dim_desc[0] == 'Spectroscopic':
            PVM_EncNReceivers = 1

        single_acq = ACQ_size[0] * PVM_EncNReceivers

        if GO_block_size == 'Standard_KBlock_Format':
            return int((np.ceil(single_acq * self.numpy_dtype.itemsize / 1024.) * 1024. / self.numpy_dtype.itemsize))
        else:
            return int(single_acq)

    def get_acquisition_length(self, channels=None):
        """ Length of single acquisition

        :return:
        """
        if self.pv_version == '360':
            return self._single_acq_length_pv_360(channels)
        else:
            return self._single_acq_length_pv_5_6(channels)

    def _single_acq_length_pv_360(self, channels=None):
        return int(self._dataset.get_value('ACQ_jobs')[0][0])

    def _single_acq_length_pv_5_6(self, channels=None):
        ACQ_size = self._dataset.get_array('ACQ_size')
        GO_block_size = self._dataset.get_str('GO_block_size')
        if channels:
            PVM_EncNReceivers = channels
        else:
            PVM_EncNReceivers = self._dataset.get_int('PVM_EncNReceivers')
        ACQ_dim_desc = self._dataset.get_list('ACQ_dim_desc')

        if ACQ_dim_desc[0] == 'Spectroscopic':
            PVM_EncNReceivers = 1

        if GO_block_size == 'Standard_KBlock_Format':
            return ACQ_size[0] * PVM_EncNReceivers
        else:
            return 2 * np.prod(self._dataset.PVM_EncMatrix,dtype=int)  * PVM_EncNReceivers // self._dataset.NSegments


    @property
    def encoded_dim(self):
        if self._encoded_dim is not None:
            return self._encoded_dim

        return self._dataset.get_int('ACQ_dim')

    @property
    def dim_size(self):
        return None

    @property
    def dim_type(self):
        return self._meta['k_space_dim_desc']

    @property
    def dim_extent(self):

        extent = []

        for dim in self.dim_type:
            if dim == "kspace_encode_step_0":
                if self._dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spatial':
                    value = 10./self._dataset.get_parameter('ACQ_fov').value[0]
                elif self._dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spectral':
                    value = 1./self._dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_1":
                if self._dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spatial':
                    value = 10./self._dataset.get_parameter('ACQ_fov').value[1]
                elif self._dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spectral':
                    value = 1./self._dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_2":
                value = 10./self._dataset.get_parameter('ACQ_fov').value[2]
            elif dim == "slice":
                value = self.dim_size[dim] * self._dataset.get_value("PVM_SliceThick")\
                        + (self.dim_size[dim] - 1) * self._dataset.get_value("PVM_SPackArrSliceGap")
            elif dim == "repetition":
                value = self._dataset.get_value("NR") * self._dataset.get_value("PVM_RepetitionTime") / 1000.
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
                if self._dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spatial':
                    value = 10./self._dataset.get_parameter('ACQ_fov').value[0]
                elif self._dataset.get_parameter('ACQ_dim_desc').value[0] == 'Spectral':
                    value = 1./self._dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_1":
                if self._dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spatial':
                    value = 10./self._dataset.get_parameter('ACQ_fov').value[1]
                elif self._dataset.get_parameter('ACQ_dim_desc').value[1] == 'Spectral':
                    value = 1./self._dataset.get_parameter('PVM_EffSWh')
            elif dim == "kspace_encode_step_2":
                value = 10./self._dataset.get_parameter('ACQ_fov').value[2]
            elif dim == "slice":
                value = self.dim_size[dim] * self._dataset.get_value("PVM_SliceThick")\
                        + (self.dim_size[dim] - 1) * self._dataset.get_value("PVM_SPackArrSliceGap")
            elif dim == "repetition":
                value = self._dataset.get_value("NR") * self._dataset.get_value("PVM_RepetitionTime") / 1000.
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


    def reshape_fw(self, data, layouts, **kwargs):

        data = self._acquisition_trim(data, layouts)

        data = data[0::2, ...] + 1j * data[1::2, ...]

        # Form encoding space
        data = self._acquisitions_to_encode(data, layouts)

        # Permute acquisition dimensions
        data = self._encode_to_permute(data, layouts)

        # Form k-space
        data = self._permute_to_kspace(data, layouts)

        # Typically for RARE, or EPI
        data = self._reorder_fid_lines(data, dir='FW')

        if self._meta['id'] == 'EPI':
            data = self._mirror_odd_lines(data)

        return data

    def _acquisition_trim(self, data, layouts):

        acquisition_offset =  layouts['acquisition_position'][0]
        acquisition_length = layouts['acquisition_position'][1]
        block_length = self.layouts['storage'][0]

        if acquisition_offset>0:
            # trim on channel level acquisition
            blocks = layouts['storage'][-1]
            channels = layouts['k_space'][self.dim_type.index('channel')]
            acquisition_offset=acquisition_offset//channels
            acquisition_length = acquisition_length // channels
            data = np.reshape(data, (-1, channels, blocks), order='F')
            return np.reshape(data[acquisition_offset:acquisition_offset+acquisition_length,:,:],(acquisition_length * channels, blocks), order='F')
        else:
            # trim on acq level
            if acquisition_length != block_length:
                return data[0:acquisition_length,:]
            else:
                return data

    def _acquisitions_to_encode(self, data, layouts):
        return np.reshape(data, layouts['encoding_space'], order='F')

    def _encode_to_permute(self, data, layouts):
        return np.transpose(data, layouts['permute'])

    def _permute_to_kspace(self, data, layouts):
        return np.reshape(data, layouts['k_space'], order='F')

    def _reorder_fid_lines(self,data, dir='FW'):
        """
        Function to sort phase encoding lines using PVM_EncSteps1
        :param data ndarray in k-space layout:
        :return:
        """
        # TODO when to use?

        # Create local copies of variables
        try:
            PVM_EncSteps1 = self._dataset.get_parameter('PVM_EncSteps1').value
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

    def reshape_bw(self, data, layouts, **kwargs):

        if self._meta['id'] == 'EPI':
            data = self._mirror_odd_lines(data)

        data = self._reorder_fid_lines(data, dir='BW')

        data = np.reshape(data, layouts['encoding_permuted'], order='F')

        data = np.transpose(data, layouts['inverse_permute'])

        data = np.reshape(data, (layouts['acquisition_position'][1]//2, layouts['storage'][1]), order='F')

        data_ = np.zeros(layouts['storage'], dtype=self.numpy_dtype, order='F')

        if layouts['acquisition_position'][0]>0:
            channels = layouts['k_space'][self.dim_type.index('channel')]
            data = np.reshape(data, (-1,channels, data.shape[-1]),order='F')
            data_ = np.reshape(data_, (-1,channels, data_.shape[-1]),order='F')
            data_[layouts['acquisition_position'][0]//channels::2,:,:] = data.real
            data_[layouts['acquisition_position'][0]//channels+1::2,:,:] = data.imag
            data = np.reshape(data, (-1, data.shape[-1]),order='F')
            data_ = np.reshape(data_, (-1, data_.shape[-1]),order='F')
        elif layouts['acquisition_position'][1] != layouts['storage'][0]:
            data_[0:layouts['acquisition_position'][1]:2,:] = data.real
            data_[1:layouts['acquisition_position'][1]+1:2,:] = data.imag
        else:
            data_[0::2,:] = data.real
            data_[1::2,:] = data.imag

        return data_

    def ra(self, slice_):

        layouts, layouts_ra = self.get_ra_layouts(slice_)

        """
        random access
        """
        array_ra =  np.zeros(layouts_ra['storage'], dtype=self.numpy_dtype)
        fp = np.memmap(self._dataset.path, dtype=self.numpy_dtype, mode='r',
                       shape=layouts['storage'], order='F')

        for index_ra in np.ndindex(layouts_ra['k_space'][1:]):
            # index of line in the original k_space
            index_full = tuple(i + o for i, o  in zip(index_ra, layouts_ra['k_space_offset'][1:]))

            # index of line in the subarray
            # index_full = self.index_to_data(layouts, (0,) + index_full)
            try:
                index_full = self.index_to_data(layouts, (0,) + index_full)
            except:
                print(index_full)
                index_full = self.index_to_data(layouts, (0,) + index_full)

            # index of line in the subarray
            # index_ra = self.index_to_data(layouts_ra, (0,)+index_ra)
            try:
                index_ra = self.index_to_data(layouts_ra, (0,)+index_ra)
            except:
                print(index_ra)
                index_ra = self.index_to_data(layouts_ra, (0,) + index_ra)


            try:
                array_ra[index_ra] = np.array(fp[index_full])
            except:
                print(index_full)

        layouts_ra['k_space'] = (layouts_ra['k_space'][0]//2,)+layouts_ra['k_space'][1:]
        layouts_ra['encoding_space'] = (layouts_ra['encoding_space'][0]//2,)+layouts_ra['encoding_space'][1:]

        array_ra = self.reshape_fw(array_ra, layouts_ra)

        singletons = tuple(i for i, v in enumerate(slice_) if isinstance(v, int))

        return np.squeeze(array_ra, axis=singletons)

    def get_ra_layouts(self, slice_):
        layouts = deepcopy(self.layouts)
        layouts['k_space'] = (layouts['k_space'][0]*2,)+layouts['k_space'][1:]
        layouts['encoding_space'] = (layouts['encoding_space'][0]*2,)+layouts['encoding_space'][1:]
        layouts['inverse_permute'] = tuple(self.permutation_inverse(layouts['permute']))
        layouts['encoding_permute'] = tuple(layouts['encoding_space'][i] for i in layouts['permute'])
        layouts['channel_index'] = self.dim_type.index('channel')
        layouts['channels'] = layouts['k_space'][layouts['channel_index']]
        layouts['acquisition_position_ch'] = (layouts['acquisition_position'][0]//layouts['channels'],
                                                      layouts['acquisition_position'][1]//layouts['channels'])
        layouts['storage_clear'] = (layouts['acquisition_position'][1], layouts['storage'][1])
        layouts['storage_clear_ch'] = (layouts['storage_clear'][0]//layouts['channels'], layouts['channels'],
                                  layouts['storage'][1])
        layouts['storage_ch'] = (layouts['storage'][0]//layouts['channels'], layouts['channels'], layouts['storage'][1])

        layouts_ra = deepcopy(layouts)

        layouts_ra['k_space'], layouts_ra['k_space_offset'] = self._get_ra_k_space_info(layouts, slice_)
        layouts_ra['channels'] = layouts_ra['k_space'][layouts_ra['channel_index']]
        layouts_ra['acquisition_position'] = (0,self.get_acquisition_length(channels=layouts_ra['channels'])) # delete offset
        # delete offset

        layouts_ra['encoding_space'], layouts_ra['storage'] = self._get_e_ra(layouts, layouts_ra)
        layouts_ra['encoding_permute'] = tuple(layouts_ra['encoding_space'][i] for i in layouts['permute'])

        return layouts, layouts_ra


    def _extrema_init(self, shape):
        min_index = np.array(shape)
        max_index = np.zeros(len(shape), dtype=int)
        return min_index, max_index

    def encode_extrema_update(self, min_enc_index, max_enc_index, enc_index):
        for i in range(len(min_enc_index)):
            if enc_index[i] < min_enc_index[i]:
                min_enc_index[i] = enc_index[i]
            if enc_index[i] > max_enc_index[i]:
                max_enc_index[i] = enc_index[i]

    def index_to_data(self, layout, index):

        # kspace to linear
        channel = index[layout['channel_index']]+1
        index = np.ravel_multi_index(index, layout['k_space'], order='F')

        # linear to encoding permuted
        index = np.unravel_index(index, layout['encoding_permute'], order='F')
        #permute
        index = tuple(index[i] for i in layout['inverse_permute'])
        # encoding space to linear
        index = np.ravel_multi_index(index, layout['encoding_space'], order='F')
        if layout['acquisition_position'][0]>0:
            index = np.unravel_index(index, layout['storage_clear_ch'], order='F')
            index = (index[0] + layout['acquisition_position_ch'][0],)+index[1:]
            index = np.ravel_multi_index(index, layout['storage_ch'], order='F')
        elif layout['acquisition_position'][1] != layout['storage'][0]:
            index = np.unravel_index(index, layout['storage_clear'], order='F')
            index = np.ravel_multi_index(index, layout['storage'], order='F')

        index = np.unravel_index(index, layout['storage'], order='F')

        index = (slice(index[0], index[0]+layout['k_space'][0]),index[1])

        return index

    def _get_e_ra(self, layout_full, layout_ra):
        min_enc_index, max_enc_index = self._extrema_init(layout_full['encoding_space'][1:])
        storage_ra = []
        for index_ra in np.ndindex(layout_ra['k_space'][1:]):
            index_full = (0,)+tuple(i + o for i, o in zip(index_ra, layout_ra['k_space_offset'][1:]))
            channel = index_full[layout_full['channel_index']]+1

            """
            index_k_to_encode
            """

            index_full = np.ravel_multi_index(index_full, layout_full['k_space'], order='F')

            # linear to encoding permuted
            index_full = np.unravel_index(index_full, layout_full['encoding_permute'], order='F')
            # permute
            index_full = tuple(index_full[i] for i in layout_full['inverse_permute'])

            """
            Update encoding space extrema
            """
            self.encode_extrema_update(min_enc_index, max_enc_index, index_full[1:])

            """
            index_encode_to_data
            """
            index_full = np.ravel_multi_index(index_full, layout_full['encoding_space'], order='F')
            index_full = np.unravel_index(index_full, layout_full['storage_clear'], order='F')
            if not index_full[1] in storage_ra:
                storage_ra.append(index_full[1])

        encoding_space_ra = max_enc_index - min_enc_index + 1
        encoding_space_ra = (layout_full['encoding_space'][0],) + tuple(encoding_space_ra)

        storage_ra = (self.get_acquisition_length(channels=layout_ra['channels']), len(storage_ra))

        return encoding_space_ra, storage_ra

    def index_k_to_encode(self, layout, index):
        index = np.ravel_multi_index(index, layout['k_space'], order='F')
        # linear to encoding permuted
        index = np.unravel_index(index, layout['encoding_permute'], order='F')
        #permute
        index = tuple(index[i] for i in layout['inverse_permute'])
        return index

    def index_encode_to_data(self, layout, index):
        channel = index[layout['channel_index']]+1

        index = np.ravel_multi_index(index, layout['encoding_space'], order='F')
        index = np.unravel_index(index, layout['storage'], order='F')

        if layout['acquisition_position'][0]>0:
            first = index[0] + (layout['acquisition_position'][0]// layout['channels']) * channel
        else:
            first = index[0]
        index = (slice(first,first+layout['k_space'][0]),index[1])
        return index



class SchemeTraj(Scheme):
    def __init__(self, dataset, meta=None, load=True, sub_params=None, fid=None):
        self._dataset = dataset
        self._meta = meta
        self._fid = fid
        self.unload()

        # rises SequenceNotMet exception
        self.validate_sequence()

        # get values of subset of parameters used for evaluation of shape lists
        if sub_params:
            self._sub_params = sub_params
        else:
            self._sub_params = self.load_sub_params()

        # validate pv version
        self.validate_pv()

        # rises ConditionNotMet exception
        self.validate_conditions()


        if load:
            self.load()

    def load(self):
        self._pv_version = self.pv_version
        self._layouts = self.layouts

    def unload(self):
        self._pv_version = None
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

        if self._layouts is not None:
            return self._layouts

        layouts = {}

        layouts['storage'] = self.proc_shape_list(self._meta['traj']['storage'])

        layouts['final'] = self.proc_shape_list(self._meta['traj']['final'])

        layouts['permute'] = self._meta['traj']['permute']

        return layouts

    @property
    def numpy_dtype(self):
        return 'float64'

    def reshape_fw(self, data, layouts):
        data = np.transpose(data, layouts['permute'])
        return np.reshape(data, layouts['final'], order='F')

    def reshape_bw(self, data, layouts):
        data = np.transpose(data, layouts['traj_permute'])
        return np.reshape(data, layouts['traj'], order='F')


class SchemeRawdata(Scheme):
    def __init__(self, dataset, load=True):
        self._dataset = dataset

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

        ACQ_jobs = self._dataset.get_nested_list('ACQ_jobs')

        if self.pv_version in ['5.1','6.0','6.0.1']:
            jobid = self._dataset.subtype.replace('job','')
            jobid = int(jobid)
            return ACQ_jobs[jobid]
        else:
            for job in ACQ_jobs:
                if '<{}>'.format(self._dataset.subtype).lower() == job[-1].lower():
                    return job


    @property
    def layouts(self):
        if self._layouts is not None:
            return self._layouts

        PVM_EncNReceivers = self._dataset.get_value('PVM_EncNReceivers')

        layouts={}
        layouts['raw']=(int(self.job[0]/2),PVM_EncNReceivers , int(self.job[3]))
        layouts['storage'] = (2, int(self.job[0]/2),PVM_EncNReceivers , int(self.job[3]))
        layouts['final'] = layouts['raw']

        return layouts

    @property
    def numpy_dtype(self):
        if self._numpy_dtype is not None:
            return self._numpy_dtype

        return np.dtype('i4')

    @property
    def dim_type(self):
        return ["kspace_encode_step_0","channel", "kspace_encode_step_1"]

    def reshape_fw(self, data, layouts, **kwargs):
        return data[0,...] + 1j * data[1,...]

    def reshape_bw(self, data, layouts, **kwargs):
        data_ = np.zeros(layouts['storage'], dtype=self.numpy_dtype, order='F')
        data_[0,...] = data.real
        data_[1, ...] = data.imag
        return data_


class SchemeSer(Scheme):
    def __init__(self, dataset, load=True):
        self._dataset = dataset
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

        PVM_SpecMatrix = self._dataset.get_value('PVM_SpecMatrix')
        PVM_Matrix = self._dataset.get_value('PVM_Matrix')
        PVM_EncNReceivers = self._dataset.get_value('PVM_EncNReceivers')

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
        self._dataset = dataset
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
            VisuFGOrderDesc = self._dataset.get_nested_list('VisuFGOrderDesc')
        except:
            VisuFGOrderDesc = []

        dim_size = []

        for i in range(len(VisuFGOrderDesc)):
            dim_size.append(VisuFGOrderDesc[i][0])

        # insert a dummy spatial dimension for single frame acquisitions
        if self.is_single_slice:
            dim_size.insert(0, 1)

        shapes['frame_groups'] = tuple(dim_size)
        shapes['frames'] = (self.frame_count,)
        shapes['block'] = tuple(self._dataset.VisuCoreSize)
        shapes['storage'] = shapes['block'] + (np.prod(dim_size, dtype=int),)
        shapes['final'] = shapes['block'] + shapes['frame_groups']

        return shapes

    @property
    def final_layout(self):
        return self.layouts['frame_groups']

    @property
    def dim_size(self):
        return self.layouts['frame_groups']

    @property
    def dim_type(self):
        if self._dim_type is not None:
            return self._dim_type

        VisuCoreDimDesc = self._dataset.get_array('VisuCoreDimDesc')
        VisuCoreDim = self._dataset.get_int('VisuCoreDim')
        try:
            VisuFGOrderDesc = self._dataset.get_nested_list('VisuFGOrderDesc')
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
        return self._dataset.get_int('VisuCoreFrameCount')

    @property
    def numpy_dtype(self):
        if self._numpy_dtype is not None:
            return self._numpy_dtype

        VisuCoreWordType = self._dataset.get_str('VisuCoreWordType')
        VisuCoreByteOrder = self._dataset.get_str('VisuCoreByteOrder')

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
        return tuple(self._dataset.get_array('VisuCoreSize'))

    @property
    def encoded_dim(self):
        if self._encoded_dim is not None:
            return self._encoded_dim
        return len(self.core_size)

    @property
    def is_single_slice(self):
        if self._is_single_slice is not None:
            return self._is_single_slice

        VisuCoreDim = self._dataset.get_int('VisuCoreDim')

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

    def reshape_fw(self, data, layouts, scale=True):

        # scale
        data = self._scale_frames(data, 'FW', layouts, scale=scale)

        # frames -> frame_groups
        data = self._frames_to_framegroups(data, layouts)

        return data

    def _scale_frames(self, data, dir, layouts, scale=True, **kwargs):

        if not scale:
            return data

        data = data.astype(np.float)
        VisuCoreDataSlope = self._dataset.get_array('VisuCoreDataSlope', dtype='f4')
        VisuCoreDataOffs = self._dataset.get_array('VisuCoreDataOffs', dtype='f4')
        if 'mask' in layouts:
            VisuCoreDataSlope = VisuCoreDataSlope[layouts['mask'].flatten(order='F')]
            VisuCoreDataOffs = VisuCoreDataOffs[layouts['mask'].flatten(order='F')]

        for frame in range(data.shape[-1]):
            if dir == 'FW':
                data[..., frame] *= VisuCoreDataSlope[frame]
                data[..., frame] += VisuCoreDataOffs[frame]
            elif dir == 'BW':
                data[..., frame] -= VisuCoreDataOffs[frame]
                data[..., frame] /= VisuCoreDataSlope[frame]

        return data

    def _frames_to_framegroups(self, data, layouts, mask=False, **kwargs):
        if mask:
            return np.reshape(data, (-1,) + layouts['frame_groups'], order='F')
        else:
            return np.reshape(data, layouts['block'] + layouts['frame_groups'], order='F')

    def reshape_bw(self, data, layouts, scale=True, ra_mask=None, **kwargs):
        data = self._framegroups_to_frames(data, layouts, **kwargs)
        data = self._scale_frames(data, 'BW', layouts, scale=scale)
        return data

    def _frames_to_vector(self, data):
        return data.flatten(order='F')

    def _framegroups_to_frames(self, data, layouts, mask=False, **kwargs):
        if mask:
            return np.reshape(data, (-1,) + layouts['frames'], order='F')
        else:
            return np.reshape(data, layouts['block'] + layouts['frames'], order='F')

    """
    Random access
    """
    def ra(self, slice_):

        layouts, layouts_ra = self._get_ra_layouts(slice_)

        array_ra = np.zeros(layouts_ra['storage'], dtype=self.numpy_dtype)

        fp = np.memmap(self._dataset.path, dtype=self.numpy_dtype, mode='r',
                       shape=layouts['storage'], order='F')

        for slice_ra, slice_full in self._generate_ra_indices(layouts_ra, layouts):
            array_ra[slice_ra] = np.array(fp[slice_full])

        array_ra = self.reshape_fw(array_ra, layouts_ra)

        singletons = tuple(i for i, v in enumerate(slice_) if isinstance(v, int))

        return np.squeeze(array_ra, axis=singletons)



    def _get_ra_layouts(self, slice_full):

        layouts = deepcopy(self.layouts)
        layouts_ra = deepcopy(self.layouts)

        layouts_ra['mask'] = np.zeros(layouts['frame_groups'], dtype=bool, order='F')
        layouts_ra['mask'][slice_full[self.encoded_dim:]] = True
        layouts_ra['frame_groups'], layouts_ra['frame_groups_offset'] = self._get_ra_shape(layouts_ra['mask'])
        layouts_ra['frames'] = (np.prod(layouts_ra['frame_groups'],dtype=int),)
        layouts_ra['storage'] = layouts_ra['block'] + layouts_ra['frames']
        layouts_ra['final'] = layouts_ra['block'] + layouts_ra['frame_groups']

        return layouts, layouts_ra

    def _get_ra_shape(self, mask):

        axes = []
        for axis in range(mask.ndim):
            axes.append(tuple(i for i in range(mask.ndim) if i!=axis))

        ra_shape = []
        ra_offset = []
        for axis in axes:
            ra_shape.append(np.count_nonzero(np.count_nonzero(mask,axis=axis)))
            ra_offset.append(np.argmax(np.count_nonzero(mask, axis=axis)))

        return tuple(ra_shape), np.array(ra_offset)

    def _generate_ra_indices(self, layouts_ra, layouts):

        for index_ra in np.ndindex(layouts_ra['final'][self.encoded_dim:]):
            index = tuple(np.array(index_ra) + layouts_ra['frame_groups_offset'])
            index = tuple(0 for i in range(self.encoded_dim)) + index
            index_ra = tuple(0 for i in range(self.encoded_dim)) + index_ra

            index_ra = np.ravel_multi_index(index_ra, layouts_ra['final'], order='F')
            index = np.ravel_multi_index(index, layouts['final'], order='F')

            index_ra = np.unravel_index(index_ra, layouts_ra['storage'], order='F')
            index = np.unravel_index(index, layouts['storage'], order='F')

            slice_ra = tuple(slice(None) for i in range(self.encoded_dim)) + index_ra[self.encoded_dim:]
            slice_full = tuple(slice(None) for i in range(self.encoded_dim)) + index[self.encoded_dim:]
            yield slice_ra, slice_full
