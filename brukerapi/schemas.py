from typing import Dict
from .jcampdx import JCAMPDX
from .exceptions import *
import numpy as np
import re
from copy import deepcopy
from pathlib import Path
import json


config_paths = {
    'core': Path(__file__).parents[0]  / "config",
    'custom': Path(__file__).parents[0]  / "config"
}


class Schema():
    """Base class for all schemes

    """
    def __init__(self, dataset):
        self._dataset = dataset

    def permutation_inverse(self, permutation):
        """Get permutation inverse to the input permutation

        :param inverse permutation: list
        :return:
        """
        inverse = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse[p] = i
        return inverse

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


class SchemaFid(Schema):
    """
    SchemeFid class
    """

    @property
    def layouts(self):
        """Dictionary of possible logical layouts of data

        - encoding_space
        - permute
        - k_space

        :return: layouts: dict
        """

        layouts = {'storage': (self._dataset.block_size,) + (self._dataset.block_count,)}
        layouts['encoding_space'] = self._dataset.encoding_space
        layouts['permute'] = self._dataset.permute
        layouts['encoding_permuted'] = tuple(np.array(layouts['encoding_space'])[np.array(layouts['permute'])])
        layouts['inverse_permute'] = self.permutation_inverse(layouts['permute'])
        layouts['k_space'] = self._dataset.k_space

        if "EPI" in self._dataset.scheme_id:
            layouts['acquisition_position'] = (self._dataset.block_size - self._dataset.acq_lenght, self._dataset.acq_lenght)
        else:
            layouts['acquisition_position'] = (0, self._dataset.acq_lenght)

        return layouts

    def deserialize(self, data, layouts):

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

        if 'EPI' in self._dataset.scheme_id:
            data = self._mirror_odd_lines(data)

        return data

    def _acquisition_trim(self, data, layouts):

        acquisition_offset =  layouts['acquisition_position'][0]
        acquisition_length = layouts['acquisition_position'][1]
        block_length = self.layouts['storage'][0]

        if acquisition_offset>0:
            # trim on channel level acquisition
            blocks = layouts['storage'][-1]
            channels = self._dataset.channels
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
            PVM_EncSteps1 = self._dataset['PVM_EncSteps1'].value
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

    def serialize(self, data, layouts):

        if 'EPI' in self._dataset.scheme_id:
            data = self._mirror_odd_lines(data)

        data = self._reorder_fid_lines(data, dir='BW')

        data = np.reshape(data, layouts['encoding_permuted'], order='F')

        data = np.transpose(data, layouts['inverse_permute'])

        data = np.reshape(data, (layouts['acquisition_position'][1]//2, layouts['storage'][1]), order='F')

        data_ = np.zeros(layouts['storage'], dtype=self._dataset.numpy_dtype, order='F')

        if layouts['acquisition_position'][0]>0:
            channels = layouts['k_space'][self._dataset.dim_type.index('channel')]
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


class SchemaTraj(Schema):

    @property
    def layouts(self):

        layouts = {}

        layouts['storage'] = self._dataset.shape_storage
        layouts['final'] = self._dataset.final
        layouts['permute'] = self._dataset.permute

        return layouts

    def deserialize(self, data, layouts):
        data = np.transpose(data, layouts['permute'])
        return np.reshape(data, layouts['final'], order='F')

    def serialize(self, data, layouts):
        data = np.transpose(data, layouts['traj_permute'])
        return np.reshape(data, layouts['traj'], order='F')


class SchemaRawdata(Schema):

    @property
    def layouts(self):
        layouts={}
        layouts['raw']=(int(self._dataset.job_desc[0]/2), self._dataset.channels , int(self._dataset.job_desc[3]))
        layouts['shape_storage'] = (2, int(self._dataset.job_desc[0]/2), self._dataset.channels , int(self._dataset.job_desc[
                                                                                                  3]))
        layouts['final'] = layouts['raw']
        return layouts

    def deserialize(self, data, layouts):
        return data[0::2,...] + 1j * data[1::2,...]

    def seralize(self, data, layouts):
        data_ = np.zeros(layouts['shape_storage'], dtype=self.numpy_dtype, order='F')
        data_[0,...] = data.real
        data_[1, ...] = data.imag
        return data_


class SchemaSer(Schema):

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

    def deserialize(self, data):
        data = data[0::2] + 1j * data[1::2]
        data = np.reshape(data, self.layouts['raw'], order='F')
        return data

    def serialize(self, data):
        raise NotImplemented


class Schema2dseq(Schema):
    """
    Schema2dseq class

    - vector: data vector as obtained from binary file
    - frames: individual frames combined in
    - framegroups: aldasdasd

    """

    @property
    def layouts(self):
        return {
            "shape_fg" : self._dataset.shape_fg,
            "shape_frames" : self._dataset.shape_frames,
            "shape_block" : self._dataset.shape_block,
            "shape_storage" : self._dataset.shape_storage,
            "shape_final": self._dataset.shape_final
        }

    def get_rel_fg_index(self, fg_type):
        try:
            return self.fg_list.index(fg_type)
        except:
            raise KeyError('Framegroup {} not found in fg_list'.format(fg_type))

    def scale(self):
        self._dataset.data = np.reshape(self._dataset.data, self._dataset.shape_storage, order='F')
        self._dataset.data = self._scale_frames(self._dataset.data, self.layouts, 'FW')
        self._dataset.data = np.reshape(self._dataset.data, self._dataset.shape_final, order='F')

    def deserialize(self, data, layouts):

        if self._dataset._kwargs.get('scale') is None:
            scale = True
        else:
            scale = self._dataset._kwargs.get('scale')
        # scale
        if scale:
            data = self._scale_frames(data, layouts, 'FW')

        # frames -> frame_groups
        data = self._frames_to_framegroups(data, layouts)

        return data

    def _scale_frames(self, data, layouts, dir):
        """

        :param data:
        :param layouts:
        :param dir:
        :return:
        """

        # dataset is created with scale state set to False
        if self._dataset._kwargs.get('scale') is False:
            return data

        # get a float copy of the data array
        data = data.astype(np.float)

        slope = self._dataset.slope if not 'mask' in layouts.keys() else self._dataset.slope[layouts['mask'].flatten(order='F')]
        offset = self._dataset.offset if not 'mask' in layouts.keys() else self._dataset.offset[layouts['mask'].flatten(order='F')]

        for frame in range(data.shape[-1]):
            if dir == 'FW':
                data[..., frame] *= float(slope[frame])
                data[..., frame] += float(offset[frame])
            elif dir == 'BW':
                data[..., frame] /= float(slope[frame])
                data[..., frame] -= float(offset[frame])

        if dir == 'BW':
            data = np.round(data)

        return data

    def _frames_to_framegroups(self, data, layouts, mask=None):
        """

        :param data:
        :param layouts:
        :param mask:
        :return:
        """
        if mask:
            return np.reshape(data, (-1,) + layouts['shape_fg'], order='F')
        else:
            return np.reshape(data, layouts['shape_final'], order='F')

    def serialize(self, data, layout):
        data = self._framegroups_to_frames(data, layout)
        data = self._scale_frames(data, layout, 'BW')
        return data

    def _frames_to_vector(self, data):
        return data.flatten(order='F')

    def _framegroups_to_frames(self, data, layouts):
        if layouts.get('mask'):
            return np.reshape(data, (-1,) + layouts['shape_fg'], order='F')
        else:
            return np.reshape(data, layouts['shape_storage'], order='F')

    """
    Random access
    """
    def ra(self, slice_):
        """
        Random access to the data matrix
        :param slice_:
        :return:
        """

        layouts, layouts_ra = self._get_ra_layouts(slice_)

        array_ra = np.zeros(layouts_ra['shape_storage'], dtype=self.numpy_dtype)

        fp = np.memmap(self._dataset.path, dtype=self.numpy_dtype, mode='r',
                       shape=layouts['shape_storage'], order='F')

        for slice_ra, slice_full in self._generate_ra_indices(layouts_ra, layouts):
            array_ra[slice_ra] = np.array(fp[slice_full])

        array_ra = self.reshape_fw(array_ra, layouts_ra)

        singletons = tuple(i for i, v in enumerate(slice_) if isinstance(v, int))

        return np.squeeze(array_ra, axis=singletons)

    def _get_ra_layouts(self, slice_full):

        layouts = deepcopy(self.layouts)
        layouts_ra = deepcopy(layouts)

        layouts_ra['mask'] = np.zeros(layouts['shape_fg'], dtype=bool, order='F')
        layouts_ra['mask'][slice_full[self.encoded_dim:]] = True
        layouts_ra['shape_fg'], layouts_ra['offset_fg'] = self._get_ra_shape(layouts_ra['mask'])
        layouts_ra['shape_frames'] = (np.prod(layouts_ra['shape_fg'],dtype=int),)
        layouts_ra['shape_storage'] = layouts_ra['shape_block'] + layouts_ra['shape_frames']
        layouts_ra['shape_final'] = layouts_ra['shape_block'] + layouts_ra['shape_fg']

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

        for index_ra in np.ndindex(layouts_ra['shape_final'][self.encoded_dim:]):
            index = tuple(np.array(index_ra) + layouts_ra['offset_fg'])
            index = tuple(0 for i in range(self.encoded_dim)) + index
            index_ra = tuple(0 for i in range(self.encoded_dim)) + index_ra

            index_ra = np.ravel_multi_index(index_ra, layouts_ra['shape_final'], order='F')
            index = np.ravel_multi_index(index, layouts['shape_final'], order='F')

            index_ra = np.unravel_index(index_ra, layouts_ra['shape_storage'], order='F')
            index = np.unravel_index(index, layouts['shape_storage'], order='F')

            slice_ra = tuple(slice(None) for i in range(self.encoded_dim)) + index_ra[self.encoded_dim:]
            slice_full = tuple(slice(None) for i in range(self.encoded_dim)) + index[self.encoded_dim:]
            yield slice_ra, slice_full
