from .utils import index_to_slice
from .dataset import Dataset

import numpy as np

class FrameGroupMerger:
    SUPPORTED_FG = ['FG_COMPLEX']

    @classmethod
    def merge(cls, dataset, fg):
        """Merge Bruker object along a dimension of specific frame group.
        Only the frame groups listed in SUPPORTED_FG can be used to split the object.

        Parameters
        ----------
        dataset - dataset to be merged

        """

        if "<{}>".format(fg) not in dataset.dim_type:
            raise ValueError(f'Dataset does not contain {fg} frame group')

        """
        CHECK if FG and index are valid
        """
        # absolute index of FG_SLICE among dimensions of the dataset
        fg_abs_index = dataset.dim_type.index("<{}>".format(fg))

        # index of FG_SLICE among frame group dimensions of the dataset
        fg_rel_index = fg_abs_index - dataset.encoded_dim

        # number of samples in self.fg dimension
        fg_size = dataset.shape[fg_abs_index]

        # merge the data array
        cls._merge_data(dataset, fg_abs_index)

        # merge the parameter values
        cls._merge_parameters(dataset, fg, fg_abs_index, fg_rel_index, fg_size)

        # reload properties based on new values of parameters
        dataset.reload_properties()

        return dataset


    @ classmethod
    def _merge_data(cls, dataset, fg_abs_index):
        """
        Merge the data array in-place

        :param dataset: Bruker dataset
        :param fg_abs_index: index of dimension of the data array to be merged
        :return:
        """
        slc_re = [slice(None, None, None) for _ in range(dataset.data.ndim)]
        slc_im = [slice(None, None, None) for _ in range(dataset.data.ndim)]
        slc_re[fg_abs_index] = 0
        slc_im[fg_abs_index] = 1

        dataset.data = dataset.data[tuple(slc_re)] + 1j * dataset.data[tuple(slc_im)]

    @classmethod
    def _merge_parameters(cls, dataset, fg, fg_abs_index, fg_rel_index, fg_size):
        """

        Merge values of individual parameters of the dataset

        :param dataset:
        :param fg:
        :param fg_abs_index:
        :param fg_rel_index:
        :param fg_size:
        :return:
        """
        cls._merge_VisuCoreFrameCount(dataset, fg_size)
        cls._merge_VisuFGOrderDescDim(dataset)
        cls._merge_VisuCoreFrameType(dataset)
        cls._merge_VisuFGOrderDesc(dataset, fg)
        cls._merge_VisuFGElemId(dataset)

    @classmethod
    def _merge_VisuCoreFrameCount(cls, dataset, fg_size):
        try:
            parameter = dataset['VisuCoreFrameCount']
        except KeyError:
            return
        new_value = int(parameter.value / fg_size)
        parameter.value = new_value

    @classmethod
    def _merge_VisuFGOrderDescDim(cls, dataset):
        try:
            parameter = dataset['VisuFGOrderDescDim']
        except KeyError:
            return
        new_value = parameter.value - 1

        if new_value > 1:
            parameter.value = new_value
        else:
            del dataset._parameters['visu_pars']['VisuFGOrderDescDim']

    @classmethod
    def _merge_VisuCoreFrameType(cls, dataset):
        try:
            parameter = dataset['VisuCoreFrameType']
        except KeyError:
            return
        parameter.value = 'COMPLEX_IMAGE'

    @classmethod
    def _merge_VisuFGOrderDesc(cls, dataset, fg):
        try:
            parameter = dataset['VisuFGOrderDesc']
        except KeyError:
            return

        size = parameter.size[0] - 1
        parameter.size = size

        value = parameter.nested
        for fg_ in value:
            if fg_[1] == '<{}>'.format(fg):
                value.remove(fg_)
        if value:
            parameter.value = value
        else:
            del dataset.parameters['visu_pars']['VisuFGOrderDesc']

    @classmethod
    def _merge_VisuFGElemId(cls, dataset):
        try:
            parameter = dataset['VisuFGElemId']
        except KeyError:
            return
        del dataset.parameters['visu_pars']['VisuFGElemId']

