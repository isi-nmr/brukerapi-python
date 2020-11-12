from .utils import index_to_slice
from .dataset import Dataset

import numpy as np
import copy
from pathlib import Path

SUPPORTED_FG = ['FG_ISA','FG_IRMODE','FG_ECHO']

class Splitter(object):

    def write(self, datasets, path_out=None):

        for dataset in datasets:
            if path_out:
                dataset.write('{}/{}/{}'.format(Path(path_out), dataset.path.parents[0].name, dataset.path.name))
            else:
                dataset.write(dataset.path)

    def _split_data(self, dataset, range, fg_abs_index):
        """
        Get slice of the data array correspondent to given slice package

        :param dataset:
        :param frame_range:
        :param fg_abs_index:
        :return:
        """
        return dataset.data[index_to_slice(range, dataset.shape, fg_abs_index)].copy()

    def _split_VisuCoreDataMin(self, dataset, visu_pars, select, fg_rel_index):
        """

        :param scheme:
        :param select: int, or slice
        :param fg_index:
        :return:
        """
        VisuCoreDataMin = visu_pars['VisuCoreDataMin']
        value = np.reshape(VisuCoreDataMin.value, dataset.shape_final[dataset.encoded_dim:], order='F')
        value = value[index_to_slice(select, value.shape, fg_rel_index)]
        VisuCoreDataMin.size = (np.prod(value.shape),)
        VisuCoreDataMin.value = value.flatten(order='F')

    def _split_VisuCoreDataMax(self, dataset, visu_pars, select, fg_rel_index):
        """

        :param scheme:
        :param select: int, or slice
        :param fg_index:
        :return:
        """
        VisuCoreDataMax = visu_pars['VisuCoreDataMax']
        value = np.reshape(VisuCoreDataMax.value, dataset.shape_final[dataset.encoded_dim:], order='F')
        value = value[index_to_slice(select, value.shape, fg_rel_index)]
        VisuCoreDataMax.size = (np.prod(value.shape),)
        VisuCoreDataMax.value = value.flatten(order='F')

    def _split_VisuCoreDataOffs(self, dataset, visu_pars, select, fg_rel_index):
        """

        :param scheme:
        :param select: int, or slice
        :param fg_index:
        :return:
        """
        VisuCoreDataOffs = visu_pars['VisuCoreDataOffs']
        value = np.reshape(VisuCoreDataOffs.value, dataset.shape_final[dataset.encoded_dim:],order='F')
        value = value[index_to_slice(select, value.shape, fg_rel_index)]
        VisuCoreDataOffs.size = (np.prod(value.shape),)
        VisuCoreDataOffs.value = value.flatten(order='F')

    def _split_VisuCoreDataSlope(self, dataset, visu_pars, select, fg_rel_index):
        """

        :param scheme:
        :param select: int, or slice
        :param fg_index:
        :return:
        """
        VisuCoreDataSlope = visu_pars['VisuCoreDataSlope']
        value = np.reshape(VisuCoreDataSlope.value, dataset.shape_final[dataset.encoded_dim:],order='F')
        value = value[index_to_slice(select, value.shape, fg_rel_index)]
        VisuCoreDataSlope.size = (np.prod(value.shape),)
        VisuCoreDataSlope.value = value.flatten(order='F')

    def _split_VisuCoreTransposition(self, dataset, visu_pars, index, fg_index):
        try:
            VisuCoreTransposition = visu_pars['VisuCoreTransposition']
        except KeyError:
            return VisuCoreTransposition

        value = np.reshape(VisuCoreTransposition.value, dataset.shape_final[dataset.encoded_dim:], order='F')
        value = value[index_to_slice(index, value.shape, fg_index - dataset.encoded_dim)]
        VisuCoreTransposition.size = (np.prod(value.shape),)
        VisuCoreTransposition.value = value.flatten(order='F')


class FrameGroupSplitter(Splitter):
    def __init__(self, fg):
        if fg not in SUPPORTED_FG:
            raise NotImplemented('Split operation for {} is not implemented'.format(fg))

        super(FrameGroupSplitter, self).__init__()
        self.fg = fg


    def split(self, dataset, select=None, write=False, path_out=None, **kwargs):
        """Split Bruker object along a dimension of specific frame group.
        Only the frame groups listed in SPLIT_FG_IMPLEMENTED can be used to split the object.

        Parameters
        ----------
        fg - list of fg's along which to split
        index - list of lists of indexes

        Returns
        -------
        List of objects incurred during splitting.

        """

        if "<{}>".format(self.fg) not in dataset.dim_type:
            raise ValueError(f'Dataset does not contain {self.fg} frame group')

        """
        CHECK if FG and index are valid
        """
        # absolute index of FG_SLICE among dimensions of the dataset
        fg_abs_index = dataset.dim_type.index("<{}>".format(self.fg))

        # index of FG_SLICE among frame group dimensions of the dataset
        fg_rel_index = fg_abs_index - dataset.encoded_dim

        # number of samples in self.fg dimension
        fg_size = dataset.shape[fg_abs_index]

        # If no index is specified, all elements of given dimension will be splited
        if select is None:
            select = list(range(0, fg_size))

        if isinstance(select, int):
            select = [select,]

        if max(select) >= fg_size:
            raise IndexError(f'Selection {select} out of bounds, size of {self.fg} dimension is {fg_size}')

        """
        PERFORM splitting
        """
        datasets = []

        for select_ in select:
            # construct a new Dataset, without loading data, the data will be supplied later
            name = '{}_{}_{}/2dseq'.format(dataset.path.parents[0].name, self.fg, select_)

            # construct a new Dataset, without loading data, the data will be supplied later
            dataset_ = Dataset(dataset.path.parents[1] / name, load=False)

            dataset_.parameters = self._split_params(dataset, select_, fg_abs_index, fg_rel_index, fg_size)

            # construct properties from the new set of parameters
            dataset_.load_properties()

            # construct schema
            dataset_.load_schema()

            # SPLIT data
            dataset_.data = self._split_data(dataset, select_, fg_abs_index)

            # append to result
            datasets.append(dataset_)

        if write:
            self.write(datasets, path_out=path_out)

        return datasets

    def _split_params(self, dataset, select, fg_abs_index, fg_rel_index, fg_size):

        visu_pars = copy.deepcopy(dataset.parameters['visu_pars'])

        self._split_VisuCoreFrameCount(visu_pars, fg_size)
        self._split_VisuFGOrderDescDim(visu_pars)
        self._split_VisuFGOrderDesc(visu_pars, self.fg)
        self._split_VisuCoreDataSlope(dataset, visu_pars, select, fg_rel_index)
        self._split_VisuCoreDataOffs(dataset, visu_pars, select, fg_rel_index)
        self._split_VisuCoreDataMin(dataset, visu_pars, select, fg_rel_index)
        self._split_VisuCoreDataMax(dataset, visu_pars, select, fg_rel_index)

        if self.fg == 'FG_ECHO':
            self._split_params_FG_ECHO(dataset, select, fg_abs_index, fg_rel_index, fg_size, visu_pars)
        if self.fg == 'FG_ISA':
            self._split_params_FG_ISA(dataset, select, fg_abs_index, fg_rel_index, fg_size, visu_pars)

        return {'visu_pars': visu_pars}

    def _split_params_FG_ISA(self, dataset, select, fg_abs_index, fg_rel_index, fg_size, visu_pars):
        self._split_VisuCoreDataUnits(visu_pars, dataset, select, fg_rel_index)
        self._split_VisuFGElemComment(visu_pars, dataset, select, fg_rel_index)

    def _split_params_FG_ECHO(self, dataset, select, fg_abs_index, fg_rel_index, fg_size, visu_pars):
        self._split_VisuAcqEchoTime(visu_pars, select)

    def _split_VisuCoreFrameCount(self, visu_pars, fg_size):
        VisuCoreFrameCount = visu_pars['VisuCoreFrameCount']
        value = int(VisuCoreFrameCount.value / fg_size)
        VisuCoreFrameCount.value = value

    def _split_VisuFGOrderDescDim(self, visu_pars):
        VisuFGOrderDescDim = visu_pars['VisuFGOrderDescDim']
        value = VisuFGOrderDescDim.value - 1

        if value > 1:
            VisuFGOrderDescDim.value = value
        else:
            del visu_pars['VisuFGOrderDescDim']

    def _split_VisuCoreDataUnits(self, visu_pars, fg_scheme, index, fg_index):
        VisuCoreDataUnits = visu_pars['VisuCoreDataUnits']
        value = VisuCoreDataUnits.value
        VisuCoreDataUnits.value = value[index]
        VisuCoreDataUnits.size = (65,)

    def _split_VisuFGOrderDesc(self, visu_pars, fg):
        VisuFGOrderDesc = visu_pars['VisuFGOrderDesc']

        size = VisuFGOrderDesc.size[0] - 1
        VisuFGOrderDesc.size = size

        value = VisuFGOrderDesc.nested
        for fg_ in value:
            if fg_[1] == '<{}>'.format(fg):
                value.remove(fg_)
        if value:
            VisuFGOrderDesc.value = value
        else:
            del visu_pars['VisuFGOrderDesc']

    def _split_VisuFGElemComment(self, visu_pars, fg_scheme, index, fg_index):

        VisuFGElemComment = visu_pars['VisuFGElemComment']

        value = VisuFGElemComment.value

        value = value[index]
        VisuFGElemComment.value = value
        VisuFGElemComment.size = (65,)

    def _split_VisuAcqEchoTime(self, visu_pars, select):
        VisuAcqEchoTime = visu_pars['VisuAcqEchoTime']
        value = VisuAcqEchoTime.value
        VisuAcqEchoTime.size=(1,)
        VisuAcqEchoTime.value = float(value[select])


class SlicePackageSplitter(Splitter):
    """
    Split 2dseq data set along individual slice packages
    """
    def split(self, dataset, write=False, path_out=None):
        """
        Split 2dseq data set containing multiple data sets into a list of 2dseq data sets containing individual slice packages.

        This functionality might be used for instance when converting data to a different data format.

        :param dataset: 2dseq dataset with multiple slice packages
        :param write: if True, splitted data sets will we writen to drive
        :param path_out: a path to store data sets (optional)
        :return: list of splitted data sets
        """

        try:
            VisuCoreSlicePacksSlices = dataset['VisuCoreSlicePacksSlices'].value
        except KeyError:
            print('Parameter VisuCoreSlicePacksSlices not found')

        # list of splitted data sets
        datasets = []

        # range of frames of given slice package
        frame_range = range(0,0)

        # absolute index of FG_SLICE among dimensions of the dataset
        fg_rel_index = dataset['VisuFGOrderDesc'].sub_list(1).index('<FG_SLICE>')

        # index of FG_SLICE among frame group dimensions of the dataset
        fg_abs_index = fg_rel_index + dataset.encoded_dim

        # slice package split loop
        for sp_index in range(len(VisuCoreSlicePacksSlices)):
            # set range
            frame_range= range(frame_range.stop, frame_range.stop + VisuCoreSlicePacksSlices[sp_index][1])

            # number of frames contained in given slice package
            frame_count = frame_range.stop - frame_range.start

            # name of the data set created by the split
            name = '{}_sp_{}/2dseq'.format(dataset.path.parents[0].name, sp_index)

            # construct a new Dataset, without loading data, the data will be supplied later
            dataset_ = Dataset(dataset.path.parents[1] / name, load=False)

            # SPLIT parameteres
            dataset_.parameters = self._split_parameters(dataset, frame_range, fg_rel_index, fg_abs_index, sp_index, frame_count)

            # construct properties from the new set of parameters
            dataset_.load_properties()

            # construct schema
            dataset_.load_schema()

            #SPLIT data
            dataset_.data = self._split_data(dataset, frame_range, fg_abs_index)

            # append to result
            datasets.append(dataset_)

        if write:
            self.write(datasets, path_out=path_out)

        return datasets

    def _split_parameters(self, dataset, frame_range, fg_rel_index, fg_abs_index, sp_index, frame_count):
        # create a copy of visu_pars of the original data set
        visu_pars_ = copy.deepcopy(dataset.parameters['visu_pars'])

        # modify individual parameters so that the resulting data set is consistent
        self._split_VisuCorePosition(visu_pars_, frame_range, frame_count)
        self._split_VisuCoreOrientation(visu_pars_, frame_range, frame_count)
        self._split_VisuCoreDataMin(dataset, visu_pars_, frame_range, fg_rel_index)
        self._split_VisuCoreDataMax(dataset, visu_pars_, frame_range, fg_rel_index)
        self._split_VisuCoreDataOffs(dataset, visu_pars_, frame_range, fg_rel_index)
        self._split_VisuCoreDataSlope(dataset, visu_pars_, frame_range, fg_rel_index)
        self._split_VisuCoreTransposition(dataset, visu_pars_, frame_range, fg_rel_index)
        self._split_VisuCoreFrameCount(dataset, visu_pars_, frame_count, fg_abs_index)
        self._split_VisuFGOrderDesc(visu_pars_, fg_rel_index, frame_count)
        self._split_VisuCoreSlicePacksDef(visu_pars_)
        self._split_VisuCoreSlicePacksSlices(visu_pars_, sp_index)
        self._split_VisuCoreSlicePacksSliceDist(visu_pars_, sp_index)

        return {"visu_pars":visu_pars_}

    def _split_VisuCoreFrameCount(self, dataset, visu_pars, frame_count, fg_ind_abs):
        VisuCoreFrameCount = visu_pars['VisuCoreFrameCount']

        layout = np.array(dataset.shape_final)
        layout[0:dataset.encoded_dim] = 1
        layout[fg_ind_abs] = 1
        frames = int(frame_count * np.prod(layout))

        VisuCoreFrameCount.value = frames

    def _split_VisuCoreOrientation(self, visu_pars, frame_range, frame_count):
        VisuCoreOrientation = visu_pars['VisuCoreOrientation']
        VisuCoreOrientation.value = VisuCoreOrientation.value[frame_range, :].flatten(order='C')
        VisuCoreOrientation.size = (frame_count, 9)

    def _split_VisuCorePosition(self, visu_pars, frame_range, frame_count):
        VisuCorePosition = visu_pars['VisuCorePosition']
        VisuCorePosition.value = VisuCorePosition.value[frame_range, :].flatten(order='C')
        VisuCorePosition.size = (frame_count, 3)

    def _split_VisuFGOrderDesc(self, visu_pars, fg_rel_ind, frame_count):
        VisuFGOrderDesc = visu_pars['VisuFGOrderDesc']
        value = VisuFGOrderDesc.value

        if isinstance(value[fg_rel_ind], list):
            value[fg_rel_ind][0] = frame_count
        else:
            value[0] = frame_count

        VisuFGOrderDesc.value = value

    def _split_VisuCoreSlicePacksDef(self,visu_pars):
        VisuCoreSlicePacksDef = visu_pars['VisuCoreSlicePacksDef']
        value = VisuCoreSlicePacksDef.value
        value[1] = 1
        VisuCoreSlicePacksDef.value = value

    def _split_VisuCoreSlicePacksSlices(self, visu_pars_, sp_index):
        VisuCoreSlicePacksSlices = visu_pars_['VisuCoreSlicePacksSlices']
        VisuCoreSlicePacksSlices.value = [VisuCoreSlicePacksSlices.value[sp_index]]

    def _split_VisuCoreSlicePacksSliceDist(self, visu_pars_, sp_index):
        VisuCoreSlicePacksSliceDist = visu_pars_['VisuCoreSlicePacksSliceDist']
        value = int(VisuCoreSlicePacksSliceDist.value[sp_index])
        VisuCoreSlicePacksSliceDist.value = value
        VisuCoreSlicePacksSliceDist.size = 1