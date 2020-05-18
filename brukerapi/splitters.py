from .utils import index_to_slice
from .dataset import Dataset
from .ontology import *

import numpy as np
import copy

SUPPORTED_FG = ['FG_ISA','FG_IRMODE']

class Splitter(object):
    pass


class FrameGroupSplitter(Splitter):
    def __init__(self, fg):
        if fg not in SUPPORTED_FG:
            raise NotImplemented('Split operation for {} is not implemented'.format(fg))

        super(FrameGroupSplitter, self).__init__()
        self.fg = fg

    def split(self, dataset, index=None, write=False, path_out=None, **kwargs):
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
        """
        CHECK if FG and index are valid
        """
        # If no index is specified, all elements of given dimension will be splited
        if index is None:
            index = []

        if isinstance(index, int):
            index = [index, ]

        if not self.fg in dataset.fg_scheme.dim_type:
            raise ValueError(f'Dataset does not contain {self.fg} frame group')

        fg_index = dataset.dim_type.index(self.fg)

        if not index:
            index = list(range(dataset.fg_scheme.dim_size(index=fg_index)))  # all items of given dimension

        if max(index) >= dataset.dim_size[fg_index]:
            raise IndexError(f'Index {index} out of bounds, size of {self.fg} dimension is {dataset.dim_size(index=fg_index)}')

        """
        PERFORM splitting
        """
        data = dataset.data
        visu_pars = dataset.parameters['visu_pars']
        fg_scheme = dataset.fg_scheme
        fg_ind = fg_scheme.dim_type.index(self.fg)

        datasets = []

        for index_ in index:
            if write and path_out:
                path = '{}_{}_{}/2dseq'.format(path_out, self.fg, index_)
            else:
                path = '{}_{}_{}/2dseq'.format(dataset.parent, self.fg, index_)

            # split visu_pars
            if self.fg == 'FG_ISA':
                visu_pars_ = self.split_params_FG_ISA(copy.deepcopy(visu_pars), fg_scheme, index_, fg_index)

            # split data array
            slc = index_to_slice(index_, data.shape, fg_ind)
            dataset_ = Dataset(path=path, load=False)
            dataset_.parameters = {'visu_pars': visu_pars_}
            dataset_.data = dataset.get_data(slc=slc, copy=True)
            dataset_.load_fg_scheme()
            datasets.append(dataset)

            if write:
                dataset_.write(dataset_.path, **kwargs)

        return datasets

    def split_params_FG_ISA(self, visu_pars, fg_scheme, index, fg_ind):

        visu_pars = split_VisuFGOrderDescDim(visu_pars)
        visu_pars = split_VisuFGOrderDesc(visu_pars, self.fg)
        visu_pars = split_VisuCoreFrameCount(visu_pars, fg_scheme.layout_shapes['frame_groups'][fg_ind])
        visu_pars = split_VisuCoreDataSlope(visu_pars, fg_scheme, index, fg_ind)
        visu_pars = split_VisuCoreDataOffs(visu_pars, fg_scheme, index, fg_ind)
        visu_pars = split_VisuCoreDataMin(visu_pars, fg_scheme, index, fg_ind)
        visu_pars = split_VisuCoreDataMax(visu_pars, fg_scheme, index, fg_ind)
        visu_pars = split_VisuCoreDataUnits(visu_pars, fg_scheme, index, fg_ind)
        visu_pars = split_VisuFGElemComment(visu_pars, fg_scheme, index, fg_ind)

        return visu_pars

    def split_params_FG_IRMODE(self, visu_pars):
        pass


class SlicePackageSplitter(Splitter):

    def split(self, dataset):

        try:
            VisuCoreSlicePacksSlices = dataset.get_list('VisuCoreSlicePacksSlices')
        except KeyError:
            print('Parameter VisuCoreSlicePacksSlices not found')

        datasets = []
        first_last = [0,0]



        fg_abs_index = dataset.dim_type.index('FG_SLICE')
        fg_rel_index = fg_abs_index - dataset.scheme.encoded_dim

        for sp_index in range(len(VisuCoreSlicePacksSlices)):
            """
            SPLIT data
            """
            slice_package = VisuCoreSlicePacksSlices[sp_index]

            # getting range in the slice framegroup
            first_last[0] = first_last[1]
            first_last[1] += slice_package[1]

            data_slices = []

            for dim_index in range(dataset.dim):
                if dim_index == fg_abs_index:
                    data_slices.append(slice(first_last[0], first_last[1]))
                else:
                    data_slices.append(slice(0, dataset.shape[dim_index]))

            data_ = np.squeeze(dataset.data[tuple(data_slices)]).copy()

            """
            SPLIT parameteres
            """
            visu_pars_ = copy.deepcopy(dataset.parameters['visu_pars'])

            """
            MODIFY relevant parameters
            """
            fg_ind = dataset.scheme.dim_type.index('FG_SLICE')
            frame_count = first_last[1] - first_last[0]
            frame_range = range(first_last[0],first_last[1])

            split_VisuCorePosition(visu_pars_, first_last)
            split_VisuCoreOrientation(visu_pars_, first_last)
            split_VisuCoreDataMin(visu_pars_, dataset.scheme, frame_range, fg_ind)
            split_VisuCoreDataMax(visu_pars_, dataset.scheme, frame_range, fg_ind)
            split_VisuCoreDataOffs(visu_pars_, dataset.scheme, frame_range, fg_ind)
            split_VisuCoreDataSlope(visu_pars_, dataset.scheme, frame_range, fg_ind)
            split_VisuCoreFrameCount(visu_pars_, dataset.scheme, frame_count, fg_ind)
            split_VisuFGOrderDesc(visu_pars_, fg_rel_index, frame_count)

            print(visu_pars_.get_value('VisuFGOrderDesc'))

            name = '{}_sp_{}/2dseq'.format(dataset.path.parents[0].name, sp_index)

            dataset_ = Dataset(dataset.path.parents[1] / name, load=False)
            dataset_.parameters = {'visu_pars':visu_pars_}
            dataset_.load_scheme()
            dataset_.data = data_

            datasets.append(dataset_)

        return datasets
