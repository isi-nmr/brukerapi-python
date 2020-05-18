import numpy as np
from .utils import index_to_slice
from .jcampdx import GenericParameter

def split_VisuFGOrderDescDim(visu_pars):
    VisuFGOrderDescDim = visu_pars.get_int('VisuFGOrderDescDim')
    VisuFGOrderDescDim -= 1
    visu_pars.set_int('VisuFGOrderDescDim', VisuFGOrderDescDim)
    return visu_pars

def split_VisuFGOrderDesc(visu_pars, fg):
    VisuFGOrderDesc = visu_pars.get_nested_list('VisuFGOrderDesc')
    for fg_  in VisuFGOrderDesc:
        if fg_[1] == '<{}>'.format(fg):
            VisuFGOrderDesc.remove(fg_)
    visu_pars.set_nested_list('VisuFGOrderDesc',VisuFGOrderDesc)
    return visu_pars

def split_VisuCoreFrameCount(visu_pars, scheme, frame_count, fg_ind_abs):
    VisuCoreFrameCount = visu_pars.get_parameter('VisuCoreFrameCount')

    layout = np.array(scheme.layouts['frame_groups'])
    layout[0:scheme.encoded_dim] = 1
    layout[fg_ind_abs] = 1
    frames = int(frame_count * np.prod(layout))

    VisuCoreFrameCount.value = frames


def split_VisuCoreDataSlope(visu_pars, scheme, index, fg_index):
    VisuCoreDataSlope = visu_pars.get_parameter('VisuCoreDataSlope')
    value = np.reshape(VisuCoreDataSlope.value, scheme.layouts['frame_groups'][scheme.encoded_dim:])
    slc = index_to_slice(index, value.shape, fg_index - scheme.encoded_dim)
    value = value[slc]
    VisuCoreDataSlope.size = (np.prod(value.shape),)
    VisuCoreDataSlope.value = value.flatten()

def split_VisuCoreDataOffs(visu_pars, scheme, index, fg_index):
    VisuCoreDataOffs = visu_pars.get_parameter('VisuCoreDataOffs')
    value = np.reshape(VisuCoreDataOffs.value, scheme.layouts['frame_groups'][scheme.encoded_dim:])
    slc = index_to_slice(index, value.shape, fg_index - scheme.encoded_dim)
    value = value[slc]
    VisuCoreDataOffs.size = (np.prod(value.shape),)
    VisuCoreDataOffs.value = value.flatten()

def split_VisuCoreDataMin(visu_pars, scheme, index, fg_index):
    VisuCoreDataMin = visu_pars.get_parameter('VisuCoreDataMin')
    value = np.reshape(VisuCoreDataMin.value, scheme.layouts['frame_groups'][scheme.encoded_dim:])
    slc = index_to_slice(index, value.shape, fg_index-scheme.encoded_dim)
    value = value[slc]
    VisuCoreDataMin.size = (np.prod(value.shape),)
    VisuCoreDataMin.value = value.flatten()

def split_VisuCoreDataMax(visu_pars, scheme, index, fg_index):
    VisuCoreDataMax = visu_pars.get_parameter('VisuCoreDataMax')
    value = np.reshape(VisuCoreDataMax.value, scheme.layouts['frame_groups'][scheme.encoded_dim:])
    slc = index_to_slice(index, value.shape, fg_index-scheme.encoded_dim)
    value = value[slc]
    VisuCoreDataMax.size = (np.prod(value.shape),)
    VisuCoreDataMax.value = value.flatten()

def split_VisuCoreDataUnits(visu_pars, fg_scheme, index, fg_index):

    parameter = visu_pars.get_parameter('VisuCoreDataUnits')

    value = parameter.value

    value = value[index]
    size = (1,)

    parameter.value = value
    parameter.size = size

    visu_pars.set_parameter('VisuCoreDataUnits',parameter)

    return visu_pars

def split_VisuFGElemComment(visu_pars, fg_scheme, index, fg_index):

    parameter = visu_pars.get_parameter('VisuFGElemComment')

    value = parameter.value

    value = value[index]
    size = (1,)

    parameter.value = value
    parameter.size = size

    visu_pars.set_parameter('VisuFGElemComment',parameter)

    return visu_pars

def split_VisuCoreOrientation(visu_pars, range):
    VisuCoreOrientation = visu_pars.get_parameter('VisuCoreOrientation')
    VisuCoreOrientation.value = VisuCoreOrientation.value[range[0]:range[1]].flatten(order='C')
    size = list(VisuCoreOrientation.size)
    size[0] = range[1] - range[0]
    VisuCoreOrientation.size = tuple(size)

def split_VisuCorePosition(visu_pars, range):
    VisuCorePosition = visu_pars.get_parameter('VisuCorePosition')
    VisuCorePosition.value = VisuCorePosition.value[range[0]:range[1]].flatten(order='C')
    size = list(VisuCorePosition.size)
    size[0] = range[1]-range[0]
    VisuCorePosition.size =tuple(size)

def split_VisuFGOrderDesc(visu_pars, fg_rel_ind, frame_count):
    #todo works for slice packages only
    VisuFGOrderDesc = visu_pars.get_parameter('VisuFGOrderDesc')
    value = VisuFGOrderDesc.value

    if isinstance(value[fg_rel_ind], list):
        value[fg_rel_ind][0] = frame_count
    else:
        value[0] = frame_count

    VisuFGOrderDesc.value = value

