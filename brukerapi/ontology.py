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

def split_VisuCoreFrameCount(visu_pars, fg_size):
    VisuCoreFrameCount = visu_pars.get_int('VisuCoreFrameCount')
    VisuCoreFrameCount /= fg_size
    visu_pars.set_int('VisuCoreFrameCount', VisuCoreFrameCount)
    return visu_pars

def split_VisuCoreDataSlope(visu_pars, fg_scheme, index, fg_index):
    parameter = visu_pars.get_parameter('VisuCoreDataSlope')
    value = np.reshape(parameter.value, fg_scheme.layout_shapes['frame_groups'][fg_scheme.core_dim:])
    slc = index_to_slice(index, value.shape, fg_index-fg_scheme.core_dim)
    value = value[slc]
    parameter.size = (np.prod(value.shape),)
    parameter.value = value.flatten()
    visu_pars.set_parameter('VisuCoreDataSlope',parameter)
    return visu_pars

def split_VisuCoreDataOffs(visu_pars, fg_scheme, index, fg_index):
    parameter = visu_pars.get_parameter('VisuCoreDataOffs')
    value = np.reshape(parameter.value, fg_scheme.layout_shapes['frame_groups'][fg_scheme.core_dim:])
    slc = index_to_slice(index, value.shape, fg_index-fg_scheme.core_dim)
    value = value[slc]
    parameter.size = (np.prod(value.shape),)
    parameter.value = value.flatten()
    visu_pars.set_parameter('VisuCoreDataOffs',parameter)
    return visu_pars

def split_VisuCoreDataMin(visu_pars, fg_scheme, index, fg_index):
    parameter = visu_pars.get_parameter('VisuCoreDataMin')
    value = np.reshape(parameter.value, fg_scheme.layout_shapes['frame_groups'][fg_scheme.core_dim:])
    slc = index_to_slice(index, value.shape, fg_index-fg_scheme.core_dim)
    value = value[slc]
    parameter.size = (np.prod(value.shape),)
    parameter.value = value.flatten()
    visu_pars.set_parameter('VisuCoreDataMin',parameter)
    return visu_pars

def split_VisuCoreDataMax(visu_pars, fg_scheme, index, fg_index):
    parameter = visu_pars.get_parameter('VisuCoreDataMax')
    value = np.reshape(parameter.value, fg_scheme.layout_shapes['frame_groups'][fg_scheme.core_dim:])
    slc = index_to_slice(index, value.shape, fg_index-fg_scheme.core_dim)
    value = value[slc]
    parameter.size = (np.prod(value.shape),)
    parameter.value = value.flatten()
    visu_pars.set_parameter('VisuCoreDataMax',parameter)
    return visu_pars

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