from brukerapi.dataset import Dataset
from brukerapi.folders import Experiment, Folder, Processing, Study
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from brukerapi.splitters import *
from brukerapi.jcampdx import JCAMPDX
from brukerapi.utils import simple_reconstruction
from brukerapi.splitters import SlicePackageSplitter
import json
# path to data cloned /

# study = Study('/home/tomas/data/bruker2nifti_qa/raw/Cyceron_DWI/20170719_075627_Lego_1_1')
#
# d = study.get_dataset(exp_id='2', proc_id='1')
#
# with d(scale=True) as reco:
#     print(d.data[0,0,0,0])

# folder = Study('/home/tomas/data/20200612_094625_lego_phantom_3_1_2')
# data_path = Path('/home/tomas/data/bruker2nifti_qa/raw/McGill_Orientation/bb20130412_APM_DEV_Orient.jl1')
# # folder = Folder(data_path)
# folder.print(recursive=True)
# folder.filter(parameter='PULPROG', operator='==', value='<RARE.ppg>')
# print('################')
# folder.print(recursive=True)
# folder.filter(type=Dataset)
# print('################')
# folder.print(recursive=True)

# d = JCAMPDX('/home/tomas/data/bruker2nifti_qa/raw/Cyceron_DWI/20170719_075627_Lego_1_1/1/acqp')
# d.get_value('ACQ_jobs')
# data_path = Path('/home/tomas/data')

# PV360
# data_path = Path('/home/tomas/data/pv360/')
# config_path = Path('test-pv360.json')

# with open(data_path / config_path) as json_file:
#     reference = json.load(json_file)['test_read_write']
# d = Dataset(data_path / Path(reference['2DSEQ_EPI']['path']), scale=False)
d = Dataset('/home/tomas/data/20200612_094625_lego_phantom_3_1_2/35/fid', add_parameters=['subject'])

print(d.id)
# print(d.dwell_s)

# method = d.parameters['method']

# print(d['PVM_VoxelGeoCub'].affine)


# print(d.scheme.layouts)
# print(d.dim_type)
# d.write('tmp/fid')
# d2 = Dataset('tmp/fid')
#
#
# print(d.shape)
# print(d2.shape)
# s = d.data[:,:,1,0,0]
# s2 = d2.data[:,:,1,0,0]
# frame = s[:,:]
# frame2 = s2[:,:]
#
# import matplotlib.pyplot as plt
# plt.subplot(1,2,1)
# plt.imshow(np.abs(d.data[:,:,0,0]))
# plt.subplot(1,2,2)
# plt.imshow(np.abs(d2.data[:,:,1,0,0]))
# plt.show()
#
# # both constructors are possible
# # EPI trim
# # d = Dataset(data_path / Path('raw/Cyceron_DWI/20170719_075627_Lego_1_1/1/fid'))
#
# # EPI no trim
# # d = Dataset('C:/data/ED_14Jul08_VEH_112HLHRO_rat.qE1/5/fid')
#
# # EPI
# # d = Dataset('C:/data/20200418_124909_lego_phantom_3_1_1/17/fid')
#
# # d=Dataset('C:/data/20180514_114437_lego_phantom_1_11/8/fid')
#
# UTE
# d = Dataset('C:/data/20200418_124909_lego_phantom_3_1_1/13/fid')

# plt.show()

#
# dataset = Dataset('C:/data/20200418_124909_lego_phantom_3_1_1/17/fid')
# dataset = Dataset('C:/data/ED_14Jul007_VEH_103PB_rat.qD1/1')
# dataset = Dataset('C:/data/ED_14Jul08_VEH_112HLHRO_rat.qE1/5/fid')
# d = Dataset('C:/data/bruker2nifti_qa/raw/Cyceron_MultiEcho/20170720_080545_Lego_1_2/2/pdata/1/2dseq')
# d = Dataset('C:/data/20200418_124909_lego_phantom_3_1_1/11/pdata/2/2dseq', random_access=True)
# d2 = Dataset('C:/data/20200418_124909_lego_phantom_3_1_1/11/pdata/2/2dseq')

# d = Dataset('C:/data/Cristina/20200313_112746_phantom_rat_13032020_phantom_rat_test_13032020_1_1_Jana/35/rawdata.Navigator')
# d.write('tmp/2dseq')
# d2 = Dataset('tmp/2dseq')
#
# print(np.array_equal(d.data, d2.data))
#

# r= simple_reconstruction(d)
#
# print(r.shape)
#
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(np.abs(d.data[:, :, 0, 0, 0]))
# plt.subplot(1,2,2)
# plt.imshow(np.abs(r[:, :, 0, 0, 0]))
# plt.show()

# datasets = FrameGroupSplitter('FG_ISA').split(dataset, write=True)

# dataset = Dataset('C:/data/bruker2nifti_qa/raw/Cyceron_DWI/20170719_075627_Lego_1_1/1/pdata/1')
#
# SlicePackageSplitter().split(dataset, write=True)

