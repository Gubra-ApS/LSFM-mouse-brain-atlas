## Import libraries
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
import scipy.ndimage.interpolation as sci_int
import scipy.ndimage.morphology as sci_morph

from skimage import data, color, img_as_uint
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imshow
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from skimage import segmentation
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage import morphology
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.restoration import (denoise_tv_chambolle)

import sys

from subprocess import call, run
import subprocess

from math import sqrt

import SimpleITK as sitk

from shutil import move

import math

import seaborn as sns

from scipy import stats, signal

#### Import custom libraries
import sys
sys.path.insert(0, 'INSERT FOLDER WHERE toolbox.py IS SAVED')

import importlib
import toolbox as bf
importlib.reload(bf)
####

# Paths
study_folder_server_avg = r'INSERT PATH TO OUTPUT FOLDER WHERE THE AVERAGE IMAGES WILL BE SAVED'

##### COMBINE REGULAR VOLUME AND THE TOP OF THE CORTEX #####
avg_normal_regi = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/A5_sym_aff_man.nii.gz'))
avg_cortex_regi = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex.nii.gz'))

# add padding to make room for cortex
avg_pad = np.zeros((avg_normal_regi.shape[0]+40,avg_normal_regi.shape[1], avg_normal_regi.shape[2]),'uint16')
avg_pad[0:avg_pad.shape[0]-40,:,:] = avg_normal_regi

bf.saveNifti(avg_pad,os.path.join(study_folder_server_avg,'nifti/A5_sym_aff_man_pad.nii.gz'))

# Crop box for rigid
avg_pad = avg_pad[178:260,:,:]
bf.saveNifti(avg_pad,os.path.join(study_folder_server_avg,'nifti/avg_gubra_normal_regi.nii.gz'))

avg_cortex_regi = avg_cortex_regi[178:260,:,:]
bf.saveNifti(avg_pad,os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex_regi.nii.gz'))

### ATLAS REGISTRATION
moving = os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex_regi.nii.gz')
fixed = os.path.join(study_folder_server_avg,'nifti/avg_gubra_normal_regi.nii.gz')
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = study_folder_server_avg+r'/nifti'
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

# rigid Registration
#def registration(params, result_name, init_trans=r'', f_mask=r'', save_nifti=True, datatype='uint16'):
hu.registration('Par_rigid.txt', 'avg_gubra_cortex_rig', datatype='uint8')

# Dummy to get tranformation parameters
moving =  os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex.nii.gz')
fixed =  os.path.join(study_folder_server_avg,'nifti/A5_sym_aff_man_pad.nii.gz')
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

# Dummy rigid registration
hu.registration('Par_rigid.txt', 'avg_gubra_cortex_rig_man', datatype='uint8')

# Apply after manually changing transformation parameters of the whole volume to the ones obtained from registration of overlapping regions
hu.transform_vol(moving, 'avg_gubra_cortex_rig_man', 'avg_gubra_cortex_rig_man2', type='vol')


# Bspline registration
normal_pad = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/A5_sym_aff_man_pad.nii.gz'))
mask = np.zeros(normal_pad.shape,'uint8')
mask[178:319,:,:] = 1
bf.saveNifti(mask, os.path.join(study_folder_server_avg,'nifti/A5_sym_bspline_man_pad_mask.nii.gz'))
fmask =os.path.join(study_folder_server_avg,'nifti/A5_sym_bspline_man_pad_mask.nii.gz')

moving = os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex_rig_man2.nii.gz')
fixed = os.path.join(study_folder_server_avg,'nifti/A5_sym_aff_man_pad.nii.gz')
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = study_folder_server_avg+r'/nifti'
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

hu.registration('Par0000bspline_soft.txt', 'avg_gubra_cortex_bspline', f_mask=fmask, datatype='uint8')


### Intensity matching
print('Performing intensity matching')
# Create overlap volumes
overlap_row_begin = 200
overlap_row_end = 250

cortex_bspline = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex_bspline.nii.gz'))
avg_normal = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/A5_sym_aff_man_pad.nii.gz'))

cortex_bspline_overlap = cortex_bspline[overlap_row_begin:overlap_row_end,324:491,:]
avg_normal_overlap = avg_normal[overlap_row_begin:overlap_row_end,324:491,:]

# Compute parameters for linear matching function
ctx_int=np.ndarray.flatten(cortex_bspline_overlap)
norm_int=np.ndarray.flatten(avg_normal_overlap)

# Fit the data ctx_int=slope*norm_int+intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(ctx_int,norm_int)
print('Slope')
print(slope)
print('Intercept')
print(intercept)

# apply to full cortex_spline volume
cortex_bspline_matched = np.copy(cortex_bspline)

for i in range(0,cortex_bspline.shape[0]):
    print(i)
    for j in range(0,cortex_bspline.shape[1]):
        for k in range(0,cortex_bspline.shape[2]):
            if cortex_bspline[i,j,k]!=0:
                cortex_bspline_matched[i,j,k]=slope*cortex_bspline[i,j,k]+intercept

# Save intensity matched image
final_sitk = sitk.GetImageFromArray(cortex_bspline_matched)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex_bspline_matched.nii.gz'))


### Blending of images
avg_normal = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/A5_sym_aff_man_pad.nii.gz'))
cortex_bspline_matched = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/avg_gubra_cortex_bspline_matched.nii.gz'))

blend_end = 250
blend_begin = blend_end-24


print('Blending images')
def sigmoid(x):
  y = np.zeros(len(x))
  for i in range(len(x)):
    y[i] = 1 / (1 + math.exp(-x[i]))
  return y

sigmoid_ = sigmoid(np.arange(-3, 3, 1/4))
alpha = np.repeat(sigmoid_.reshape((len(sigmoid_), 1)), repeats=(491), axis=1)

blended = np.zeros(cortex_bspline_matched[blend_begin:blend_end,:,:].shape)

# loop dimension
for i in range(blended.shape[2]):
    image2_connect = cortex_bspline_matched[blend_begin:blend_end,:,i]
    image1_connect = avg_normal[blend_begin:blend_end,:,i]
    blended[:,:,i] = image1_connect * (1.0 - alpha) + image2_connect * alpha

#save final volume
final = np.copy(avg_normal)

final[blend_begin:blend_end,:,:] = blended

final[blend_end:319,:,:] = cortex_bspline_matched[blend_end:319,:,:]

#Save intensity matched image
final_sitk = sitk.GetImageFromArray(final)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server_avg,'nifti/avg_gubra_final.nii.gz'))






##### COMBINE REGULAR + CORTEX VOLUME AND THEOLFACTORY BULB #####
avg_normal_regi = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/avg_gubra_final.nii.gz'))
avg_olf_regi = bf.readNifti(os.path.join(study_folder_server_avg,'nifti//avg_gubra_olf.nii.gz'))

# add padding to make room for olfactory bulb
avg_pad = np.zeros((avg_normal_regi.shape[0],avg_normal_regi.shape[1]+155, avg_normal_regi.shape[2]),'uint16')
avg_pad[:,155:avg_pad.shape[1],:] = avg_normal_regi

bf.saveNifti(avg_pad,os.path.join(study_folder_server_avg,'nifti/avg_gubra_final_pad.nii.gz'))


# Add padding to olfactory bulb volume
avg_olf_regi=avg_olf_regi[27:,0:646,:]
print(avg_olf_regi.shape)
olf_pad = np.zeros((avg_olf_regi.shape[0]+39,avg_olf_regi.shape[1], avg_olf_regi.shape[2]),'uint16')
print(olf_pad.shape)
olf_pad[39:olf_pad.shape[0],:,:] = avg_olf_regi

bf.saveNifti(olf_pad,os.path.join(study_folder_server_avg,'nifti/avg_gubra_olf_pad.nii.gz'))

# # Crop box for rigid/affine align
avg_normal_regi = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/avg_gubra_final_pad.nii.gz'))
avg_olf_regi = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/avg_gubra_olf_pad.nii.gz'))

avg_normal_regi = avg_normal_regi[:,255:355,:]
bf.saveNifti(avg_normal_regi,os.path.join(study_folder_server_avg,'nifti/avg_gubra_final_pad_regi.nii.gz'))

avg_olf_regi = avg_olf_regi[:,255:355,:]
bf.saveNifti(avg_olf_regi,os.path.join(study_folder_server_avg,'nifti/avg_gubra_olf_pad_regi.nii.gz'))


# ATLAS REGISTRATION
moving = os.path.join(study_folder_server_avg,'nifti/avg_gubra_olf_pad_regi.nii.gz')
fixed = os.path.join(study_folder_server_avg,'nifti/avg_gubra_final_pad_regi.nii.gz')
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = study_folder_server_avg+r'/nifti'
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

# rigid Registration
hu.registration('Par_rigid_olf.txt', 'avg_gubra_olf_rig', datatype='uint8')

## Dummy to get tranformation parameters
moving = os.path.join(study_folder_server_avg,'nifti/avg_gubra_olf_pad.nii.gz')
fixed = os.path.join(study_folder_server_avg,'nifti/avg_gubra_final_pad.nii.gz')
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = study_folder_server_avg+r'/nifti'
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

# Dummy rigid registration
hu.registration('Par_rigid_olf.txt', 'avg_gubra_olf_rig_man', datatype='uint8')

# Apply after manually changing transformation parameters of the whole volume to the ones obtained from registration of overlapping regions
hu.transform_vol(moving, 'avg_gubra_olf_rig_man', 'avg_gubra_olf_rig_man2', type='vol')


# Bspline registration
normal_pad = bf.readNifti(os.path.join(study_folder_server,'nifti/avg_gubra_final_pad.nii.gz'))
mask = np.zeros(normal_pad.shape,'uint8')
mask[:,168:340,:] = 1
bf.saveNifti(mask, os.path.join(study_folder_server,'nifti/avg_gubra_final_pad_mask.nii.gz'))

fmask = os.path.join(study_folder_server,'nifti/avg_gubra_final_pad_mask.nii.gz')
moving = os.path.join(study_folder_server,'nifti/avg_gubra_olf_rig_man2.nii.gz')
fixed = os.path.join(study_folder_server,'nifti/avg_gubra_final_pad.nii.gz')
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = study_folder_server_avg+r'/nifti'
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

hu.registration('Par0000bspline_soft_olf.txt', 'avg_gubra_olf_bspline', f_mask=fmask, datatype='uint8')


### Intensity matching
print('Performing intensity matching')

# Create overlap volumes
overlap_row_end = 186
overlap_row_begin = overlap_row_end-24
#
olf_bspline = bf.readNifti(os.path.join(study_folder_server,'nifti/avg_gubra_olf_bspline.nii.gz'))
avg_normal = bf.readNifti(os.path.join(study_folder_server,'nifti/avg_gubra_final_pad.nii.gz'))

olf_bspline_overlap = olf_bspline[:,overlap_row_begin:overlap_row_end,:]
avg_normal_overlap = avg_normal[:,overlap_row_begin:overlap_row_end,:]

bf.saveNifti(olf_bspline_overlap, os.path.join(study_folder_server,'nifti/avg_gubra_olf_bspline_overlap.nii.gz'))
bf.saveNifti(avg_normal_overlap, os.path.join(study_folder_server,'nifti/avg_gubra_final_pad_overlap.nii.gz'))

# Compute parameters for linear matching function
olf_int=np.ndarray.flatten(olf_bspline_overlap)
norm_int=np.ndarray.flatten(avg_normal_overlap)

# Fit the data ctx_int=slope*norm_int+intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(olf_int,norm_int)
print('Slope')
print(slope)
print('Intercept')
print(intercept)

# apply to full cortex_spline volume
olf_bspline_matched = np.copy(olf_bspline)

for i in range(0,olf_bspline.shape[0]):
    print(i)
    for j in range(0,olf_bspline.shape[1]):
        for k in range(0,olf_bspline.shape[2]):
            if olf_bspline[i,j,k]!=0:
                olf_bspline_matched[i,j,k]=slope*olf_bspline[i,j,k]+intercept

# Save intensity matched image
final_sitk = sitk.GetImageFromArray(olf_bspline_matched)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server,'nifti/avg_gubra_olf_bspline_matched.nii.gz'))

avg_normal = bf.readNifti(os.path.join(study_folder_server,'nifti/avg_gubra_final_pad.nii.gz'))
olf_bspline_matched = bf.readNifti(os.path.join(study_folder_server,'nifti/avg_gubra_olf_bspline_matched.nii.gz'))

# Blending of images
blend_end = 186
blend_begin = blend_end-24


print('Blending images')
def sigmoid(x):
  y = np.zeros(len(x))
  for i in range(len(x)):
    y[i] = 1 / (1 + math.exp(-x[i]))
  return y

sigmoid_ = sigmoid(np.arange(-3, 3, 1/4))
alpha = np.repeat(sigmoid_.reshape((len(sigmoid_), 1)), repeats=(491), axis=1)
print(alpha.shape)

blended = np.zeros(olf_bspline_matched[:,blend_begin:blend_end,:].shape)
print(blended.shape)

# loop dimension
for i in range(blended.shape[0]):
    image2_connect = gaussian_filter(olf_bspline_matched[i,blend_begin:blend_end,:], sigma=0.9)
    image1_connect = avg_normal[i,blend_begin:blend_end,:]
    blended[i,:,:] = image2_connect * (1.0 - alpha) + image1_connect * alpha

#save final volume
final = np.copy(avg_normal)

final[:,blend_begin:blend_end,:] = blended

final[:,0:blend_begin,:] = gaussian_filter(olf_bspline_matched[:,0:blend_begin,:],sigma=0.3)


#Save intensity matched image
final_sitk = sitk.GetImageFromArray(final)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server,'nifti/avg_gubra_final2.nii.gz'))
