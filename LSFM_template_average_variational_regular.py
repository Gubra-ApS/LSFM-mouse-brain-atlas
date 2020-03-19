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
import skimage.filters as sf

#### Import custom libraries
import sys
sys.path.insert(0, 'INSERT FOLDER WHERE toolbox.py IS SAVED')

import importlib
import toolbox as bf
importlib.reload(bf)


# Paths
study_folder_server = r'INSERT PATH TO INPUT FOLDER'
study_folder_server_avg = r'INSERT PATH TO OUTPUT FOLDER WHERE THE AVERAGE WILL BE SAVED'

# Selected reference brain from study
reference_brain_id = r'INSERT REFERENCE SAMPLE ID'

# Path to ccfv3 atlas
cffv3_dir = r'INSERT PATH TO THE FOLDER WHERE THE CCFV3 ATLAS IS LOCATED'

# List of samples to include in atlas
sample_id = []
full_brain_udrive = []
sample_flip = []

#### SAMPLES
sample_id.append(r'INSERT SAMPLE ID')
full_brain_udrive.append(r'INSERT FULL PATH TO SAMPLE FOLDER')
sample_flip.append('0') # 0 for no flipping, 1 for flipping

sample_id.append(r'INSERT SAMPLE ID')
sample_flip.append('0')
full_brain_udrive.append(r'INSERT FULL PATH TO SAMPLE FOLDER')

#### 1) Align to Allen atlas for symmetry
### Align to CCFv3 for symmetry
moving = os.path.join(study_folder_server+r'/nifti',reference_brain_id+'_regi.nii.gz')
fixed = cffv3_dir + r'/average_template_25.nii.gz' # ccfv3 template file in 25 um resolution
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = os.path.join(study_folder_server_avg,'nifti')
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

## Affine Registration
 hu.registration('affine_DTI.txt', 'ref_sym_aff', datatype='uint8')

# MANUAL change computed parameters in "ref_sym_aff_man.txt"

# Manual apply computed transform
hu.transform_vol(moving, 'ref_sym_aff_man', 'ref_sym_aff_man', type='vol')


#### 2) Run affine to the reference brain and average, result is A0
fixed = study_folder_server+r'/nifti'+r'ref_sym_aff_man.nii.gz'
moving = os.path.join(study_folder_server+r'/nifti',reference_brain_id+'_regi.nii.gz')
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = study_folder_server+r'/nifti'
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

# do registrations
for n in range(len(sample_id)):
    moving = os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_regi.nii.gz')
    hu = bf.Huginn(moving,fixed,elastix_path,result_path)
    hu.registration('Affine_Gubra_June2019.txt', sample_id[n]+'_aff_A0', datatype='uint8')


# compute average brain
temp = bf.readNifti(fixed)

avg = np.zeros(temp.shape,'float')
avg = avg + temp
divide_mask = np.zeros(temp.shape, 'float')
divide_mask[temp>0]=1

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = bf.readNifti(os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_aff_A0.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask)
bf.saveNifti(avg.astype('uint8'), study_folder_server_avg + '/nifti/A0.nii.gz')
print(counter)

#### 3) Run bspline to A0 in single (low) resolution and average, result is A1
fixed = study_folder_server_avg + '/nifti/A0.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_aff_A0.nii.gz')
    hu = bf.Huginn(moving,fixed,elastix_path,result_path)
    hu.registration('Bspline_Gubra_June2019_step1.txt', sample_id[n]+'_bspline_A1', datatype='uint8')

    # delete files to save space
    os.remove(os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_aff_A0.nii.gz'))

# compute average brain
temp = bf.readNifti(fixed)

avg = np.zeros(temp.shape,'float')
avg = avg + temp
divide_mask = np.zeros(temp.shape, 'float')
divide_mask[temp>0]=1

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = bf.readNifti(os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline_A1.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask)
bf.saveNifti(avg.astype('uint8'), study_folder_server_avg + '/nifti/A1.nii.gz')
print(counter)

#### 4) Run bspline to A1 in single (medium) resolution and average, result is A2
fixed = study_folder_server_avg + '/nifti/A1.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A1.nii.gz')
    hu = bf.Huginn(moving,fixed,elastix_path,result_path)
    hu.registration('Bspline_Gubra_June2019_step2.txt', sample_id[n]+'_bspline_A2', datatype='uint8')

    # delete files to save space
    os.remove(os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A1.nii.gz'))

# compute average brain
temp = bf.readNifti(fixed)

avg = np.zeros(temp.shape,'float')
avg = avg + temp
divide_mask = np.zeros(temp.shape, 'float')
divide_mask[temp>0]=1

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = bf.readNifti(os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline_A2.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask)
bf.saveNifti(avg.astype('uint8'), study_folder_server_avg + '/nifti/A2.nii.gz')
print(counter)

#### 5) Run bspline to A2 in single (high) resolution and average, result is A3
fixed = study_folder_server_avg + '/nifti/A2.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A2.nii.gz')
    hu = bf.Huginn(moving,fixed,elastix_path,result_path)
    hu.registration('Bspline_Gubra_June2019_step3.txt', sample_id[n]+'_bspline_A3', datatype='uint8')

    # delete files to save space
    os.remove(os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A2.nii.gz'))

# compute average brain
temp = bf.readNifti(fixed)

avg = np.zeros(temp.shape,'float')
avg = avg + temp
divide_mask = np.zeros(temp.shape, 'float')
divide_mask[temp>0]=1

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = bf.readNifti(os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline_A3.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask)
bf.saveNifti(avg.astype('uint8'), study_folder_server_avg + '/nifti/A3.nii.gz')
print(counter)

#### 6) Run bspline to A3 in single (higher) resolution and average, result is A4
fixed = study_folder_server_avg + '/nifti/A3.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A3.nii.gz')
    hu = bf.Huginn(moving,fixed,elastix_path,result_path)
    hu.registration('Bspline_Gubra_June2019_step4.txt', sample_id[n]+'_bspline_A4', datatype='uint8')

    # delete files to save space
    os.remove(os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A3.nii.gz'))

# compute average brain
temp = bf.readNifti(fixed)

avg = np.zeros(temp.shape,'float')
avg = avg + temp
divide_mask = np.zeros(temp.shape, 'float')
divide_mask[temp>0]=1

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = bf.readNifti(os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline_A4.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask)
bf.saveNifti(avg.astype('uint8'), study_folder_server_avg + '/nifti/A4.nii.gz')
print(counter)

#### 7) Run bspline to A4 in single (highest) resolution and average, result is A5
fixed = study_folder_server_avg + '/nifti/A4.nii.gz'

# do registrations
for n in range(len(sample_id)):
    moving = os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A4.nii.gz')
    hu = bf.Huginn(moving,fixed,elastix_path,result_path)
    hu.registration('Bspline_Gubra_June2019_step5.txt', sample_id[n]+'_bspline_A5', datatype='uint8')

    # delete files to save space
    os.remove(os.path.join(study_folder_server+r'/nifti',sample_id[n]+'_bspline_A4.nii.gz'))


# # compute average brain
temp = bf.readNifti(fixed)

avg = np.zeros(temp.shape,'float')
avg = avg + temp
divide_mask = np.zeros(temp.shape, 'float')
divide_mask[temp>0]=1
divide_mask[(temp.shape[0]-14):temp.shape[0],160:491,:] = 0

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = bf.readNifti(os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline_A5.nii.gz'))

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1
    temp_mask[(temp.shape[0]-14):temp.shape[0],160:491,:] = 0

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1

avg = np.divide(avg,divide_mask)
bf.saveNifti(avg.astype('uint8'), study_folder_server_avg + '/nifti/A5.nii.gz')
print(counter)

#### 8) Flip average to normal orientation!!!!!!!
A5 = bf.readNifti(study_folder_server_avg + '/nifti/A5.nii.gz')
A5 = np.flip(A5,1)
bf.saveNifti(A5,study_folder_server_avg + '/nifti/A5_flip.nii.gz')

#### 9) Align to Allen atlas for symmetry
moving = study_folder_server_avg + '/nifti/A5_flip.nii.gz'
fixed = cffv3_dir + r'/average_template_25.nii.gz'
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'
result_path = study_folder_server_avg+r'/nifti'
hu = bf.Huginn(moving,fixed,elastix_path,result_path)

### Affine Registration
hu.registration('affine_DTI.txt', 'A5_sym_aff', datatype='uint8')



# MANUAL change computed parameters in "A5_sym_aff_man.txt"

# Manual apply computed transform
#hu.transform_vol(moving, 'A5_sym_aff_man', 'A5_sym_aff_man', type='vol')
