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

#### Import custom libraries
import sys
sys.path.insert(0, 'INSERT FOLDER WHERE toolbox.py IS SAVED')

import importlib
import toolbox as bf
importlib.reload(bf)
####

# Paths
study_folder_server = r'INSERT PATH TO INPUT FOLDER'
study_folder_server_avg = r'INSERT PATH TO OUTPUT FOLDER WHERE THE AVERAGE WILL BE SAVED'


# Selected reference brain from study
reference_brain_id = r'INSERT REFERENCE SAMPLE ID'


# Define path and sample names
sample_id = []
sample_id_no_filter = []
full_brain_udrive = []

#### SAMPLES
sample_id.append(r'INSERT SAMPLE ID')
full_brain_udrive.append(r'INSERT FULL PATH TO SAMPLE FOLDER')



# compute average brain
temp = sitk.ReadImage(os.path.join(study_folder_server_indi+r'/nifti',reference_brain_id+'_regi.nii.gz'))
temp = sitk.GetArrayFromImage(temp)
temp[temp<0] = 0
temp[temp>255] = 255


avg = np.zeros(temp.shape,'float')
avg = avg + temp
divide_mask = np.zeros(temp.shape, 'float')
divide_mask[temp>0]=1

counter = 1
for i in range(len(sample_id)):
    print(sample_id[i])

    temp = sitk.ReadImage(os.path.join(study_folder_server_indi+r'/nifti',sample_id[i]+'_bspline.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)
    temp[temp<0] = 0
    temp[temp>255] = 255

    temp_mask = np.copy(temp)
    temp_mask[temp_mask>0] = 1

    avg = avg + temp
    divide_mask = divide_mask + temp_mask
    counter = counter+1



# Use bounding box to remove any full zero rows
bw = np.copy(divide_mask)
bw[bw>0] = 1
props = regionprops(bw.astype('uint16'))
minz, minr, minc, maxz, maxr, maxc = props[0].bbox
print([minz, minr, minc, maxz, maxr, maxc])

divide_mask = divide_mask[minz:maxz,minr:maxr,minc:maxc]
avg = avg[minz:maxz,minr:maxr,minc:maxc]

avg = np.divide(avg,divide_mask)

# add some padding for registration
avg = np.pad(avg,((0,30),(0,245),(0,0)),'constant', constant_values=(0, 0))
avg=avg[:,80:avg.shape[1],:]

# Save average brain
avg = avg.astype('uint16')
final_sitk = sitk.GetImageFromArray(avg)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti','avg_gubra_olf.nii.gz'))

print(counter)
