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
from skimage.filters import gaussian

import sys

from subprocess import call, run
import subprocess

from math import sqrt

import SimpleITK as sitk

from shutil import move

import math

import seaborn as sns
import pandas as pd

from skimage.morphology import disk, dilation, erosion

from skimage import data
import skimage.filters as sf
import matplotlib.pyplot as plt

#### Import custom libraries
import sys
sys.path.insert(0, 'INSERT FOLDER WHERE toolbox.py IS SAVED')

import importlib
import toolbox as bf
importlib.reload(bf)
####

# Paths
study_folder_server_avg = r'INSERT PATH TO OUTPUT FOLDER WHERE THE AVERAGE IMAGES WILL BE SAVED'

#
#### test image sharpening
image = bf.readNifti(os.path.join(study_folder_server_avg,'nifti/avg_gubra_final.nii.gz'))
for i in range(image.shape[0]):
    temp = image[i,:,:]
    temp = sf.unsharp_mask(temp, radius=1, amount=1.5, preserve_range=True)
    temp[temp<0] = 0
    temp[temp>255] = 255
    image[i,:,:] = np.uint8(temp)

print(image.shape)

bf.saveNifti(image,os.path.join(study_folder_server_avg,'nifti/avg_gubra_final_sharp.nii.gz'))

### GET Full symmetry - move up before version 1 registration
# Define midline + start/stop
blend_patting = 12

midline1 = 246
start1 = 0; stop1 = 50+blend_patting

midline2 = 247
start2 = 50-blend_patting; stop2 = 185+blend_patting

midline3 = 246
start3 = 185-blend_patting; stop3 = 491


# take out parts
buffer = 6 # pixels on each side not mirrored
part1 = image[:,start1:stop1,:]
part2 = image[:,start2:stop2,:]
part3 = image[:,start3:stop3,:]

# take out half
part1_half = part1[:,:,buffer:midline1]
part2_half = part2[:,:,buffer:midline2]
part3_half = part3[:,:,buffer:midline3]

# save midline
part1_midline = part1[:,:,midline1-2:midline1+2]
part2_midline = part2[:,:,midline2-2:midline2+2]
part3_midline = part3[:,:,midline3-2:midline3+2]

# construct flipped
part1_sym = np.concatenate((part1_half, np.flip(part1_half[:,:,0:part1_half.shape[2]-2],2)),axis=2)
part2_sym = np.concatenate((part2_half, np.flip(part2_half[:,:,0:part1_half.shape[2]-2],2)),axis=2)
part3_sym = np.concatenate((part3_half, np.flip(part3_half[:,:,0:part1_half.shape[2]-2],2)),axis=2)

# place on common midline
common_midline = 245

offset1 = int(common_midline-(part1_sym.shape[2]/2))
offset2 = int(common_midline-(part2_sym.shape[2]/2))
offset3 = int(common_midline-(part3_sym.shape[2]/2))

part1_final = np.copy(part1)
part1_final[:,:,offset1:offset1+part1_sym.shape[2]] = part1_sym
part2_final = np.copy(part2)
part2_final[:,:,offset2:offset2+part2_sym.shape[2]] = part2_sym
part3_final = np.copy(part3)
part3_final[:,:,offset3:offset3+part3_sym.shape[2]] = part3_sym

# blend parts together
# define blend function
def sigmoid(x):
  y = np.zeros(len(x))
  for i in range(len(x)):
    y[i] = 1 / (1 + math.exp(-x[i]))
  return y

sigmoid_ = sigmoid(np.arange(-3, 3, 1/4))
alpha = np.repeat(sigmoid_.reshape((len(sigmoid_), 1)), repeats=(491), axis=1)

### First blend
# pull out parts
part1_blend = part1_final[:,part1_final.shape[1]-blend_patting*2:part1_final.shape[1],:]
part2_blend = part2_final[:,0:blend_patting*2,:]
blended_first = np.zeros(part1_blend.shape)

# loop dimension
for i in range(blended_first.shape[0]):
    image1_connect = part1_blend[i,:,:]
    image2_connect = part2_blend[i,:,:]
    blended_first[i,:,:] = image1_connect * (1.0 - alpha) + image2_connect * alpha

### Second blend
# pull out parts
part2_blend = part2_final[:,part2_final.shape[1]-blend_patting*2:part2_final.shape[1],:]
part3_blend = part3_final[:,0:blend_patting*2,:]
blended_second = np.zeros(part2_blend.shape)

# loop dimension
for i in range(blended_second.shape[0]):
    image1_connect = part2_blend[i,:,:]
    image2_connect = part3_blend[i,:,:]
    blended_second[i,:,:] = image1_connect * (1.0 - alpha) + image2_connect * alpha

# put parts together
final = np.concatenate((part1_final[:,0:part1_final.shape[1]-blend_patting*2,:],blended_first,part2_final[:,blend_patting*2:part2_final.shape[1]-blend_patting*2,:], blended_second, part3_final[:,blend_patting*2:part3_final.shape[1]-blend_patting*2,:]),axis=1)
bf.saveNifti(final,os.path.join(study_folder_server_avg,'nifti/final.nii.gz'))


#Final symmetry of all volumes : mirror one hemisphere
final_touch = bf.readNifti(s.path.join(study_folder_server_avg,'nifti/final.nii.gz'))
temp = final_touch[:,:,1:230]
print(temp.shape)
print(final_touch.shape)
temp = np.flip(temp,2)
final_touch[:,:,232:232+temp.shape[2]] = temp
bf.saveNifti(final_touch, s.path.join(study_folder_server_avg,'nifti/final_touch_symm.nii.gz'))


# Gamma correction of template
for i in range(final_touch.shape[0]):
  final_touch[i,:,:] = skimage.exposure.adjust_gamma(final_touch[i,:,:], 0.9)

final_touch = skimage.exposure.rescale_intensity(final_touch, in_range='image', out_range='uint8')

bf.saveNifti(final_touch, s.path.join(study_folder_server_avg,'nifti/gubra_template.nii.gz'))
