# -*- coding: utf-8 -*-

## Import libraries
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
import scipy.ndimage.interpolation as sci_int
import scipy.ndimage.morphology as sci_morph

import pandas as pd

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

import sys

from subprocess import call, run
import subprocess

from math import sqrt

import SimpleITK as sitk

from shutil import move

import math

import seaborn as sns

import cv2

# load annotation volume
ano_full = sitk.ReadImage(os.path.join(r'INSERT PATH TO AIBS CCFV3 25um','ano_25.nii.gz'))
ano_full = sitk.GetArrayFromImage(ano_full)

# Make a new volume
ano_large_structures = np.zeros(ano_full.shape)

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','cereb_cortex.csv'),header=None)

for index, row in df.iterrows():
    ano_large_structures[ano_full==row[0]] = 1
    
print('Done Cereb cortex')


# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','cereb_nuclei.csv'),header=None)

for index, row in df.iterrows():
    ano_large_structures[ano_full==row[0]] = 2
    
print('Done cereb_nuclei')

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','interbrain.csv'),header=None)

for index, row in df.iterrows():
    ano_large_structures[ano_full==row[0]] = 3
    
print('Done interbrain')

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','midbrain.csv'),header=None)

for index, row in df.iterrows():
    ano_large_structures[ano_full==row[0]] = 3
    
print('Done midbrain')

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','cerebellum.csv'),header=None)

for index, row in df.iterrows():
    ano_large_structures[ano_full==row[0]] = 4
    
print('Done cerebellum')

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','hindbrain.csv'),header=None)

for index, row in df.iterrows():
    ano_large_structures[ano_full==row[0]] = 5
    
print('Done hindbrain')

# Save the volume
ano_large_structures = ano_large_structures.astype('uint16')
final_sitk = sitk.GetImageFromArray(ano_large_structures)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(r'INSERT PATH TO AIBS CCFV3 25um',r'ano_large_structures.nii.gz'))

# Fill in ventricular and white matter tracts
ano_large_structures = sitk.ReadImage(os.path.join(r'INSERT PATH TO AIBS CCFV3 25um','ano_large_structures.nii.gz'))
ano_large_structures = sitk.GetArrayFromImage(ano_large_structures)

test_bw = np.copy(ano_large_structures)
test_bw[test_bw>0] = 1

inv_bw = np.ones(test_bw.shape)
inv_bw[test_bw==1] = 0
print(inv_bw.shape)
print(np.min(inv_bw))
print(np.max(inv_bw))

edt, inds = ndimage.distance_transform_edt(inv_bw, return_indices=True)

test_full = np.zeros(test_bw.shape)

for i in range(test_bw.shape[0]):
    for j in range(test_bw.shape[1]):
        for k in range(test_bw.shape[2]):
            test_full[i,j,k] = ano_large_structures[inds[0,i,j,k], inds[1,i,j,k], inds[2,i,j,k]]

#temp_niti_sitk = sitk.GetImageFromArray(test_full.astype('uint16'))
#temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
#sitk.WriteImage(temp_niti_sitk,r'INSERT PATH TO AIBS CCFV3 25um' + r'/test_full.nii.gz')

ano_large_structures = sitk.ReadImage(os.path.join(r'INSERT PATH TO AIBS CCFV3 25um','ano_large_structures.nii.gz'))
ano_large_structures = sitk.GetArrayFromImage(ano_large_structures)

ano_large_structures_full = np.copy(ano_large_structures)
ano_large_structures_full[ano_full>0] = test_full[ano_full>0]

temp_niti_sitk = sitk.GetImageFromArray(ano_large_structures_full.astype('uint16'))
temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
sitk.WriteImage(temp_niti_sitk,r'INSERT PATH TO AIBS CCFV3 25um' + r'/ano_large_structures_full.nii.gz')

# Mask for ventricles and fibers
# load annotation volume
ano_full = sitk.ReadImage(os.path.join(r'INSERT PATH TO AIBS CCFV3 25um','ano_25.nii.gz'))
ano_full = sitk.GetArrayFromImage(ano_full)

# Make new volume
ven_white = np.zeros(ano_full.shape)

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular.csv'),header=None)

for index, row in df.iterrows():
    ven_white[ano_full==row[0]] = 1
    
print('Done ventricular')

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','fiber_tracts.csv'),header=None)

for index, row in df.iterrows():
    ven_white[ano_full==row[0]] = 1
    
print('Done fiber_tracts')

temp_niti_sitk = sitk.GetImageFromArray(ven_white.astype('uint16'))
temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
sitk.WriteImage(temp_niti_sitk,r'INSERT PATH TO AIBS CCFV3 25um' + r'/vent_fiber.nii.gz')

# Add septal complex
ano_large_structures = sitk.ReadImage(os.path.join(r'INSERT PATH TO AIBS CCFV3 25um','ano_large_structures_full_man.nii.gz')) # after manual correction
ano_large_structures = sitk.GetArrayFromImage(ano_large_structures)

 load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','lateral_septal.csv'),header=None)

for index, row in df.iterrows():
    ano_large_structures[ano_full==row[0]] = 6
    
print('Done LSX')

temp_niti_sitk = sitk.GetImageFromArray(ano_large_structures.astype('uint16'))
temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
sitk.WriteImage(temp_niti_sitk,r'INSERT PATH TO AIBS CCFV3 25um' + r'/ano_large_structures_full_man_lsx.nii.gz')

# Symmetry
ano_large_structures_fix = sitk.ReadImage(os.path.join(r'INSERT PATH TO AIBS CCFV3 25um','ano_large_structures_full_man_lsx1.nii.gz')) # after manual correction
ano_large_structures_fix = sitk.GetArrayFromImage(ano_large_structures_fix)

ano_large_structures_fix[ano_large_structures_fix==7] = 6

print(ano_large_structures_fix.shape)

part1 = ano_large_structures_fix[:,:,227:455]
print(part1.shape)

part2 = np.flip(part1,2)
print(part2.shape)

plt.imshow(part2[176,:,:])

final = np.concatenate((part2,part1),2)

plt.imshow(final[176,:,:])
print(final.shape)


temp_niti_sitk = sitk.GetImageFromArray(final.astype('uint16'))
temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
sitk.WriteImage(temp_niti_sitk,r'INSERT PATH TO AIBS CCFV3 25um' + r'/ano_large_structures_full_man_sym.nii.gz')



