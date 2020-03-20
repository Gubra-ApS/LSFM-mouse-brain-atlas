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

import pandas as pd

#### Import custom libraries
import sys
sys.path.insert(0, 'INSERT FOLDER WHERE toolbox.py IS SAVED')

import importlib
import toolbox as bf
importlib.reload(bf)
####

# Path where to save volumes
study_folder_server = r'INSERT PATH TO OUTPUT FOLDER WHERE THE VOLUMES WILL BE SAVED'

gubra_template_dir = r'INSERT PATH TO THE FOLDER WITH THE AVERAGE BRAIN'
cffv3_dir = r'INSERT PATH TO THE FOLDER WITH THE aibs ccfV3 25 um BRAIN'


# Use bg masked versions for initial registration
gubra_temp = bf.readNifti(os.path.join(gubra_template_dir,'gubra_template.nii.gz'))
gubra_mask = bf.readNifti(os.path.join(gubra_template_dir,'gubra_tissue_mask.nii.gz')) # tissue mask for LSFM template
gubra_temp[gubra_mask==0] = 0
bf.saveNifti(gubra_temp.astype('uint8'),os.path.join(gubra_template_dir,'gubra_template_masked.nii.gz'))


################# INITIAL FULL VOLUME REGISTRATION #############################
## AFFINE
program = r'elastix -threads 16 '; #elastix program is added to PATH
fixed_name = r'-f ' + os.path.join(gubra_template_dir+'gubra_template_masked.nii.gz ');
moving_name = r'-m ' + os.path.join(cffv3_dir + r'average_template_25.nii.gz ')
outdir = r'-out ' + study_folder_server + r'/elastix/workingDir ';
params = r'-p ' + study_folder_server + r'/elastix/Par0000affine_cm.txt';

os.system(program + fixed_name + moving_name  + outdir + params)
move(study_folder_server + r'/elastix/workingDir/result.0.nii.gz', os.path.join(study_folder_server+r'/nifti_25um','ccfv3_affine.nii.gz'))
move(study_folder_server + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server+r'/nifti_25um','ccfv3_affine.txt'))

## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_affine.nii.gz'))
temp = sitk.GetArrayFromImage(temp)
temp[temp<0] = 0

final = temp
final = final.astype('uint8')
final_sitk = sitk.GetImageFromArray(final)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_affine.nii.gz'))


# # BSPLINE
params = r'-p ' + study_folder_server + r'/elastix/Par0000bspline_cm_full.txt ';
t0 = r'-t0 ' + os.path.join(study_folder_server+r'/nifti_25um','ccfv3_affine.txt')

os.system(program + fixed_name + moving_name  + outdir + params + t0)

move(study_folder_server + r'/elastix/workingDir/result.0.nii.gz', os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline.nii.gz'))
move(study_folder_server + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline.txt'))

## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline.nii.gz'))
temp = sitk.GetArrayFromImage(temp)
temp[temp<0] = 0

final = temp
final = final.astype('uint8')
final_sitk = sitk.GetImageFromArray(final)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline.nii.gz'))


# move annotation image along
program = r'INSERT PATH TO TRANSFORMIX ';
moving_name = r'-in ' + os.path.join(cffv3_dir + r'ano_large_structures_full_man_sym.nii.gz ')
outdir = r'-out ' + study_folder_server + r'/elastix/workingDir ';
trans_params = r'-tp ' + os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline.txt');

####
with open(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline.txt'), 'r') as file:
    # read a list of lines into data
    data = file.readlines()

data[-8] = "(FinalBSplineInterpolationOrder 0) \n"

## and write everything back
with open(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline.txt'), 'w') as file:
    file.writelines( data )

# transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
os.system(program + moving_name + outdir + trans_params)

move(study_folder_server + r'/elastix/workingDir/result.nii.gz', os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline_ano_large.nii.gz'))

## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline_ano_large.nii.gz'))
temp = sitk.GetArrayFromImage(temp)

final = temp
final = final.astype('uint32')
final_sitk = sitk.GetImageFromArray(final)
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline_ano_large.nii.gz'))

# MANUALLY CORRECT ccfv3_bspline_ano_large TO INCLUDE ALL TISSUE OF gubra_template.nii.gz

# USE BG MASK TO REMOVE ANY PARTS OF CCFV3_BSPLINE... WITH NO TISSUE
ano_large = bf.readNifti(os.path.join(gubra_template_dir,'ccfv3_bspline_ano_large_man1.nii.gz'))
gubra_mask = bf.readNifti(os.path.join(gubra_template_dir,'gubra_tissue_mask.nii.gz'))
ano_large[gubra_mask==0] = 0
bf.saveNifti(ano_large.astype('uint8'),os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline_ano_large_final.nii.gz'))


################# SPLIT INTO PARTS AND ALIGN CCFV3 ANNOTATIONS TO LSFM TEMPLATE #############################

ano_ids = []
ano_names = []

ano_ids.append(1)
ano_ids.append(2)
ano_ids.append(3)
ano_ids.append(4)
ano_ids.append(5)
ano_ids.append(6)

ano_names.append('cortex')
ano_names.append('cerebral nuclei')
ano_names.append('interbrain')
ano_names.append('cerebellum')
ano_names.append('hindbrain')
ano_names.append('septal')

gubra_avg = sitk.ReadImage(os.path.join(gubra_template_dir,'gubra_template_masked.nii.gz'))
gubra_avg = sitk.GetArrayFromImage(gubra_avg)

ano_parts_bspline = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline_ano_large_final.nii.gz'))
ano_parts_bspline = sitk.GetArrayFromImage(ano_parts_bspline)

ccfv3_template = sitk.ReadImage(os.path.join(cffv3_dir + r'average_template_25.nii.gz'))
ccfv3_template = sitk.GetArrayFromImage(ccfv3_template)

ano_parts = sitk.ReadImage(os.path.join(cffv3_dir + r'ano_large_structures_full_man_sym.nii.gz'))
ano_parts = sitk.GetArrayFromImage(ano_parts)

ccfv3_ano = sitk.ReadImage(os.path.join(cffv3_dir + r'ano_25.nii.gz'))
ccfv3_ano = sitk.GetArrayFromImage(ccfv3_ano)


for i in range(len(ano_ids)):

    ## Align individual parts
    temp_gubra = np.copy(gubra_avg)
    temp_gubra[ano_parts_bspline!=ano_ids[i]] = 0

    temp_ccfv3 = np.copy(ccfv3_template)
    temp_ccfv3[ano_parts!=ano_ids[i]] = 0

    temp_mask = np.ones(ano_parts_bspline.shape) # fixed space
    temp_mask[ano_parts_bspline!=ano_ids[i]] = 0

    temp_ano_full = np.copy(ccfv3_ano)
    temp_ano_full[ano_parts!=ano_ids[i]] = 0

    final_sitk = sitk.GetImageFromArray(temp_gubra.astype('uint8'))
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','gubra_'+ano_names[i]+'.nii.gz'))

    final_sitk = sitk.GetImageFromArray(temp_ccfv3.astype('uint8'))
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_'+ano_names[i]+'.nii.gz'))

    final_sitk = sitk.GetImageFromArray(temp_mask.astype('uint8'))
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','mask_'+ano_names[i]+'.nii.gz'))

    final_sitk = sitk.GetImageFromArray(temp_ano_full.astype('uint32'))
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','anoFull_'+ano_names[i]+'.nii.gz'))


    # PRE-PROCESSING OF HINDBRAIN PARTS
    if ano_names[i] == 'hindbrain':
        ccfv3_hind = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_'+'hindbrain'+'.nii.gz'))
        ccfv3_hind = sitk.GetArrayFromImage(ccfv3_hind)

        ccfv3_hind_ano = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','anoFull_'+'hindbrain'+'.nii.gz'))
        ccfv3_hind_ano = sitk.GetArrayFromImage(ccfv3_hind_ano)

        # mask ventricular system
        df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular.csv'),header=None)

        for index, row in df.iterrows():
            ccfv3_hind[ccfv3_hind_ano==row[0]] = 0

        temp_niti_sitk = sitk.GetImageFromArray(ccfv3_hind.astype('uint8'))
        temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_'+'hindbrain'+'.nii.gz'))


        gubra_hind = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','gubra_'+'hindbrain'+'.nii.gz'))
        gubra_hind = sitk.GetArrayFromImage(gubra_hind)

        threshold = 55 # manual selected

        gubra_bw = np.copy(gubra_hind)
        gubra_bw[gubra_bw<threshold] = 0
        gubra_bw[gubra_bw>0] = 1

        gubra_label = skimage.measure.label(gubra_bw, neighbors=None, connectivity=2)

        props = regionprops(gubra_label)

        gubra_bw = skimage.morphology.remove_small_objects(gubra_label, 50000 ,connectivity=2)
        gubra_bw[gubra_bw>0] = 1

        gubra_hind[gubra_bw!=1] = 0

        temp_niti_sitk = sitk.GetImageFromArray(gubra_hind.astype('uint8'))
        temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server+r'/nifti_25um','gubra_'+'hindbrain'+'.nii.gz'))
    #############################


    # PRE-PROCESSING OF SEPTAL PARTS
    if ano_names[i] == 'septal':
        ccfv3_hind = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_'+'septal'+'.nii.gz'))
        ccfv3_hind = sitk.GetArrayFromImage(ccfv3_hind)

        ccfv3_hind_ano = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','anoFull_'+'septal'+'.nii.gz'))
        # ccfv3_hind_ano = sitk.GetArrayFromImage(ccfv3_hind_ano)

        ccfv3_hind[ccfv3_hind<18] = 0
        temp_niti_sitk = sitk.GetImageFromArray(ccfv3_hind.astype('uint8'))
        temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_'+'septal'+'.nii.gz'))

        gubra_hind = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','gubra_'+'septal'+'.nii.gz'))
        gubra_hind = sitk.GetArrayFromImage(gubra_hind)

        gubra_hind[gubra_hind<40] = 0
        temp_niti_sitk = sitk.GetImageFromArray(gubra_hind.astype('uint8'))
        temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server+r'/nifti_25um','gubra_'+'septal'+'.nii.gz'))
    #############################


    # AFFINE
    program = r'elastix -threads 16 '; # elastix is added to PATH
    fixed_name = r'-f ' + os.path.join(study_folder_server+r'/nifti_25um','gubra_'+ano_names[i]+'.nii.gz ');
    moving_name = r'-m ' + os.path.join(study_folder_server+r'/nifti_25um','ccfv3_'+ano_names[i]+'.nii.gz ')
    outdir = r'-out ' + study_folder_server + r'/elastix/workingDir ';
    params = r'-p ' + study_folder_server + r'/elastix/Par0000affine_parts.txt ';
    fmask = r'-fmask ' + os.path.join(study_folder_server+r'/nifti_25um','mask_'+ano_names[i]+'.nii.gz')

    os.system(program + fixed_name + moving_name  + outdir + params + fmask)
    move(study_folder_server + r'/elastix/workingDir/result.0.nii.gz', os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_affine.nii.gz'))
    move(study_folder_server + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_affine.txt'))

    ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
    temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_affine.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)
    temp[temp<0] = 0

    final = temp
    final = final.astype('uint8')
    final_sitk = sitk.GetImageFromArray(final)
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_affine.nii.gz'))


    # BSPLINE
    if ano_names[i] == 'septal':
        params = r'-p ' + study_folder_server + r'/elastix/Par0000bspline_parts_septal.txt ';
    elif ano_names[i] == 'hindbrain':
        params = r'-p ' + study_folder_server + r'/elastix/Par0000bspline_parts_hindbrain.txt ';
    else:
        params = r'-p ' + study_folder_server + r'/elastix/Par0000bspline_parts.txt ';


    t0 = r'-t0 ' + os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_affine.txt ')

    os.system(program + fixed_name + moving_name  + outdir + params + t0 + fmask)

    move(study_folder_server + r'/elastix/workingDir/result.0.nii.gz', os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.nii.gz'))
    move(study_folder_server + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.txt'))

    ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
    temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)
    temp[temp<0] = 0

    final = temp
    final = final.astype('uint8')
    final_sitk = sitk.GetImageFromArray(final)
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.nii.gz'))

    # Move full annotations along
    program = r'INSERT TRANSFORMIX PATH ';
    moving_name = r'-in ' + os.path.join(study_folder_server+r'/nifti_25um','anoFull_'+ano_names[i]+'.nii.gz ')
    outdir = r'-out ' + study_folder_server + r'/elastix/workingDir ';
    trans_params = r'-tp ' + os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.txt');

    #### Change B-Spline interpolation order to 0 for transforming annotation file
    with open(os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.txt'), 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    data[-8] = "(FinalBSplineInterpolationOrder 0) \n"
    ## and write everything back
    with open(os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.txt'), 'w') as file:
        file.writelines( data )

    # Perform transformation
    os.system(program + moving_name + outdir + trans_params)

    move(study_folder_server + r'/elastix/workingDir/result.nii.gz', os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline_anoFull.nii.gz'))

    ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
    temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline_anoFull.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)

    final = temp
    final = final.astype('uint32')
    final_sitk = sitk.GetImageFromArray(final)
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline_anoFull.nii.gz'))


################# COLLECT PARTS #############################
collected = np.zeros(gubra_avg.shape)
collected_ano = np.zeros(gubra_avg.shape)

for i in range(len(ano_ids)):
    temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)

    temp_ano = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um',ano_names[i]+'_bspline_anoFull.nii.gz'))
    temp_ano = sitk.GetArrayFromImage(temp_ano)

    collected[ano_parts_bspline==ano_ids[i]] = temp[ano_parts_bspline==ano_ids[i]]
    collected_ano[ano_parts_bspline==ano_ids[i]] = temp_ano[ano_parts_bspline==ano_ids[i]]

final_sitk = sitk.GetImageFromArray(collected.astype('uint8'))
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline.nii.gz'))

final_sitk = sitk.GetImageFromArray(collected_ano.astype('uint32'))
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ano.nii.gz'))


################ FILL IN BORDER GAPS BY DISTANCE/FEATURE MAP COMPUTATION (ANO + TEMPLATE) #############################
collected_ano = bf.readNifti(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ano.nii.gz'))

test_bw = np.copy(collected_ano)
test_bw[test_bw>0] = 1
inv_bw = np.ones(test_bw.shape)
inv_bw[test_bw==1] = 0

edt, inds = ndimage.distance_transform_edt(inv_bw, return_indices=True)

test_full = np.zeros(test_bw.shape)
for i in range(test_bw.shape[0]):
    for j in range(test_bw.shape[1]):
        for k in range(test_bw.shape[2]):
            test_full[i,j,k] = collected_ano[inds[0,i,j,k], inds[1,i,j,k], inds[2,i,j,k]]

test_full[ano_parts_bspline==0] = 0
collected_ano[collected_ano==0] = test_full[collected_ano==0]

# add a BG mask
tissue_mask = bf.readNifti(os.path.join(gubra_template_dir,'gubra_tissue_mask.nii.gz'))

collected_ano[tissue_mask==0] = 0
final_sitk = sitk.GetImageFromArray(collected_ano.astype('uint32'))
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ano_final.nii.gz'))


## Save ventricular mask
ano_full = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ano_final.nii.gz'))
ano_full = sitk.GetArrayFromImage(ano_full)

# Make new volume
ven_white = np.zeros(ano_full.shape)

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular.csv'),header=None)

for index, row in df.iterrows():
    ven_white[ano_full==row[0]] = 1

print('Done ventricular')

temp_niti_sitk = sitk.GetImageFromArray(ven_white.astype('uint8'))
temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ventricular.nii.gz'))


# Use manually corrected ventricular mask to change any non-ventricular labels inside the ventricular system
#to nearest ventricular label
ano = bf.readNifti(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ano_final.nii.gz'))
ventr_mask = bf.readNifti(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ventricular_man.nii.gz'))

ano_new = np.copy(ano)
ano_new[ventr_mask!=1] = 0
ano_new[ventr_mask==1] = 0

df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular.csv'),header=None)

for index, row in df.iterrows():
    ano_new[ano==row[0]] = row[0]


# Fill in missing labels
test_bw = np.copy(ano_new)
test_bw[test_bw>0] = 1
inv_bw = np.ones(test_bw.shape)
inv_bw[test_bw==1] = 0

edt, inds = ndimage.distance_transform_edt(inv_bw, return_indices=True)

test_full = np.zeros(test_bw.shape)
for i in range(test_bw.shape[0]):
    for j in range(test_bw.shape[1]):
        for k in range(test_bw.shape[2]):
            test_full[i,j,k] = ano_new[inds[0,i,j,k], inds[1,i,j,k], inds[2,i,j,k]]

test_full[ventr_mask==0] = 0
ano_new[ano_new==0] = test_full[ano_new==0]


ano[ventr_mask==1] = ano_new[ventr_mask==1]

# Fill in non-ventricular labels in all tissue
ventr_mask = bf.readNifti(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ventricular_man.nii.gz'))

ano_new = np.copy(ano)
ano_new[ventr_mask!=1] = 0
ano_new[ventr_mask==1] = 0

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular.csv'),header=None)

for index, row in df.iterrows():
    ano_new[ano==row[0]] = 1

use_mask = np.copy(ano_new)
use_mask[ventr_mask==1] = 0

ano_temp=np.copy(ano)
ano_temp[ano_new==1] = 0
ano_temp[ventr_mask==1] = 0


# Fill in missing labels
test_bw = np.copy(ano_temp)
test_bw[test_bw>0] = 1
inv_bw = np.ones(test_bw.shape)
inv_bw[test_bw==1] = 0

edt, inds = ndimage.distance_transform_edt(inv_bw, return_indices=True)

test_full = np.zeros(test_bw.shape)
for i in range(test_bw.shape[0]):
    for j in range(test_bw.shape[1]):
        for k in range(test_bw.shape[2]):
            test_full[i,j,k] = ano[inds[0,i,j,k], inds[1,i,j,k], inds[2,i,j,k]]


ano_temp[ano_temp==0] = test_full[ano_temp==0]
ano_temp[ventr_mask==1] = 0


ano[use_mask==1] = ano_temp[use_mask==1]
bf.saveNifti(ano, os.path.join(study_folder_server+r'/nifti_25um','ccfv3_parts_bspline_ventricular_man_corr.nii.gz'))
