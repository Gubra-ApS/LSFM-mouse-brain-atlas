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
sys.path.insert(0, '/home/cgs/custom_python_packages')

import importlib
import bifrost as bf
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

# USE BG MASK TO REMOVE ANY PARTS OF CCFV3_BSPLINE WITH NO TISSUE
ano_large = bf.readNifti(os.path.join(gubra_template_dir,'ccfv3_bspline_ano_large_man1.nii.gz'))
gubra_mask = bf.readNifti(os.path.join(gubra_template_dir,'gubra_tissue_mask.nii.gz'))
ano_large[gubra_mask==0] = 0
bf.saveNifti(ano_large.astype('uint8'),os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline_ano_large_final.nii.gz'))



####### Add olfactory bulb annotation to the Allen CCFv3 parental regions ########

## Load volumes
aibs_ano_olf = bf.readNifti(os.path.join(cffv3_dir,'ccfv3_ano_olf.nii.gz'))
aibs_ano = bf.readNifti(os.path.join(cffv3_dir,'ano_large_structures_full_man_sym.nii.gz'))

## Add olfactory annotation to the parental region called cortex
# Expand the parental region volume
aibs_ano_pad = np.zeros((aibs_ano_olf.shape[0],aibs_ano_olf.shape[1], aibs_ano_olf.shape[2]))
z_diff = aibs_ano_olf.shape[1] - aibs_ano.shape[1]
aibs_ano_pad[:,z_diff:,:] = aibs_ano

# Make an annotation volume for olfactory bulb  using the cortex label (parental region) which is 1
aibs_ano_olf[aibs_ano_olf>0] = 1

# Add olfactory label to the parental label volume
aibs_ano_pad[:,0:z_diff,:] = aibs_ano_olf[:,0:z_diff,:]

 # Save
bf.saveNifti(aibs_ano_pad,os.path.join(cffv3_dir,'ano_large_structures_full_man_sym_olf.nii.gz'))


### Add olfactory bulb annotation to the Gubra atlas parental regions
## Load volumes
aibs_ano_olf = bf.readNifti(os.path.join(gubra_template_dir,'gubra_template_olf.nii.gz'))
aibs_ano = bf.readNifti(os.path.join(study_folder_server+r'/nifti_25um','ccfv3_bspline_ano_large_final.nii.gz')

## Add olfactory annotation to the parental region called cortex
# Expand the parental region volume
aibs_ano_pad = np.zeros((aibs_ano_olf.shape[0],aibs_ano_olf.shape[1], aibs_ano_olf.shape[2]))
z_diff = aibs_ano_olf.shape[1] - aibs_ano.shape[1]
aibs_ano_pad[:,z_diff:,:] = aibs_ano

# Make an annotation volume for olfactory bulb  using the cortex label (parental region) which is 1
aibs_ano_olf[aibs_ano_olf>0] = 1

# Add olfactory label to the parental label volume
aibs_ano_pad[:,0:155,:] = aibs_ano_olf[:,0:155,:]

 # Save
bf.saveNifti(aibs_ano_pad,os.path.join(study_folder_server,'ano_large_structures_olf.nii.gz'))

# Correct manually the segmentation for the olfactory bulb and make symmetric
ano_olf = bf.readNifti(os.path.join(study_folder_server,'ano_large_structures_olf_man.nii.gz'))
ano_olf_sym = np.copy(ano_olf)
ano_half = np.flip(ano_olf[:,:,0:230],axis=2)
ano_olf_sym[:,:,231:] = ano_half


bf.saveNifti(ano_olf_sym,os.path.join(study_folder_server,'ano_large_structures_olf_man_sym.nii.gz'))



################# SPLIT INTO PARTS AND ALIGN CCFV3 ANNOTATIONS TO LSFM TEMPLATE #############################

## Make individual anatomical and ano volumes per large brain region
## Gubra template
temp = bf.readNifti(os.path.join(gubra_template_dir,'gubra_template_olf.nii.gz'))
ano_large = bf.readNifti(os.path.join(study_folder_server,'ano_large_structures_olf_man_sym.nii.gz'))


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

for i in range(0,len(ano_ids)):
    temp_segm = np.zeros((ano_large.shape[0],ano_large.shape[1],ano_large.shape[2]))

    temp_segm[ano_large==ano_ids[i]] = temp[ano_large==ano_ids[i]]

    bf.saveNifti(temp_segm,os.path.join(study_folder_server,'gubra_'+ano_names[i]+'.nii.gz'))

## Allen CCFv3 template
# Give all labels alternative labels due to data type problem with elastix
ano_full = bf.readNifti(os.path.join(cffv3_dir,'ccfv3_ano_olf.nii.gz'))
labels_old = []
for i in range(0, ano_full.shape[0]):
    for j in range(0, ano_full.shape[1]):
        for k in range(0, ano_full.shape[2]):
            temp_label_old = ano_full[i,j,k]
            if temp_label_old not in labels_old:
                print(temp_label_old)
                labels_old.append(temp_label_old)

labels_array = np.empty((len(labels_old),2))
labels_array[:,0] = np.array(labels_old)
filler = np.arange(0,len(labels_old),1)
labels_array[:,1] = np.array(filler)
#index = np.arange(labels_array.shape[0])
#dict = np.put(labels_array,[index,1],filler)
df = pd.DataFrame(labels_array)
df.columns = ['true label', 'given label']
df.to_csv(os.path.join(study_folder_server,'label_encoding.csv'))

# Make a temporary label volume with new, given label iDs
df = pd.read_csv(os.path.join(study_folder_server,'label_encoding.csv'))
ano_full = bf.readNifti(os.path.join(cffv3_dir,'ccfv3_ano_olf.nii.gz'))
ano_full_new = np.zeros(ano_full.shape)
for i in range(0,len(df)):
    print(i)
    true_label = df.iloc[i]['true label']
    ano_full_new[ano_full==int(true_label)]=df.iloc[i]['given label']
bf.saveNifti(ano_full_new.astype('float32'),os.path.join(cffv3_dir,'ccfv3_ano_olf_encoded.nii.gz'))


temp = bf.readNifti(os.path.join(cffv3_dir,'ccfv3_template_olf.nii.gz'))
ano_large = bf.readNifti(os.path.join(cffv3_dir,'ano_large_structures_full_man_sym_olf.nii.gz'))
ano_full = bf.readNifti(os.path.join(cffv3_dir,'ccfv3_ano_olf_encoded.nii.gz'))

ano_ids = []
ano_names = []

ano_ids.append(1)
ano_ids.append(2)
ano_ids.append(3)
ano_ids.append(4)
ano_ids.append(5)
ano_ids.append(6)

ano_names.append('cortex')
ano_names.append('cerebral_nuclei')
ano_names.append('interbrain')
ano_names.append('cerebellum')
ano_names.append('hindbrain')
ano_names.append('septal')

for i in range(0,len(ano_ids)):
    temp_segm = np.zeros((ano_large.shape[0],ano_large.shape[1],ano_large.shape[2]))
    temp_segm_full = np.zeros((ano_large.shape[0],ano_large.shape[1],ano_large.shape[2]))

    temp_segm[ano_large==ano_ids[i]] = temp[ano_large==ano_ids[i]]
    temp_segm_full[ano_large==ano_ids[i]] = ano_full[ano_large==ano_ids[i]]

    bf.saveNifti(temp_segm,os.path.join(study_folder_server,'ccfv3_'+ano_names[i]+'.nii.gz'))
    bf.saveNifti(temp_segm_full.astype('float32'),os.path.join(study_folder_server,'anoFull_'+ano_names[i]+'.nii.gz'))


## Align ccfv3 annotations to the lsfm template
ano_ids = []
ano_names = []

ano_ids.append(1)
ano_ids.append(2)
ano_ids.append(3)
ano_ids.append(4)
ano_ids.append(5)
ano_ids.append(6)

ano_names.append('cortex')
ano_names.append('cerebral_nuclei')
ano_names.append('interbrain')
ano_names.append('cerebellum')
ano_names.append('hindbrain')
ano_names.append('septal')


ano_parts_bspline = sitk.ReadImage(os.path.join(cffv3_dir,'ano_large_structures_full_man_sym_olf.nii.gz'))
ano_parts_bspline = sitk.GetArrayFromImage(ano_parts_bspline)


for i in range(len(ano_ids)):

    temp_mask = np.ones(ano_parts_bspline.shape) # fixed space
    temp_mask[ano_parts_bspline!=ano_ids[i]] = 0

    final_sitk = sitk.GetImageFromArray(temp_mask.astype('uint8'))
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server,'mask_'+ano_names[i]+'.nii.gz'))

    # PRE-PROCESSING OF HINDBRAIN PARTS
    if ano_names[i] == 'hindbrain':
        ccfv3_hind = sitk.ReadImage(os.path.join(study_folder_server,'ccfv3_'+'hindbrain'+'.nii.gz'))
        ccfv3_hind = sitk.GetArrayFromImage(ccfv3_hind)

        ccfv3_hind_ano = sitk.ReadImage(os.path.join(study_folder_server,'anoFull_'+'hindbrain'+'.nii.gz'))
        ccfv3_hind_ano = sitk.GetArrayFromImage(ccfv3_hind_ano)

        # mask ventricular system
        df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular_encoded.csv'),header=None)

        for index, row in df.iterrows():
            ccfv3_hind[ccfv3_hind_ano==float(row[0])] = 0

        temp_niti_sitk = sitk.GetImageFromArray(ccfv3_hind.astype('uint8'))
        temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server,'ccfv3_'+'hindbrain'+'.nii.gz'))


        gubra_hind = sitk.ReadImage(os.path.join(study_folder_server,'gubra_'+'hindbrain'+'.nii.gz'))
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
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server,'gubra_'+'hindbrain'+'.nii.gz'))
    #############################


    # PRE-PROCESSING OF SEPTAL PARTS
    if ano_names[i] == 'septal':
        ccfv3_hind = sitk.ReadImage(os.path.join(study_folder_server,'ccfv3_'+'septal'+'.nii.gz'))
        ccfv3_hind = sitk.GetArrayFromImage(ccfv3_hind)

        ccfv3_hind_ano = sitk.ReadImage(os.path.join(study_folder_server,'anoFull_'+'septal'+'.nii.gz'))
        # ccfv3_hind_ano = sitk.GetArrayFromImage(ccfv3_hind_ano)

        ccfv3_hind[ccfv3_hind<18] = 0
        temp_niti_sitk = sitk.GetImageFromArray(ccfv3_hind.astype('uint8'))
        temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server,'ccfv3_'+'septal'+'.nii.gz'))

        gubra_hind = sitk.ReadImage(os.path.join(study_folder_server,'gubra_'+'septal'+'.nii.gz'))
        gubra_hind = sitk.GetArrayFromImage(gubra_hind)

        gubra_hind[gubra_hind<40] = 0
        temp_niti_sitk = sitk.GetImageFromArray(gubra_hind.astype('uint8'))
        temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
        sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server,'gubra_'+'septal'+'.nii.gz'))
    #############################

    elastix_path = r'INSERT PATH TO ELASTIX FOLDER'
    # AFFINE
    program = r'elastix -threads 16 '; # elastix is added to PATH
    fixed_name = r'-f ' + os.path.join(study_folder_server,'gubra_'+ano_names[i]+'.nii.gz ');
    moving_name = r'-m ' + os.path.join(study_folder_server,'ccfv3_'+ano_names[i]+'.nii.gz ')
    outdir = r'-out ' + elastix_path + r'/elastix/workingDir ';
    params = r'-p ' + elastix_path + r'/elastix/Par0000affine_parts.txt ';
    fmask = r'-fmask ' + os.path.join(study_folder_server,'mask_'+ano_names[i]+'.nii.gz')

    os.system(program + fixed_name + moving_name  + outdir + params + fmask)
    move(elastix_path + r'/elastix/workingDir/result.0.nii.gz', os.path.join(study_folder_server,ano_names[i]+'_affine.nii.gz'))
    move(elastix_path + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server,ano_names[i]+'_affine.txt'))

    ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
    temp = sitk.ReadImage(os.path.join(study_folder_server,ano_names[i]+'_affine.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)
    temp[temp<0] = 0

    final = temp
    final = final.astype('uint8')
    final_sitk = sitk.GetImageFromArray(final)
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server,ano_names[i]+'_affine.nii.gz'))


    # BSPLINE
    if ano_names[i] == 'septal':
        params = r'-p ' + elastix_path + r'/elastix/Par0000bspline_parts_septal.txt ';
    elif ano_names[i] == 'hindbrain':
        params = r'-p ' + elastix_path + r'/elastix/Par0000bspline_parts_hindbrain.txt ';
    else:
        params = r'-p ' + elastix_path + r'/elastix/Par0000bspline_parts.txt ';


    t0 = r'-t0 ' + os.path.join(study_folder_server,ano_names[i]+'_affine.txt ')

    os.system(program + fixed_name + moving_name  + outdir + params + t0 + fmask)

    move(elastix_path + r'/elastix/workingDir/result.0.nii.gz', os.path.join(study_folder_server,ano_names[i]+'_bspline.nii.gz'))
    move(elastix_path + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server,ano_names[i]+'_bspline.txt'))

    ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
    temp = sitk.ReadImage(os.path.join(study_folder_server,ano_names[i]+'_bspline.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)
    temp[temp<0] = 0

    final = temp
    final = final.astype('uint8')
    final_sitk = sitk.GetImageFromArray(final)
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server,ano_names[i]+'_bspline.nii.gz'))

    # Move full annotations along
    program = r'PATH TO TRANSFORMIX ';
    moving_name = r'-in ' + os.path.join(study_folder_server,'anoFull_'+ano_names[i]+'.nii.gz ')
    outdir = r'-out ' + elastix_path + r'/elastix/workingDir ';
    trans_params = r'-tp ' + os.path.join(study_folder_server,ano_names[i]+'_bspline.txt');

    #### Change Affine interpolation order to 0 for transforming annotation file
    with open(os.path.join(study_folder_server,ano_names[i]+'_affine.txt'), 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    data[-8] = '(FinalBSplineInterpolationOrder 0) \n'
    ## and write everything back
    with open(os.path.join(study_folder_server,ano_names[i]+'_affine.txt'), 'w') as file:
        file.writelines( data )

    #### Change B-Spline interpolation order to 0 for transforming annotation file
    with open(os.path.join(study_folder_server,ano_names[i]+'_bspline.txt'), 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    data[-8] = '(FinalBSplineInterpolationOrder 0) \n'
    ## and write everything back
    with open(os.path.join(study_folder_server,ano_names[i]+'_bspline.txt'), 'w') as file:
        file.writelines( data )

    # Perform transformation
    os.system(program + moving_name + outdir + trans_params)

    move(elastix_path + r'/elastix/workingDir/result.nii.gz', os.path.join(study_folder_server,ano_names[i]+'_bspline_anoFull.nii.gz'))

    ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
    temp = sitk.ReadImage(os.path.join(study_folder_server,ano_names[i]+'_bspline_anoFull.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)

    final = temp
    final = final.astype('uint32')
    final_sitk = sitk.GetImageFromArray(final)
    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
    sitk.WriteImage(final_sitk,os.path.join(study_folder_server,ano_names[i]+'_bspline_anoFull.nii.gz'))



## Collect the parts
ano_ids = []
ano_names = []
#
ano_ids.append(1)
ano_ids.append(2)
ano_ids.append(3)
ano_ids.append(4)
ano_ids.append(5)
ano_ids.append(6)

ano_names.append('cortex')
ano_names.append('cerebral_nuclei')
ano_names.append('interbrain')
ano_names.append('cerebellum')
ano_names.append('hindbrain')
ano_names.append('septal')

gubra_avg = bf.readNifti(os.path.join(gubra_template_dir,'gubra_template_olf.nii.gz'))
collected = np.zeros(gubra_avg.shape)
collected_ano = np.zeros(gubra_avg.shape)

ano_parts_bspline = sitk.ReadImage(os.path.join(study_folder_server,'ano_large_structures_olf_man_sym.nii.gz'))
ano_parts_bspline = sitk.GetArrayFromImage(ano_parts_bspline)

for i in range(len(ano_ids)):
    temp = sitk.ReadImage(os.path.join(study_folder_server,ano_names[i]+'_bspline.nii.gz'))
    temp = sitk.GetArrayFromImage(temp)

    temp_ano = sitk.ReadImage(os.path.join(study_folder_server,ano_names[i]+'_bspline_anoFull.nii.gz'))
    temp_ano = sitk.GetArrayFromImage(temp_ano)

    collected[ano_parts_bspline==ano_ids[i]] = temp[ano_parts_bspline==ano_ids[i]]
    collected_ano[ano_parts_bspline==ano_ids[i]] = temp_ano[ano_parts_bspline==ano_ids[i]]

final_sitk = sitk.GetImageFromArray(collected.astype('uint8'))
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server,'ccfv3_parts_bspline.nii.gz'))

final_sitk = sitk.GetImageFromArray(collected_ano.astype('uint32'))
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server,'ccfv3_parts_bspline_ano.nii.gz'))


# Gubra tissue mask
mask = np.zeros(ano_parts_bspline.shape)
mask[ano_parts_bspline>0]=1
bf.saveNifti(mask.astype('uint8'),os.path.join(gubra_template_dir,'gubra_tissue_mask_olf.nii.gz'))

################ FILL IN BORDER GAPS BY DISTANCE/FEATURE MAP COMPUTATION (ANO + TEMPLATE) #############################
collected_ano = bf.readNifti(os.path.join(study_folder_server,'ccfv3_parts_bspline_ano.nii.gz'))

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
tissue_mask = bf.readNifti(os.path.join(gubra_template_dir,'gubra_tissue_mask_olf.nii.gz'))

collected_ano[tissue_mask==0] = 0
final_sitk = sitk.GetImageFromArray(collected_ano.astype('uint32'))
final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
sitk.WriteImage(final_sitk,os.path.join(study_folder_server,'ccfv3_parts_bspline_ano_final.nii.gz'))


## Save ventricular mask
ano_full = sitk.ReadImage(os.path.join(study_folder_server,'ccfv3_parts_bspline_ano_final.nii.gz'))
ano_full = sitk.GetArrayFromImage(ano_full)

# Make new volume
ven_white = np.zeros(ano_full.shape)

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular_encoded.csv'),header=None)

for index, row in df.iterrows():
    ven_white[ano_full==row[0]] = 1

print('Done ventricular')

temp_niti_sitk = sitk.GetImageFromArray(ven_white.astype('uint8'))
temp_niti_sitk.SetOrigin((round(temp_niti_sitk.GetWidth()/2), round(temp_niti_sitk.GetHeight()/2), -round(temp_niti_sitk.GetDepth()/2)))
sitk.WriteImage(temp_niti_sitk,os.path.join(study_folder_server,'ccfv3_parts_bspline_ventricular.nii.gz'))


# Use manually corrected ventricular mask to change any non-ventricular labels inside the ventricular system
#to nearest ventricular label
ano = bf.readNifti(os.path.join(study_folder_server,'ccfv3_parts_bspline_ano_final.nii.gz'))
ventr_mask = bf.readNifti(os.path.join(gubra_template_dir, 'gubra_ventricular_mask_olf.nii.gz'))

ano_new = np.copy(ano)
ano_new[ventr_mask!=1] = 0
ano_new[ventr_mask==1] = 0

df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular_encoded.csv'),header=None)

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
ventr_mask = bf.readNifti(os.path.join(gubra_template_dir, 'gubra_ventricular_mask_olf.nii.gz'))

ano_new = np.copy(ano)
ano_new[ventr_mask!=1] = 0
ano_new[ventr_mask==1] = 0

# load large structure excel sheet
df = pd.read_csv(os.path.join(r'INSERT PATH TO CSV FILES','ventricular_encoded.csv'),header=None)

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
bf.saveNifti(ano, os.path.join(study_folder_server,'ccfv3_parts_bspline_ventricular_man_corr.nii.gz'))


## Encode the labels back to the Allen labels
# Make a temporary label volume with new, given label iDs
df = pd.read_csv(os.path.join(study_folder_server,'label_encoding.csv'))
ano_full = bf.readNifti(os.path.join(study_folder_server,'ccfv3_parts_bspline_ventricular_man_corr.nii.gz'))
ano_full_new = np.zeros(ano_full.shape)
for i in range(0,len(df)):
    print(i)
    given_label = df.iloc[i]['given label']
    ano_full_new[ano_full==int(given_label)]=df.iloc[i]['true label']
bf.saveNifti(ano_full_new.astype('uint32'),os.path.join(study_folder_server,'ccfv3_parts_bspline_ventricular_man_corr_true_labels.nii.gz'))


## Check the labels
ano_full = bf.readNifti(os.path.join(study_folder_server,'ccfv3_parts_bspline_ventricular_man_corr_true_labels.nii.gz'))
labels_old = []
for i in range(0, ano_full.shape[0]):
    for j in range(0, ano_full.shape[1]):
        for k in range(0, ano_full.shape[2]):
            temp_label_old = ano_full[i,j,k]
            if temp_label_old not in labels_old:
                print(temp_label_old)
                labels_old.append(temp_label_old)

labels_array = np.empty((len(labels_old),1))
labels_array[:,0] = np.array(labels_old)
df = pd.DataFrame(labels_array)
df.columns = ['true label']
df.to_csv(os.path.join(study_folder_server,'label_check.csv'))
