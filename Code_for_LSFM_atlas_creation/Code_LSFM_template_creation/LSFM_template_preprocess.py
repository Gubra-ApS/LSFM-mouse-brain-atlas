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
study_folder_server = r'INSERT PATH TO THE OUTPUT FOLDER '

# Selected reference brain from study
reference_brain_id = r'INSERT REFERENCE SAMPLE ID'


# Define path and sample names
sample_id = []
full_brain_udrive = []
sample_flip = []

#### SAMPLES
# # # # reference brain
sample_id.append(r'INSERT REFERENCE SAMPLE ID')
full_brain_udrive.append(r'INSERT FULL PATH TO REFERENCE SAMPLE FOLDER')
sample_flip.append('0') # 0 for no flipping, 1 for flipping

# # # # other brain samples
sample_id.append(r'INSERT SAMPLE ID')
sample_flip.append('0')
full_brain_udrive.append(r'INSERT FULL PATH TO SAMPLE FOLDER')

#### NOTE: Working directory and paramter files for registrations
# ### CREATE A FOLDER "elastix" IN THE "study_folder_server"
# ### ADD A FOLDER "workingDir" along with affine and Bspline registration parameter files in the "elastix" folder


# Analyze samples
for i in range(len(sample_id)):
    if sample_flip[i] == '1' or sample_flip[i] == '0':

        print(sample_id[i])

        ### Read auto and spec of full Brain
        auto = []
        file_names = os.listdir(full_brain_udrive[i])
        file_names.sort()

        nb_files = len(file_names)
        file_loop = range(0,nb_files)

        # read all files
        for idx,val in enumerate(file_loop):
            if os.path.isfile(os.path.join(full_brain_udrive[i], file_names[idx])):
                if os.path.join(full_brain_udrive[i], file_names[idx]).split('.')[-1]=='tif':
                    if 'C00' in file_names[idx]:
                        #print(idx)
                        # read image with scikit
                        im = imread(os.path.join(full_brain_udrive[i], file_names[idx]),as_grey=False, plugin='pil')
                        # resize image to allow efficient registration
                        image_rescaled = rescale(im, 4.794/20) # think abut interpolation order and anti-aliasing
                        image_rescaled_uint = img_as_uint(image_rescaled)

                        auto.append(image_rescaled_uint)

        # Convert list to array
        auto = np.array(auto)
        print(auto.shape)

        auto = downscale_local_mean(auto, (2,1,1))

        # Pre-processing
        #CREATE CORSE TISSUE MASK
        cutoff = np.percentile(auto,5)
        print(cutoff)

        mask = np.zeros(auto.shape,'uint8')
        mask[auto>cutoff] = 1

        bf.saveNifti(auto, study_folder_server+r'/nifti/auto.nii.gz')
        bf.saveNifti(mask, study_folder_server+r'/nifti/mask.nii.gz')


        #### BIAS FIELD CORRECTION #####
        program = r'INSERT PATH TO ../run_Bias_field_correction.sh '
        runtime_path = r'INSERT PATH TO ../MATLAB/MATLAB_Runtime/v93 '
        vol_in =  study_folder_server+r'/nifti/auto.nii.gz '
        mask_in = study_folder_server+r'/nifti/mask.nii.gz '
        cor_out = study_folder_server+r'/nifti/auto_biascor.nii.gz '
        origin_out = study_folder_server+r'/nifti/origin.nii.gz '

        print(program + runtime_path + vol_in  + mask_in + cor_out + origin_out)
        os.system(program + runtime_path + vol_in  + mask_in + cor_out + origin_out)

        corr = bf.readNifti(study_folder_server+r'/nifti/auto_biascor.nii.gz')

        # RESCALE TO 8 BIT
        scale_limit = np.percentile(corr, (99.999))
        corr = skimage.exposure.rescale_intensity(corr, in_range=(0,scale_limit), out_range='uint8')
        auto = np.copy(corr)

        # rescale intensity based on mean/median of tissue
        auto_temp = np.copy(auto)
        scale_thres = skimage.filters.threshold_otsu(auto_temp)
        auto_temp[auto_temp<scale_thres] = 0
        nz_mean = np.mean(auto_temp[auto_temp>0].flatten())
        print(nz_mean)
        scale_fact = 120/nz_mean
        auto = auto*scale_fact
        print(np.max(auto))
        auto[auto>255] = 255
        auto= auto.astype('uint8')


        # CLAHE IN THREE ORTHOGONAL PLANES
        auto_temp_hor = np.copy(auto)
        auto_temp_cor = np.copy(auto)
        auto_temp_sag = np.copy(auto)

        for h in range(auto_temp_cor.shape[1]):
            temp_img = auto_temp_cor[:,h,:]

            temp_img[0:2,0:2] = 250
            temp_img[3:5,3:5] = 0

            clahe_im = skimage.exposure.equalize_adapthist(temp_img, kernel_size=(int(temp_img.shape[0]/3),int(temp_img.shape[1]/6)), clip_limit=0.01, nbins=255)
            clahe_im[0:2,0:2] = 0

            clahe_im = clahe_im*255
            clahe_im[clahe_im<0] = 0
            clahe_im = np.uint8(clahe_im)
            auto_temp_cor[:,h,:] = clahe_im

        for h in range(auto_temp_hor.shape[0]):
            temp_img = auto_temp_hor[h,:,:]

            temp_img[0:2,0:2] = 250
            temp_img[3:5,3:5] = 0

            clahe_im = skimage.exposure.equalize_adapthist(temp_img, kernel_size=(int(temp_img.shape[0]/3),int(temp_img.shape[1]/6)), clip_limit=0.01, nbins=255)
            clahe_im[0:2,0:2] = 0

            clahe_im = clahe_im*255
            clahe_im[clahe_im<0] = 0
            clahe_im = np.uint8(clahe_im)
            auto_temp_hor[h,:,:] = clahe_im

        for h in range(auto_temp_sag.shape[2]):
            temp_img = auto_temp_sag[:,:,h]

            temp_img[0:2,0:2] = 250
            temp_img[3:5,3:5] = 0

            clahe_im = skimage.exposure.equalize_adapthist(temp_img, kernel_size=(int(temp_img.shape[0]/3),int(temp_img.shape[1]/6)), clip_limit=0.01, nbins=255)
            clahe_im[0:2,0:2] = 0

            clahe_im = clahe_im*255
            clahe_im[clahe_im<0] = 0
            clahe_im = np.uint8(clahe_im)
            auto_temp_sag[:,:,h] = clahe_im

        # combine angles
        clahe_all = np.zeros((auto_temp_sag.shape[0],auto_temp_sag.shape[1],auto_temp_sag.shape[2],3))
        clahe_all[:,:,:,0] = auto_temp_hor
        clahe_all[:,:,:,1] = auto_temp_cor
        clahe_all[:,:,:,2] = auto_temp_sag
        clahe_all_mean = np.mean(clahe_all,3)

        # combine with original volume
        clahe_final = np.zeros((auto_temp_sag.shape[0],auto_temp_sag.shape[1],auto_temp_sag.shape[2],2))
        clahe_final[:,:,:,0] = clahe_all_mean
        clahe_final[:,:,:,1] = auto

        clahe_final = np.mean(clahe_final,3)


        # # HISTOGRAM MATCHING
        reference = bf.readNifti(os.path.join(study_folder_server+r'/nifti',reference_brain_id+'_regi.nii.gz'))
        clahe_final = skimage.transform.match_histograms(clahe_final.astype('float'), reference.astype('float'), multichannel=False)

        # flip if needed
        if sample_flip[i] == '1':
            # flip brain
            clahe_final = np.flip(clahe_final,2)
            print('Flipping sample...')


        # save volume as nifty file
        clahe_final = np.uint8(clahe_final)
        final_sitk = sitk.GetImageFromArray(clahe_final)
        final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
        sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_regi.nii.gz'))
        print('Brain ready for registration')


        # #### UNCOMMENT IF CORTEX OR OLFACTORY VOLUME
        #
        # ## AFFINE
        # program = r'elastix -threads 16 '; #elastix is added to PATH
        # fixed_name = r'-f ' + os.path.join(study_folder_server+r'/nifti',reference_brain_id+'_regi_orig.nii.gz ');
        # moving_name = r'-m ' + os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_regi_orig.nii.gz ');
        # outdir = r'-out ' + study_folder_server + r'/elastix/workingDir ';
        # params = r'-p ' + study_folder_server + r'/elastix/Affine_Gubra_June2019.txt ';
        #
        # os.system(program + fixed_name + moving_name  + outdir + params)
        # move(study_folder_server + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_affine.txt'))
        #
        #
        # # BSPLINE
        # params = r'-p ' + study_folder_server + r'/elastix/Bspline_Gubra_June2019.txt ';
        # t0 = r'-t0 ' + os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_affine.txt')
        #
        # os.system(program + fixed_name + moving_name  + outdir + params + t0)
        #
        # move(study_folder_server + r'/elastix/workingDir/result.0.nii.gz', os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline_orig.nii.gz'))
        # move(study_folder_server + r'/elastix/workingDir/TransformParameters.0.txt', os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline.txt'))
        #
        # ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
        # temp = sitk.ReadImage(os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline_orig.nii.gz'))
        # temp = sitk.GetArrayFromImage(temp)
        #
        # final = temp
        # final = final.astype('uint8')
        # final_sitk = sitk.GetImageFromArray(final)
        # final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))
        # sitk.WriteImage(final_sitk,os.path.join(study_folder_server+r'/nifti',sample_id[i]+'_bspline.nii.gz'))
