## Import libraries
import os
from PIL import Image
import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from scipy import ndimage
import scipy.ndimage.interpolation as sci_int
import scipy.ndimage.morphology as sci_morph
from skimage.morphology import disk, dilation

import skimage
from skimage import data, color, img_as_uint
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imshow
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max
from skimage import morphology
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.restoration import (denoise_tv_chambolle)
from skimage.measure import label, regionprops

import sys
from subprocess import call, run
import subprocess
from math import sqrt
import SimpleITK as sitk
from shutil import move
import math
#import seaborn as sns
import pandas as pd

import cv2
import scipy.ndimage

from skimage.morphology import ball

# Keras:
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K



class Huginn:
    def __init__(self, moving, fixed, elastix_path, result_path):
        self.moving = moving
        self.fixed = fixed
        self.elastix_path = elastix_path
        self.result_path = result_path

    def registration(self, params, result_name, init_trans=r'', f_mask=r'', save_nifti=True, datatype='uint16', origin_vol=False):
        program = r'elastix -threads 16 '; #elastix is added to PATH
        fixed_name = r'-f ' + self.fixed + r' ';
        moving_name = r'-m ' + self.moving + r' ';
        outdir = r'-out ' + self.elastix_path + r'/workingDir ';
        params = r'-p ' + self.elastix_path + r'/' + params + r' ';

        if f_mask!=r'':
            fmask =  r'-fmask ' + f_mask
            if init_trans != r'':
                t0 = r'-t0 ' + os.path.join(self.result_path,init_trans + r'.txt ')
                os.system(program + fixed_name + moving_name  + outdir + params + t0 + fmask)
            else:
                os.system(program + fixed_name + moving_name  + outdir + params + fmask)
        else:
            if init_trans != r'':
                t0 = r'-t0 ' + os.path.join(self.result_path,init_trans + r'.txt ')
                os.system(program + fixed_name + moving_name  + outdir + params + t0)
            else:
                print(program + fixed_name + moving_name  + outdir + params)
                os.system(program + fixed_name + moving_name  + outdir + params)

        move(self.elastix_path + r'/workingDir/TransformParameters.0.txt', os.path.join(self.result_path,result_name + r'.txt'))

        if save_nifti==True:
            move(self.elastix_path + r'/workingDir/result.0.nii.gz', os.path.join(self.result_path,result_name + r'.nii.gz'))

            ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
            temp = sitk.ReadImage(os.path.join(self.result_path,result_name + r'.nii.gz'))
            temp = sitk.GetArrayFromImage(temp)
            temp[temp<0] = 0
            if datatype=='uint16':
                temp[temp>65535] = 65535
                temp = temp.astype('uint16')
            elif datatype=='uint8':
                temp[temp>255] = 255
                temp = temp.astype('uint8')

            final_sitk = sitk.GetImageFromArray(temp)
            if isinstance(origin_vol, bool)==False:
                sitk_origin =sitk.GetImageFromArray(origin_vol)
                final_sitk.SetOrigin((round(sitk_origin.GetWidth()/2), round(sitk_origin.GetHeight()/2), -round(sitk_origin.GetDepth()/2)))
            else:
                final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))

            sitk.WriteImage(final_sitk,os.path.join(self.result_path,result_name + r'.nii.gz'))

        return True

    def transform_vol(self, volume, trans_params, result_name, type='intensity', datatype='uint16', origin_vol=False):
        program = r'/home/cgs/src/Elastix/bin/transformix ';
        moving_name = r'-in ' + volume + r' ';
        outdir = r'-out ' + self.elastix_path + r'/workingDir ';
        tp = r'-tp ' + os.path.join(self.result_path,trans_params + r'.txt ');

        # Set interpolation to intensity data as default
        with open(os.path.join(self.result_path,trans_params + r'.txt'), 'r') as file:
            # read a list of lines into data
            data = file.readlines()

        data[-8] = "(FinalBSplineInterpolationOrder 0) \n"

        ## and write everything back
        with open(os.path.join(self.result_path,trans_params + r'.txt'), 'w') as file:
            file.writelines( data )

        ####
        if type=='ano':
            with open(os.path.join(self.result_path,trans_params + r'.txt'), 'r') as file:
                # read a list of lines into data
                data = file.readlines()

            data[-8] = "(FinalBSplineInterpolationOrder 0) \n"

            ## and write everything back
            with open(os.path.join(self.result_path,trans_params + r'.txt'), 'w') as file:
                file.writelines( data )

        # transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
        print(program + moving_name + outdir + tp)
        #exit()
        os.system(program + moving_name + outdir + tp)

        move(self.elastix_path + r'/workingDir/result.nii.gz', os.path.join(self.result_path,result_name + '.nii.gz'))

        ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
        temp = sitk.ReadImage(os.path.join(self.result_path,result_name + '.nii.gz'))
        temp = sitk.GetArrayFromImage(temp)

        temp[temp<0] = 0
        if datatype=='uint16':
            temp[temp>65535] = 65535
            temp = temp.astype('uint16')
        elif datatype=='uint8':
            temp[temp>255] = 255
            temp = temp.astype('uint8')

        final_sitk = sitk.GetImageFromArray(temp)
        if isinstance(origin_vol, bool)==False:
            sitk_origin =sitk.GetImageFromArray(origin_vol)
            final_sitk.SetOrigin((round(sitk_origin.GetWidth()/2), round(sitk_origin.GetHeight()/2), -round(sitk_origin.GetDepth()/2)))
        else:
            final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))

        sitk.WriteImage(final_sitk,os.path.join(self.result_path,result_name + r'.nii.gz'))


        return os.path.join(self.result_path,result_name + r'.nii.gz')


    def transform_points(self, points, input_image, trans_params, result_name):
        program = r'/home/cgs/src/Elastix/bin/transformix ';
        input_points = r'-def ' + points + ' ';
        outdir = r'-out ' + self.elastix_path + r'/workingDir ';
        inputImage = r'-in ' + input_image + ' ';
        trans_params = r'-tp ' + os.path.join(self.result_path,trans_params + r'.txt ');

        #transformix -def inputPoints.txt -out outputDirectory -tp TransformParameters.txt
        print(program + input_points  + outdir + inputImage + trans_params)
        os.system(program + input_points  + outdir + inputImage + trans_params)

        move(self.elastix_path + r'/workingDir/outputpoints.txt', os.path.join(self.result_path,result_name + '.txt'))

        return os.path.join(self.result_path,result_name + '.txt')


class Muninn:
    # Initializer / Instance Attributes
    def __init__(self, auto):
        """
        Initialize class with:
            -self.auto: Brain imaged in autofluorescence channel as numpy array
        """
        # adjust dimensions:
        if len(auto.shape) < 3:
            self.auto = auto.reshape(1,auto.shape[0],auto.shape[1])
        else:
            self.auto = auto

    # set up unet:
    def unet():
        """
        Set up U-net model
        # Arguments
            None
        # Returns
            model: Keras model of U-net
        """
        # define dice coefficient:----------------------------------------------

        def dice_coef(y_true, y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

        def dice_coef_loss(y_true, y_pred):
            return 1-dice_coef(y_true, y_pred)


        # define network:-------------------------------------------------------
        inputs = Input((None, None, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        model.compile(optimizer=Adam(lr = 0.0001), loss=dice_coef_loss, metrics=[dice_coef])

        return model

    # predict segmentation:
    def pred_segmentation(self,weights):
        """
        Predict segmentation using U-net
        # Arguments
            self.auto: Brain imaged in autofluorescence channel as numpy array
            weights: weights of trained model in hdf5 format
        # Returns
            mask: segmentation mask (numpy array)
        """

        # set up network:
        model=Muninn.unet()

        # pad input with zeros:
        dim_x=(int(self.auto.shape[1]/32.0)+1 ) *32
        dim_y=(int(self.auto.shape[2]/32.0)+1 ) *32
        inp=np.zeros((self.auto.shape[0],dim_x, dim_y,1))
        inp[:self.auto.shape[0],:self.auto.shape[1],:self.auto.shape[2],:1]=self.auto.reshape((self.auto.shape[0],self.auto.shape[1],self.auto.shape[2],1))

        # predict segmentation mask:
        model.load_weights(weights)
        mask=model.predict(inp, batch_size=1)
        mask=mask[:self.auto.shape[0],:self.auto.shape[1],:self.auto.shape[2],0]

        # binarize predictions:
        mask[mask<0.5]=0.0
        mask[mask>=0.5]=1.0

        return mask


def readRawTiffs(path, scale_z=1, scale_xy=1, channels='both'):
    # channels: both, auto, spec
    ### Read auto and spec of full Brain
    file_names = os.listdir(path)
    file_names.sort()

    nb_files = len(file_names)
    print(nb_files)
    file_loop = range(0,nb_files)

    if channels=='both':
        auto = []
        spec = []
    elif channels=='auto':
        auto = []
    elif channels=='spec':
        spec = []

    # read all files
    for idx,val in enumerate(file_loop):
        if os.path.isfile(os.path.join(path, file_names[idx])):
            if os.path.join(path, file_names[idx]).split('.')[-1]=='tif':
                if channels=='both':
                    if 'C00' in file_names[idx]:
                        # read image with scikit
                        im = imread(os.path.join(path, file_names[idx]),as_gray=False, plugin='pil')
                        if scale_xy != 1:
                            # resize image to allow efficient registration
                            im = rescale(im, scale_xy) # think abut interpolation order and anti-aliasing

                        image_rescaled_uint = img_as_uint(im)
                        auto.append(image_rescaled_uint)

                    if 'C01' in file_names[idx]:
                        # read image with scikit
                        im = imread(os.path.join(path, file_names[idx]),as_gray=False, plugin='pil')
                        if scale_xy != 1:
                            # resize image to allow efficient registration
                            im = rescale(im, scale_xy) # think abut interpolation order and anti-aliasing

                        image_rescaled_uint = img_as_uint(im)
                        spec.append(image_rescaled_uint)
                elif channels=='auto':
                    if 'C00' in file_names[idx]:
                        # read image with scikit
                        im = imread(os.path.join(path, file_names[idx]),as_gray=False, plugin='pil')
                        if scale_xy != 1:
                            # resize image to allow efficient registration
                            im = rescale(im, scale_xy) # think abut interpolation order and anti-aliasing

                        image_rescaled_uint = img_as_uint(im)
                        auto.append(image_rescaled_uint)
                elif channels=='spec':
                    if 'C01' in file_names[idx]:
                        # read image with scikit
                        im = imread(os.path.join(path, file_names[idx]),as_gray=False, plugin='pil')
                        if scale_xy != 1:
                            # resize image to allow efficient registration
                            im = rescale(im, scale_xy) # think abut interpolation order and anti-aliasing

                        image_rescaled_uint = img_as_uint(im)
                        spec.append(image_rescaled_uint)


    # Convert list to array
    if channels=='both':
        auto = np.array(auto)
        spec = np.array(spec)
        print(spec.shape)

        if scale_z != 1:
            auto = downscale_local_mean(auto, (scale_z,1,1))
            spec = downscale_local_mean(spec, (scale_z,1,1))

        print(auto.shape)
        print(spec.shape)

    elif channels=='auto':
        auto = np.array(auto)

        if scale_z != 1:
            auto = downscale_local_mean(auto, (scale_z,1,1))

        print(auto.shape)

    elif channels=='spec':
        spec = np.array(spec)

        if scale_z != 1:
            spec = downscale_local_mean(spec, (scale_z,1,1))

        print(spec.shape)


    if channels=='both':
        return auto, spec
    elif channels=='auto':
        return auto
    elif channels=='spec':
        return spec

def readRawTiffsTag(path, scale_z=1, scale_xy=1, tag='C00'):
    # channels: both, auto, spec
    ### Read auto and spec of full Brain
    file_names = os.listdir(path)
    file_names.sort()

    nb_files = len(file_names)
    print(nb_files)
    file_loop = range(0,nb_files)

    auto = []

    # read all files
    for idx,val in enumerate(file_loop):
        if os.path.isfile(os.path.join(path, file_names[idx])):
            if os.path.join(path, file_names[idx]).split('.')[-1]=='tif':
                if tag in file_names[idx]:
                    # read image with scikit
                    im = imread(os.path.join(path, file_names[idx]),as_gray=False, plugin='pil')
                    if scale_xy != 1:
                        # resize image to allow efficient registration
                        im = rescale(im, scale_xy) # think abut interpolation order and anti-aliasing

                    image_rescaled_uint = img_as_uint(im)
                    auto.append(image_rescaled_uint)

    # Convert list to array
    auto = np.array(auto)

    if scale_z != 1:
        auto = downscale_local_mean(auto, (scale_z,1,1))

    print(auto.shape)

    return auto


def vol2Tiffs(volume, save_path, datatype='uint16'):
    # create  directoty if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(avg_veh.shape[0]):
        im = volume[i,:,:].astype(datatype)
        # save image
        skimage.external.tifffile.imsave(save_path,+'/z'+str(i+1)+'.tif', im)

    return True


def readNifti(filename):
    temp = sitk.ReadImage(filename)
    temp = sitk.GetArrayFromImage(temp)

    return temp


def saveNifti(vol, filename, origin_vol=False):
    final_sitk = sitk.GetImageFromArray(vol)
    if isinstance(origin_vol, bool)==False:
        sitk_origin =sitk.GetImageFromArray(origin_vol)
        final_sitk.SetOrigin((round(sitk_origin.GetWidth()/2), round(sitk_origin.GetHeight()/2), -round(sitk_origin.GetDepth()/2)))
    else:
        final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))

    sitk.WriteImage(final_sitk,filename)

    return True


def points2vol(vol_shape, points, dilate=0):
    # vol shape: shape of desired volume
    # points: list of (z,y,x) (z,row,col) coordinates
    # result_path: full path+filename of nifti file to be saved
    # dilate: option to expand the size of each points [int]

    coords_full = np.zeros(vol_shape)

    for i in range(len(points)):
        #if (int(points[i][0]) < coords_full.shape[0]-dilate) & (int(points[i][1]) < coords_full.shape[1]-dilate) & (int(points[i][2]) < coords_full.shape[2]-dilate):
        try:
            # put into volume
            if dilate==0:
                coords_full[int(points[i][0]),int(points[i][1]),int(points[i][2])] = coords_full[int(points[i][0]),int(points[i][1]),int(points[i][2])] + 1
            else:
                coords_full[int(points[i][0])-dilate:int(points[i][0])+dilate,int(points[i][1])-dilate:int(points[i][1])+dilate,int(points[i][2]-dilate):int(points[i][2])+dilate] = coords_full[int(points[i][0]-dilate):int(points[i][0])+dilate,int(points[i][1]-dilate):int(points[i][1])+dilate,int(points[i][2]-dilate):int(points[i][2])+dilate] + 1
        except:
            print('Coordinate out of volume dimensions..')
    return coords_full

def points2vol_encode(vol_shape, points, encode, dilate=0):
    # vol shape: shape of desired volume
    # points: list of (z,y,x) (z,row,col) coordinates
    # result_path: full path+filename of nifti file to be saved
    # encode: List with values to encode at points

    coords_full = np.zeros(vol_shape,'uint8')

    for i in range(len(points)):
        if (int(points[i][0]) < coords_full.shape[0]-dilate) & (int(points[i][1]) < coords_full.shape[1]-dilate) & (int(points[i][2]) < coords_full.shape[2]-dilate):
            # put into volume
            if dilate==0:
                coords_full[int(points[i][0]),int(points[i][1]),int(points[i][2])] = coords_full[int(points[i][0]),int(points[i][1]),int(points[i][2])] + encode[i]
            else:
                coords_full[int(points[i][0]):int(points[i][0])+dilate,int(points[i][1]):int(points[i][1])+dilate,int(points[i][2]):int(points[i][2])+dilate] = coords_full[int(points[i][0]):int(points[i][0])+dilate,int(points[i][1]):int(points[i][1])+dilate,int(points[i][2]):int(points[i][2])+dilate] + encode[i]
        else:
            print('Coordinate out of volume dimensions..')
    return coords_full


def cell_detector(im, back_subtract_disk, max_filter_cube, watershed_threshold, cell_size_threshold, bg_thres):
    ###########################################################################
    ### STEP 1: background subtraction

    # perform background subtraction slice-by-slice
    dtype = im.dtype
    im = np.array(im, dtype = float)

    for z in range(im.shape[0]):
        im[z] = im[z] - cv2.morphologyEx(im[z], cv2.MORPH_OPEN, back_subtract_disk)

    im[im < 0] = 0
    im = np.array(im, dtype = dtype)

    im[im<bg_thres] = 0

    ###########################################################################
    ### STEP 2: detect local maxima and extract z-y-x coordinate list

    # maximum filter (use constant value outside borders to avoid finding too many maxima)
    mask_peak = scipy.ndimage.filters.maximum_filter(im, footprint=max_filter_cube, mode='constant', cval=2**16-1) == im

    # neat trick to assign coords to all non-zero voxels
    coords = np.nonzero(mask_peak)
    coords = np.vstack(coords).T

    ###########################################################################
    # STEP 3: establish connection between coords and watershed seeds
    weights = np.arange(1, coords.shape[0]+1, dtype='uint32')
    mask_seeds = np.zeros( im.shape, dtype=weights.dtype)

    for zyx, weight in zip(coords, weights):
        mask_seeds[zyx[0], zyx[1], zyx[2]] += weight

    ###########################################################################
    ### STEP 4: seeded watershed
    mask_thresh = im > watershed_threshold
    mask_ws = watershed(-im, mask_seeds, mask=mask_thresh)

    ###########################################################################
    ### STEP : filter away watershed regions that are too small or too large
    cell_size = scipy.ndimage.measurements.sum( np.ones( im.shape, dtype = bool), labels = mask_ws, index = weights )

    idx_keep = (cell_size > cell_size_threshold[0]) & (cell_size < cell_size_threshold[1])
    coords = coords[idx_keep, :]

    # return cell centre coordinates
    return coords


def cell_detector_v2(im, im_auto, back_subtract_disk, max_filter_cube, watershed_threshold, cell_size_threshold, bg_thres, bg_auto_thres):

    ###########################################################################
    ### STEP 1: background subtraction

    # perform background subtraction slice-by-slice
    dtype = im.dtype
    im = np.array(im, dtype = float)
    im_auto = np.array(im_auto, dtype = float)

    for z in range(im.shape[0]):
        im[z] = im[z] - cv2.morphologyEx(im[z], cv2.MORPH_OPEN, back_subtract_disk)

    for z in range(im_auto.shape[0]):
        im_auto[z] = im_auto[z] - cv2.morphologyEx(im_auto[z], cv2.MORPH_OPEN, back_subtract_disk)

    # im_debug_spec = np.copy(im)
    im[im < 0] = 0;

    # im_debug_auto = np.copy(im_auto)
    im_auto[im_auto < bg_auto_thres] = 0;
    im_auto[im_auto == 0] = 1; # avoid dividing by zero
    #im_debug_auto = np.copy(im_auto)

    # threshold feature
    im = im/im_auto
    #im = np.array(im, dtype = dtype)

    # im_debug = np.copy(im)
    im[im<bg_thres] = 0

    ###########################################################################
    ### STEP 2: detect local maxima and extract z-y-x coordinate list

    # maximum filter (use constant value outside borders to avoid finding too many maxima)
    mask_peak = scipy.ndimage.filters.maximum_filter(im, footprint=max_filter_cube, mode='constant', cval=2**16-1) == im

    # neat trick to assign coords to all non-zero voxels
    coords = np.nonzero(mask_peak)
    coords = np.vstack(coords).T


    ###########################################################################
    # STEP 3: establish connection between coords and watershed seeds
    weights = np.arange(1, coords.shape[0]+1, dtype='uint32')
    mask_seeds = np.zeros( im.shape, dtype=weights.dtype)

    for zyx, weight in zip(coords, weights):
        mask_seeds[zyx[0], zyx[1], zyx[2]] += weight


    ###########################################################################
    ### STEP 4: seeded watershed
    mask_thresh = im > watershed_threshold
    mask_ws = watershed(-im, mask_seeds, mask=mask_thresh)


    ###########################################################################
    ### STEP : filter away watershed regions that are too small or too large
    cell_size = scipy.ndimage.measurements.sum( np.ones( im.shape, dtype = bool), labels = mask_ws, index = weights )

    idx_keep = (cell_size > cell_size_threshold[0]) & (cell_size < cell_size_threshold[1])
    coords = coords[idx_keep, :]

    # return cell centre coordinates
    return coords


def cell_detector_debug(im, im_auto, back_subtract_disk, max_filter_cube, watershed_threshold, cell_size_threshold, bg_thres, bg_auto_thres):

    ###########################################################################
    ### STEP 1: background subtraction

    # perform background subtraction slice-by-slice
    dtype = im.dtype
    im = np.array(im, dtype = float)
    im_auto = np.array(im_auto, dtype = float)

    for z in range(im.shape[0]):
        im[z] = im[z] - cv2.morphologyEx(im[z], cv2.MORPH_OPEN, back_subtract_disk)

    for z in range(im_auto.shape[0]):
        im_auto[z] = im_auto[z] - cv2.morphologyEx(im_auto[z], cv2.MORPH_OPEN, back_subtract_disk)

    im_debug_spec = np.copy(im)
    im[im < 0] = 0;

    im_debug_auto = np.copy(im_auto)
    im_auto[im_auto < bg_auto_thres] = 0;
    im_auto[im_auto == 0] = 1; # avoid dividing by zero
    #im_debug_auto = np.copy(im_auto)

    # threshold feature
    im = im/im_auto
    #im = np.array(im, dtype = dtype)

    im_debug = np.copy(im)
    im[im<bg_thres] = 0
    


    ###########################################################################
    ### STEP 2: detect local maxima and extract z-y-x coordinate list

    # maximum filter (use constant value outside borders to avoid finding too many maxima)
    mask_peak = scipy.ndimage.filters.maximum_filter(im, footprint=max_filter_cube, mode='constant', cval=2**16-1) == im

    # neat trick to assign coords to all non-zero voxels
    coords = np.nonzero(mask_peak)
    coords = np.vstack(coords).T


    ###########################################################################
    # STEP 3: establish connection between coords and watershed seeds
    weights = np.arange(1, coords.shape[0]+1, dtype='uint32')
    mask_seeds = np.zeros( im.shape, dtype=weights.dtype)

    for zyx, weight in zip(coords, weights):
        mask_seeds[zyx[0], zyx[1], zyx[2]] += weight


    ###########################################################################
    ### STEP 4: seeded watershed
    mask_thresh = im > watershed_threshold
    mask_ws = watershed(-im, mask_seeds, mask=mask_thresh)


    ###########################################################################
    ### STEP : filter away watershed regions that are too small or too large
    cell_size = scipy.ndimage.measurements.sum( np.ones( im.shape, dtype = bool), labels = mask_ws, index = weights )

    idx_keep = (cell_size > cell_size_threshold[0]) & (cell_size < cell_size_threshold[1])
    coords = coords[idx_keep, :]

    # return cell centre coordinates
    return coords, im_debug_spec, im_debug_auto, im_debug, mask_ws


def plaque_detector_debug(im, im_auto, back_subtract_disk, max_filter_cube, watershed_threshold, cell_size_threshold, bg_thres, bg_auto_thres):

    ###########################################################################
    ### STEP 1: background subtraction

    # perform background subtraction slice-by-slice
    dtype = im.dtype
    im = np.array(im, dtype = float)
    im_auto = np.array(im_auto, dtype = float)

    for z in range(im.shape[0]):
        im[z] = im[z] - cv2.morphologyEx(im[z], cv2.MORPH_OPEN, back_subtract_disk)

    for z in range(im_auto.shape[0]):
        im_auto[z] = im_auto[z] - cv2.morphologyEx(im_auto[z], cv2.MORPH_OPEN, back_subtract_disk)

    im_debug_spec = np.copy(im)
    

    im_debug_auto = np.copy(im_auto)
    #im_auto[im_auto < bg_auto_thres] = 0;
    #im_auto[im_auto == 0] = 1; # avoid dividing by zero
    #im_debug_auto = np.copy(im_auto)

    # threshold feature
    im = im - im_auto
    im[im < 0] = 0
    #im = np.array(im, dtype = dtype)

    im_debug = np.copy(im)
    
    im[im<bg_thres] = 0
    
    ###########################################################################
    ### STEP 2: detect local maxima and extract z-y-x coordinate list

    # maximum filter (use constant value outside borders to avoid finding too many maxima)
    mask_peak = scipy.ndimage.filters.maximum_filter(im, footprint=max_filter_cube, mode='constant', cval=2**16-1) == im

    # neat trick to assign coords to all non-zero voxels
    coords = np.nonzero(mask_peak)
    coords = np.vstack(coords).T


    ###########################################################################
    # STEP 3: establish connection between coords and watershed seeds
    weights = np.arange(1, coords.shape[0]+1, dtype='uint32')
    mask_seeds = np.zeros( im.shape, dtype=weights.dtype)

    for zyx, weight in zip(coords, weights):
        mask_seeds[zyx[0], zyx[1], zyx[2]] += weight


    ###########################################################################
    ### STEP 4: seeded watershed
    mask_thresh = im > watershed_threshold
    mask_ws = watershed(-im, mask_seeds, mask=mask_thresh)


    ###########################################################################
    ### STEP : filter away watershed regions that are too small or too large
    cell_size = scipy.ndimage.measurements.sum( np.ones( im.shape, dtype = bool), labels = mask_ws, index = weights )

    idx_keep = (cell_size > cell_size_threshold[0]) & (cell_size < cell_size_threshold[1])
    coords = coords[idx_keep, :]

    # return cell centre coordinates
    return coords, im_debug_spec, im_debug_auto, im_debug, mask_ws





def vol2coords(vol):
    dtype = vol.dtype
    vol = np.array(vol, dtype = float)

    coords = np.nonzero(vol)
    coords = np.vstack(coords).T

    return coords


def rm_noise(pred, signal_mask, dilation_radius):
    """
    Noise removal of signal overlaying with predicted structure (i.e. blood vessels)
    # Arguments
        pred: predicted structure (i.e. blood vessels) as numpy array
        signal_mask: cFos signal mask (numpy array)
        dilation_radius: dilation radius for predicted structure (float)
    # Returns
        clean_signal_mask: cleaned cFos signal mask (numpy array)
    """
    clean_signal_mask=[]
    r_mask_debug=[]
    k_mask_debug=[]
    for i in range(0,pred.shape[0]):
        print(i)
        pred_curr=pred[i]
        pred_curr[pred_curr>1] = 1
        signal_mask_curr=signal_mask[i]
        signal_mask_curr[signal_mask_curr>1] = 1

        # dilate predictions:
        d_pred=dilation(pred_curr, selem=disk(dilation_radius), out=None)

        # binary signal mask:
        binary_mask=np.copy(signal_mask_curr)
        binary_mask[binary_mask>1.0]=1.0

        # get overlap of pred and signal:
        overlay=d_pred + binary_mask
        overlay[overlay<=1.0]=0.0
        overlay[overlay>1.0]=1.0


        # get centroids of overlay structure:
        lo=label(overlay)
        r=regionprops(lo)
        c=[(int(x.centroid[0]),int(x.centroid[1])) for x in r]

        # get label signal mask:
        rk_mask=label(binary_mask)

        # mark signal that overlaps with prediction:
        for p in c:
            x=p[0]
            y=p[1]

            # look up label value:
            if x < rk_mask.shape[1]-1:
                if y < rk_mask.shape[0]-1:
                    v=max(rk_mask[x,y], rk_mask[x+1,y], rk_mask[x,y+1], rk_mask[x+1,y+1], rk_mask[x-1,y], rk_mask[x,y-1], rk_mask[x-1,y-1])

                    # set to -1:
                    if v>0:
                        rk_mask[rk_mask==v]=-1
                else:
                    print(y)
            else:
                print(x)


        # split into remove and keep masks:
        r_mask=np.copy(rk_mask)
        k_mask=np.copy(rk_mask)
        k_mask[k_mask<0]=0.0
        k_mask[k_mask>0]=1.0
        r_mask[r_mask>0]=0.0
        r_mask[r_mask==-1.0]=1.0

        # update original signal heatmap:
        clean_signal_mask_curr=np.zeros((signal_mask_curr.shape))
        clean_signal_mask_curr= ((signal_mask_curr + k_mask) -1) - r_mask
        clean_signal_mask_curr[clean_signal_mask_curr<0.0]=0.0

        # save:
        clean_signal_mask.append(clean_signal_mask_curr)
        r_mask_debug.append(r_mask)
        k_mask_debug.append(k_mask)

    # convert to np array:
    clean_signal_mask=np.array(clean_signal_mask)
    r_mask_debug=np.array(r_mask_debug)
    k_mask_debug=np.array(k_mask_debug)

    return clean_signal_mask



def create_atlas_heatmap(vol_shape, atlas_coords, radius):

    sphere = ball(radius)
    heatmap = np.zeros( vol_shape, dtype='uint16' )

    for ac in atlas_coords:
        # does the sphere fit in our volume?
        if (ac[0]-radius > 0) & (ac[0]+radius < heatmap.shape[0]) & (ac[1]-radius > 0) & (ac[1]+radius < heatmap.shape[1]) & (ac[2]-radius > 0) & (ac[2]+radius < heatmap.shape[2]):
            heatmap[ (ac[0]-radius):(ac[0]+radius+1), (ac[1]-radius):(ac[1]+radius+1), (ac[2]-radius):(ac[2]+radius+1) ] += sphere

    return heatmap

def string2int(str_name):
    # May be used as a helper funtion to sort(key=string2int)
    # This will correctly sort e.g chpl_Z1.tiff, chpl_Z10.tiff, chpl_Z2.tiff
    part1, part2 = str_name.split('Z')
    part3, part4 = part2.split('.')
    return int(part3)

def unmix(auto, spec, nbins=30):
    ### Perform unmixing
    # Compute histogram for auto and spec
    #nbins = 30
    h0 = np.histogram(auto.ravel(), bins=nbins, range=(0,np.percentile(auto,99.99)) )

    # calculate ratio between all voxels represented by each histogram bin
    ratio = np.zeros(nbins)
    estimate = np.copy(auto)
    estimate = estimate.astype('float')

    for k in range( len(h0[0]) ):
        print(k)
        v0 = auto[ (auto > h0[1][k]) & (auto < h0[1][k+1]) ]
        v1 = spec[ (auto > h0[1][k]) & (auto < h0[1][k+1]) ]

        if (len(v0) > 0) and (len(v1) > 0):
            ratio[k] = np.nanmedian( np.divide(v0,v1) )
            estimate[ (auto > h0[1][k]) & (auto < h0[1][k+1]) ] = estimate[ (auto > h0[1][k]) & (auto < h0[1][k+1]) ] / ratio[k]

    unmix = spec.astype('float') - estimate.astype('float')
    unmix[unmix<0] = 0
    unmix = unmix.astype('uint16')

    return unmix

def vol2Imaris(vol, path):
    # exports 16 bit tiff stacks which can be loaded in Imaris
    # If the path does not exsists it will be created
    vol[vol>65535] = 65535
    vol[vol<0] = 0

    if not os.path.exists(path): os.makedirs(path)

    for k in range(vol.shape[0]):
        print(k)
        # uint16 convertion
        im = vol[k,:,:].astype('uint16')

        # save prediction image as uint16
        temp = "%04d" % (k,)

        skimage.external.tifffile.imsave(path+r'/z'+temp+'.tif', im)

def Imaris2vol(path):
    file_names = os.listdir(path)
    file_names.sort()

    nb_files = len(file_names)
    print(nb_files)
    file_loop = range(0,nb_files)

    vol = []

    # read all files
    for idx,val in enumerate(file_loop):
        if os.path.isfile(os.path.join(path, file_names[idx])):
            if os.path.join(path, file_names[idx]).split('.')[-1]=='tif':
                # read image with scikit
                im = imread(os.path.join(path, file_names[idx]),as_gray=False, plugin='pil')
                image_rescaled_uint = img_as_uint(im)
                vol.append(image_rescaled_uint)

    vol = np.array(vol)
    print(vol.shape)
    return vol

    
def t_point_algorithm(vec, n_bins, bin_range):
    
    h = np.histogram( vec, bins=n_bins, range=bin_range)

    # turn histogram into x,y coordinates
    yh = h[0]
    xh = ( h[1][1:] + h[1][:-1] ) / 2
    
    idx0 = np.argmax( yh == yh.max() ) # mode of histogram
    idx1 = len(h[0])
    
    eps1 = np.zeros( idx1-idx0-2 )
    eps2 = np.zeros( idx1-idx0-2 )
    
    for i,k in enumerate( range( idx0+1, idx1-1 ) ):
        
        #######################################################################
        # D1 fitting line
        
        # estiamte slope from maximum to current threshold (k)
        x1 = np.array( (xh[idx0], xh[k]) )
        y1 = np.array( (yh[idx0], yh[k]) )
        b1 = np.polyfit(x1, y1, 1)
        
        x    = xh[idx0:k]
        y    = yh[idx0:k]
        yhat = b1[0]*x + b1[1]
        eps1[i] = ((y-yhat)**2).sum()
        #######################################################################    
    
        #######################################################################
        # D2 fitting line
        
        # estiamte slope from maximum to current threshold (k)
        x2 = np.array( (xh[k], xh[idx1-1]) )
        y2 = np.array( (yh[k], yh[idx1-1]) )
        b2 = np.polyfit(x2, y2, 1)
    
        x    = xh[k:idx1]
        y    = yh[k:idx1]
        yhat = b2[0]*x + b2[1]
        eps2[i] = ((y-yhat)**2).sum()
        #######################################################################

    # determine optimal k
    k = xh[idx0+1:idx1-1]
    eps = eps1 + eps2
    k_opt = k[ np.argmax( eps == eps.min() ) ]
    
    return k_opt

