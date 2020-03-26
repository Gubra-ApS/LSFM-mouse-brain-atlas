# -*- coding: utf-8 -*-

### Import module
import os, sys, stat
from shutil import move
import glob
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.restoration import denoise_tv_chambolle
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data, color, img_as_uint
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from skimage.morphology import binary_dilation

from PIL import Image # for reading header information

from skimage.morphology import watershed, disk, ball

import pandas as pd

#### Import custom libraries
import sys
sys.path.insert(0, 'INSERT PATH TO THE FOLDER WHERE TOOLBOX IS SAVED')

import importlib
import toolbox as bf
importlib.reload(bf)
####



### Scan INPUT ################################################################
folder_input = 'INSERT PATH TO THE PARENT FOLDER WITH SCAN FOLDERS'

###############################################################################

### USER OUTPUT ################################################################
folder_output_path = 'INSERT PATH TO THE OUTPUT FOLDER'

###############################################################################


### ATLAS VERSION ################################################################
atlas_version = '_v3'
#atlas_version = '_CCFv3'

###############################################################################


### CELL DETECTION PARAMETERS ################################################################
bg_auto_thres = 350; bg_spec_thres = 800

###############################################################################


### edge removal radius ################################################################
edge_removal_radius = 4

###############################################################################


### ventricular removal radius ################################################################
ventricular_removal_radius = 4

###############################################################################


### Density map "cell size" ################################################################
density_map_radius = 5

###############################################################################

#### NOTE: Working directory and paramter files for registrations
# ### CREATE A FOLDER "elastix"
# ### ADD A FOLDER "workingDir" along with affine and Bspline registration parameter files in the "elastix" folder

# Elastix path
elastix_path = r'INSERT FULL PATH TO A FOLDER CALLED "elastix"'


# should cell detection validation be enabled
# if so, remember to change cFosFileRange in parameter file to limit file size
cell_check = 0
remove_chpl = 0

# additional parameters
ID_tag_length = 3
group_tag_length = 3

# Turn off plots
plt.ioff()


# create folders in the input folder
folder_parameters = os.path.join( folder_output_path, 'bifrost_parameter_files' )
folder_output = os.path.join( folder_output_path, 'bifrost_output' )
folder_maxproj_output = os.path.join( folder_output_path, 'pictures/max_projections' )
folder_histogram_output = os.path.join( folder_output_path, 'pictures/histograms' )
folder_nifti_output = os.path.join( folder_output_path, 'nifti' )
folder_coords_output = os.path.join( folder_output_path, 'coords' )
if not os.path.exists( folder_parameters ): os.makedirs( folder_parameters )
if not os.path.exists( folder_output ): os.makedirs( folder_output )
if not os.path.exists( folder_maxproj_output ): os.makedirs( folder_maxproj_output )
if not os.path.exists( folder_histogram_output ): os.makedirs( folder_histogram_output )
if not os.path.exists( folder_nifti_output ): os.makedirs( folder_nifti_output )
if not os.path.exists( folder_coords_output ): os.makedirs( folder_coords_output )
#
#
# Search for scan folders
folders = glob.glob( os.path.join( folder_input, '*ID*') ) # ID is out search term
folders.sort()

# create ClearMap folders in the input folder as well as max projections
folder_output = os.path.join( folder_output_path, 'bifrost_output' )
folder_maxproj_output = os.path.join( folder_output_path, 'pictures/max_projections' )
folder_histogram_output = os.path.join( folder_output_path, 'pictures/histograms' )
folder_nifti_output = os.path.join( folder_output_path, 'nifti' )
folder_coords_output = os.path.join( folder_output_path, 'coords' )
if not os.path.exists( folder_output ): os.makedirs( folder_output )
if not os.path.exists( folder_maxproj_output ): os.makedirs( folder_maxproj_output )
if not os.path.exists( folder_histogram_output ): os.makedirs( folder_histogram_output )
if not os.path.exists( folder_nifti_output ): os.makedirs( folder_nifti_output )
if not os.path.exists( folder_coords_output ): os.makedirs( folder_coords_output )
#
#
# Search for scan folders
folders = glob.glob( os.path.join( folder_input, '*ID*') ) # ID is out search term
folders.sort()

for folder_number, fldr in enumerate(folders):
    if os.path.isdir(fldr):
        basename = os.path.basename(fldr)
        print(basename)

# ladies and gentlemen, start your timers!
start_time = time.time()

for folder_number, fldr in enumerate(folders):
    if os.path.isdir(fldr):
        basename = os.path.basename(fldr)


        if not os.path.exists( folder_output_path + 'bifrost_output/' + basename ): os.makedirs( folder_output_path + 'bifrost_output/' + basename )

        tag   = 'ID' + fldr[ (fldr.find('ID') + 2):(fldr.find('ID') + ID_tag_length + 2) ]
        group = 'g'  + fldr[ (fldr.find('g')  + 1):(fldr.find('g')  + group_tag_length + 1) ]
        print(tag)

        ## ALIGN CHANNELS
        print('Reading raw tiffs..')
        auto, spec = bf.readRawTiffs(fldr)

        print('Aligning channels..')
        for i in range(auto.shape[0]):
            print(i)

            temp_auto = auto[i,:,:]
            bf.saveNifti(temp_auto, folder_output_path + 'nifti/' + 'temp_auto.nii.gz')
            # copy previously computed transformation file from individual SCAN folders
            shutil.copy(fldr+'/auto_aff_'+str(i)+'.txt',folder_output_path + 'nifti/'+'auto_aff_'+str(i)+'.txt')

            #### Align channels
            moving = folder_output_path + 'nifti/' + 'temp_auto.nii.gz'
            fixed = 'dummy'
            result_path = folder_output_path + 'nifti'
            hu = bf.Huginn(moving,fixed,elastix_path,result_path)

            hu.transform_vol(moving, 'auto_aff_'+str(i), 'temp_auto_aff')
            auto[i,:,:] = bf.readNifti(folder_output_path + 'nifti/' + 'temp_auto_aff.nii.gz')
            os.remove(folder_output_path + 'nifti/'+'auto_aff_'+str(i)+'.txt')


        # # debug save volumes
        if cell_check==1:
            bf.saveNifti(spec, folder_output_path + 'nifti/' + tag + 'spec_full.nii.gz')
            # bf.saveNifti(auto, folder_output_path + 'nifti/' + tag + 'auto_full.nii.gz')



        # # ### PRE-PROCESSING FOR REGISTRATION
        print('Downsampling to 20um..')
        # Run at 20um resolution
        auto_20 = []
        spec_20 = []

        for d in range(auto.shape[0]):
            auto_20.append(img_as_uint(rescale(auto[d,:,:], 4.794/20)))
            spec_20.append(img_as_uint(rescale(spec[d,:,:], 4.794/20)))

        auto_20 = np.array(auto_20)
        spec_20 = np.array(spec_20)
        auto_20 = downscale_local_mean(auto_20, (2,1,1))
        spec_20 = downscale_local_mean(spec_20, (2,1,1))
        print(auto_20.shape)


        # # debug save volumes
        bf.saveNifti(auto_20, folder_output_path + 'nifti/' + tag + 'auto_20.nii.gz')
        bf.saveNifti(spec_20, folder_output_path + 'nifti/' + tag + 'spec_20.nii.gz')


        print('Starting pre-proccesing..')
        cutoff = np.percentile(auto_20,5)

        mask = np.zeros(auto_20.shape,'uint8')
        mask[auto_20>cutoff] = 1

        bf.saveNifti(auto_20, folder_output_path+r'nifti/auto_temp.nii.gz')
        bf.saveNifti(mask, folder_output_path+r'nifti/mask_temp.nii.gz')


        #### BIAS FIELD CORRECTION #####
        program_bias = r'INSERT PATH TO ../run_Bias_field_correction.sh '
        runtime_path = r'INSERT PATH TO ../MATLAB/MATLAB_Runtime/v93 '
        vol_in =  folder_output_path+r'nifti/auto_temp.nii.gz '
        mask_in = folder_output_path+r'nifti/mask_temp.nii.gz '
        cor_out = folder_output_path+r'nifti/auto_biascor.nii.gz '
        origin_out = folder_output_path+r'nifti/origin.nii.gz '

        print(program_bias + runtime_path + vol_in  + mask_in + cor_out + origin_out)
        os.system(program_bias + runtime_path + vol_in  + mask_in + cor_out + origin_out)

        corr = bf.readNifti(folder_output_path+r'nifti/auto_biascor.nii.gz')

        # fix matlab orientation issue..
        corr = np.flip(corr,1)
        corr = np.flip(corr,2)

        # RESCALE TO 8 BIT
        scale_limit = np.percentile(corr, (99.999))
        corr = skimage.exposure.rescale_intensity(corr, in_range=(0,scale_limit), out_range='uint8')
        auto_20 = np.copy(corr)

        # rescale intensity based on mean/median of tissue
        auto_temp = np.copy(auto_20)
        scale_thres = skimage.filters.threshold_otsu(auto_temp)
        auto_temp[auto_temp<scale_thres] = 0
        nz_mean = np.mean(auto_temp[auto_temp>0].flatten())
        scale_fact = 120/nz_mean
        auto_20 = auto_20*scale_fact
        auto_20[auto_20>255] = 255
        auto_20= auto_20.astype('uint8')

        # CLAHE IN THREE ORTHOGONAL PLANES
        auto_temp_hor = np.copy(auto_20)
        auto_temp_cor = np.copy(auto_20)
        auto_temp_sag = np.copy(auto_20)

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
        clahe_final[:,:,:,1] = auto_20

        clahe_final = np.mean(clahe_final,3)


        # # Save file for Registration
        if not os.path.exists( folder_output_path + 'nifti' ):
            os.makedirs( folder_output_path + 'nifti' )

        bf.saveNifti(clahe_final, folder_output_path + 'nifti/' + tag + '_regi.nii.gz')



        # # ### ATLAS REGISTRATION (atlas -> sample)
        moving = 'INSERT PATH TO THE ATLAS FILES .../atlas'+atlas_version+'/gubra_template.nii.gz'
        fixed = folder_output_path + 'nifti/' + tag + '_regi.nii.gz'
        result_path = folder_output_path + 'nifti'
        hu = bf.Huginn(moving,fixed,elastix_path,result_path)

        # affine Registration
        hu.registration('Affine_Gubra_June2019.txt', tag+'aff', datatype='uint8')

        # Bspline registration
        hu.registration('Bspline_Gubra_June2019.txt', tag+'bspline', init_trans=tag+'aff', datatype='uint8')

        # ## Transform atlas annotation to sample SPACE
        ano_path = 'INSERT PATH TO THE ATLAS FILES ../atlas'+atlas_version+'/gubra_ano_reduced_v2.nii.gz'
        hu.transform_vol(ano_path, tag+'bspline', tag+'ano_bspline', type='ano')

        ## Transform ventricular mask
        vent_path = 'INSERT PATH TO THE ATLAS FILES ../atlas'+atlas_version+'/gubra_ventricular_mask.nii.gz'
        hu.transform_vol(vent_path, tag+'bspline', tag+'ventricular_bspline', type='ano')

        ## Transform tissue masks
        tissue_path = 'INSERT PATH TO THE ATLAS FILES ../atlas'+atlas_version+'/gubra_tissue_mask.nii.gz'
        hu.transform_vol(tissue_path, tag+'bspline', tag+'tissue_bspline', type='ano')



        #### DETECT CELLS - specific channel
        print(spec.shape)
        print(auto.shape)
        # read raw data
        # cell detect chunk-by-chunk with overlap
        # all parameters
        voxel_size_x =  4.7944; voxel_size_y =  4.7944; voxel_size_z = 10.0000
        voxel_size = voxel_size_x * voxel_size_y * voxel_size_z

        voxel_size_clearmap = 4.06 * 4.06 * 3
        cell_size_min_clearmap = 20 * voxel_size_clearmap  # clearmap std. min size
        cell_size_max_clearmap = 900 * voxel_size_clearmap # clearmap std. max size

        cell_size_min = cell_size_min_clearmap / voxel_size
        cell_size_min = 8
        cell_size_max = cell_size_max_clearmap / voxel_size

        back_subtract_disk = disk(3)
        max_filter_cube = np.ones((3,5,5))
        watershed_threshold = bg_spec_thres
        cell_size_threshold = (cell_size_min, cell_size_max)

        print('Detecting cells - spec')
        seq = range(spec.shape[0])
        chunk_size = 20
        chunk_overlap = 5
        chunks = []
        for i in range(0, len(seq) - chunk_overlap, chunk_size - chunk_overlap):
            chunks.append(seq[i:i + chunk_size])

        coord_list = []

        for chunk in chunks:
            print(chunk)

            # load image chunk
            chunk_im = np.copy( spec[chunk] )
            chunk_im_auto = np.copy( auto[chunk] )

            # detect cell coords in chunk
            chunk_coords = bf.cell_detector_v2(chunk_im, chunk_im_auto, back_subtract_disk, max_filter_cube, watershed_threshold, cell_size_threshold,bg_spec_thres,bg_auto_thres)

            # # update chunk offset and add to current coords
            chunk_offset = chunk.start
            chunk_coords[:,0] = chunk_coords[:,0] + chunk_offset

            # add to current list
            coord_list.append(chunk_coords)

        coords = np.vstack(coord_list)
        coords.shape

        # efficient way of detecting unique rows
        # (http://www.ryanhmckenna.com/2017/01/efficiently-remove-duplicate-rows-from.html)
        y = ( coords.dot( np.random.rand(coords.shape[1]) ) * 1e9 ).astype('uint64')
        _, idx = np.unique(y, return_index=True)
        new = coords[idx]


        cell_coords = []
        for k in range(new.shape[0]):
            cell_coords.append([new[k,0],new[k,1],new[k,2]])

        if cell_check==1:
            # Write signal as binary volumes
            signal_mask = bf.points2vol(spec.shape, cell_coords)
            bf.saveNifti(signal_mask, folder_output_path + 'nifti/' + tag + '_cell_coords_full.nii.gz')


        # Scale points to 20 um
        if not os.path.exists( folder_output_path + 'coords' ):
            os.makedirs( folder_output_path + 'coords' )

        adjust_z_scale = 10/20
        adjust_row_scale = 4.794/20
        adjust_col_scale = 4.794/20

        for k in range(len(cell_coords)):
            cell_coords[k][0] = np.round(cell_coords[k][0]*adjust_z_scale)
            cell_coords[k][1] = np.round(cell_coords[k][1]*adjust_row_scale)
            cell_coords[k][2] = np.round(cell_coords[k][2]*adjust_col_scale)

        # save debug volume of points
        coords_vol = bf.points2vol(spec_20.shape, cell_coords)
        bf.saveNifti(coords_vol, folder_output_path + 'nifti/' + tag + '_cell_coords.nii.gz')

        #remove edge signal
        ano_mask = bf.readNifti( folder_output_path + 'nifti/' + tag+'tissue_bspline.nii.gz') #
        ano_mask[ano_mask>0] = 1
        #bf.saveNifti(ano_mask, folder_output_path + 'nifti/' + tag + '_ano_mask.nii.gz')

        ano_mask=binary_erosion(ano_mask, selem=ball(edge_removal_radius))
        ano_mask = np.invert(ano_mask)
        #bf.saveNifti(ano_mask.astype('uint8'), folder_output_path + 'nifti/' + tag + '_ano_mask_eroded.nii.gz')

        # remove ventricular signal (dilate ventricular component)
        ventricular_mask = bf.readNifti( folder_output_path + 'nifti/' + tag + 'ventricular_bspline.nii.gz') # TEMP - delete
        ventricular_mask[ventricular_mask>0] = 1
        #bf.saveNifti(ventricular_mask, folder_output_path + 'nifti/' + tag + '_ventricular_mask.nii.gz')

        ventricular_mask=binary_dilation(ventricular_mask, selem=ball(ventricular_removal_radius))
        #bf.saveNifti(ventricular_mask.astype('uint8'), folder_output_path + 'nifti/' + tag + '_ventricular_mask_dilated.nii.gz')

        # combine masks
        final_mask = np.bitwise_or(ano_mask,ventricular_mask)

        # save the CVOs
        SFO_id = 338
        OV_id = 763
        ME_id = 10671
        AP_id = 207

        SFO_mask = bf.readNifti( folder_output_path + 'nifti/' + tag + 'ano_bspline.nii.gz')
        OV_mask = np.copy(SFO_mask)
        ME_mask = np.copy(SFO_mask)
        AP_mask = np.copy(SFO_mask)

        SFO_mask[SFO_mask!=SFO_id] = 0
        SFO_mask[SFO_mask>0] = 1
        OV_mask[OV_mask!=OV_id] = 0
        OV_mask[OV_mask>0] = 1
        ME_mask[ME_mask!=ME_id] = 0
        ME_mask[ME_mask>0] = 1
        AP_mask[AP_mask!=AP_id] = 0
        AP_mask[AP_mask>0] = 1

        CVO_mask = SFO_mask + OV_mask + ME_mask + AP_mask
        CVO_mask[CVO_mask>1] = 1

        final_mask[CVO_mask==1] = 0
        bf.saveNifti(final_mask.astype('uint8'), folder_output_path + 'nifti/' + tag + '_final_mask.nii.gz')

        # Apply mask and update coords
        coords_vol_clean = bf.rm_noise(pred=final_mask, signal_mask=coords_vol, dilation_radius=1)
        bf.saveNifti(coords_vol_clean.astype('uint8'), folder_output_path + 'nifti/' + tag + '_cell_coords_clean.nii.gz')


        #### SAVE COUNTS FOR STATISTICS
        print('Saving counts for statistics...')
        # TEMP DELETE again
        coords_vol_clean = bf.readNifti(folder_output_path + 'nifti/' + tag + '_cell_coords_clean.nii.gz')
        ano_sample = bf.readNifti(folder_output_path + 'nifti/' + tag + 'ano_bspline.nii.gz')

        # read csv_template
        df = pd.read_csv(os.path.join('INSERT PATH TO','annotated_counts_template_anov2.csv'),header=None)

        for index, row in df.iterrows():
            # compute unmix total signal in each brain region
            tot_sig = np.sum(coords_vol_clean[ano_sample==row[0]])
            df.set_value(index,1,tot_sig)

        # write data frame as csv file
        df.to_csv(folder_output_path + 'bifrost_output/' + basename + '/annotated_counts.nii.csv', header=False, index=False)
#
        #### HEAT MAP GENERATION
        print('Generating density maps..')
        # ### ATLAS REGISTRATION
        fixed = 'INSERT PATH TO THE ATLAS FILES ../atlas'+atlas_version+'/gubra_template.nii.gz'
        moving = folder_output_path + 'nifti/' + tag + '_regi.nii.gz'
        result_path = folder_output_path + 'nifti'
        hu = bf.Huginn(moving,fixed,elastix_path,result_path)

        #
        # affine Registration
        hu.registration('Affine_Gubra_June2019.txt', tag+'aff_atlas', datatype='uint8')

        # Bspline registration
        hu.registration('Bspline_Gubra_June2019.txt', tag+'bspline_atlas', init_trans=tag+'aff_atlas', datatype='uint8')

        # Transform coords vol
        coords_path = folder_output_path + 'nifti/' + tag + '_cell_coords_clean.nii.gz'
        hu.transform_vol(coords_path, tag+'bspline_atlas', tag+'coords_clean_bspline', type='ano')

        coords_vol_clean_bspline = bf.readNifti(folder_output_path + 'nifti/' + tag + 'coords_clean_bspline.nii.gz')
        seq = range(coords_vol_clean_bspline.shape[0])
        chunk_size = 25
        chunk_overlap = 5
        chunks = []
        for i in range(0, len(seq) - chunk_overlap, chunk_size - chunk_overlap):
            chunks.append(seq[i:i + chunk_size])

        coord_list = []
        for chunk in chunks:
            print(chunk)

            # load image chunk
            chunk_im = np.copy( coords_vol_clean_bspline[chunk] )

            # detect cell coords in chunk
            chunk_coords = bf.vol2coords(chunk_im)

            # update chunk offset and add to current coords
            chunk_offset = chunk.start
            chunk_coords[:,0] = chunk_coords[:,0] + chunk_offset

            # add to current list
            coord_list.append(chunk_coords)

        coords = np.vstack(coord_list)
        coords.shape

        # efficient way of detecting unique rows
        # (http://www.ryanhmckenna.com/2017/01/efficiently-remove-duplicate-rows-from.html)
        y = ( coords.dot( np.random.rand(coords.shape[1]) ) * 1e9 ).astype('uint64')
        _, idx = np.unique(y, return_index=True)
        new = coords[idx]

        print('cell_coords updates...')

        cell_coords_atlas = []
        for k in range(new.shape[0]):
            cell_coords_atlas.append([new[k,0],new[k,1],new[k,2]])

        template = bf.readNifti('/home/administrator/toolboxes/Bifrost/cfos_analysis/atlas'+atlas_version+'/gubra_template.nii.gz')
        heatmap = bf.create_atlas_heatmap(template.shape, cell_coords_atlas, density_map_radius)

        bf.saveNifti(heatmap.astype('uint16'), folder_output_path + 'bifrost_output/' + basename + '/cells_heatmap.nii.gz')


# how long did it take?
elapsed_time = time.time() - start_time
print('Time for running script:')
print(elapsed_time)
