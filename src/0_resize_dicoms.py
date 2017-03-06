#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:59:56 2017

@author: spanish kagglers
"""
# Based on:
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

import os
import time
from random import shuffle
import pickle
import numpy as np
import scipy.ndimage
import dicom

import sys
sys.path.append("../")
# Import our competition variables, has to be before matplotlib
from competition_config import *
D0 = resize_dashboard
if not os.path.exists(D0['OUTPUT_DIRECTORY']):
    os.makedirs(D0['OUTPUT_DIRECTORY'])

START_TIME = time.time()

def read_ct_scan(folder_name):
    # Read the slices from the dicom file
    slices_l = [dicom.read_file(folder_name + filename) for \
               filename in os.listdir(folder_name)]

    # Sort the dicom slices in their respective order
    slices_l.sort(key=lambda x: int(x.InstanceNumber))

    slice_thickness = slices_l[0].ImagePositionPatient[2] - \
    slices_l[1].ImagePositionPatient[2] # Will be < 0 on inverted images

#    Guido uses:
#    try:
#        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - \
#        slices[1].ImagePositionPatient[2])
#    except:
#        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for sli in slices_l:
        sli.SliceThickness = slice_thickness

    # Get the pixel values for all the slices
    slices_a = np.stack([s.pixel_array for s in slices_l])
    slices_a = slices_a.astype(np.int16)

    # Convert to Hounsfield units (HU)
    intercept = slices_l[0].RescaleIntercept
    slope = slices_l[0].RescaleSlope

    if slope != 1:
        slices_a = slope * slices_a.astype(np.float64)
        slices_a = slices_a.astype(np.int16)

    slices_a += np.int16(intercept)

    # Saw -2048 instead of -2000, original has == -2000, changed for <= -2000
    # Instead of 0 (water), we use -1000 (air)
    slices_a[slices_a <= -2000] = -1000

    slices_a += 1000 # So it will show

    # For inverted lungs:
    if slice_thickness < 0:
        slices_a = slices_a[::-1]

    slices_a = np.array(slices_a, dtype=np.int16)

    return slices_a, slices_l

def resize(ct_scan, original_scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([abs(original_scan[0].SliceThickness)] + \
                       original_scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = ct_scan.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / ct_scan.shape
    new_spacing = spacing / real_resize_factor

    ct_scan = scipy.ndimage.interpolation.zoom(ct_scan, real_resize_factor, mode='nearest')
    # ct_scan = scipy.ndimage.zoom(ct_scan, real_resize_factor)
    # ct_scan.resize(np.array(new_shape, dtype=np.int16))

    return ct_scan, new_spacing


def resize_all_ct_scans(input_path, output_path): # Iterate through all patients
    # Initialize lenght of previus batch to not iterate through errors
    patients = []
    while True:
        # batch_to_process function from competition_config
        # it returns a *.pickle list, we will remove the extension this time
        patients, len_prev_batch = \
        batch_to_process(input_path, output_path, AWS, input_is_folder=True), len(patients)
        if len(patients) == 0 or len(patients) == len_prev_batch:
            break

        for patient in patients:
            i_start_time = time.time()
            # If pickle already exist, skip to the next patient
            if os.path.isfile(output_path + patient + '.pickle'):
                print(patient + ' already processed. Skipping...')
                continue

            # Resize
            print('Resizing ' + patient + '...')
            try:
                if AWS: download_from_s3(input_path + patient, input_is_folder=True)
                ct_scan, original_scan = read_ct_scan(input_path + patient + '/')

                # Resize pixel spacing to a certain isotrpic resolution
                resized_a, new_spacing = \
                resize(ct_scan, original_scan, D0['NEW_SPACING'])

                print('Shape before:', ct_scan.shape, '-', 'After:', resized_a.shape)

                # Save object as a .pickle
                with open(output_path + patient + '.pickle', 'wb') as handle:
                    pickle.dump(resized_a, handle, protocol=PICKLE_PROTOCOL)
                    if AWS:
                        upload_to_s3(output_path + patient + '.pickle')
                        clean_after_upload(input_path + patient, \
                                           output_path + patient + '.pickle', \
                                           input_is_folder=True)

                # Print and time to finish
                i_time = time.time() - i_start_time
                print('Done in ' + \
                    str(time.strftime('%H:%M:%S', time.gmtime(i_time))))

            except ValueError:
                print(patient + ' patient rised a ValueError! Continuing...')
            except IndexError:
                print(patient + ' patient rised an IndexError! Continuing...')

resize_all_ct_scans(D0['INPUT_DIRECTORY'], D0['OUTPUT_DIRECTORY'])

print("Total elapsed time: " + \
      str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
