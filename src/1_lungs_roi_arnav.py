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
import pickle

import numpy as np # linear algebra
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi

import sys
sys.path.append("../")
# Import our competition variables, has to be before matplotlib
from competition_config import *
D1 = arnavs_lugns_roi_dashboard
import matplotlib.pyplot as plt


def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)


def get_segmented_lungs(image, plot=False):
    '''Segment the lungs from the given 2D slice'''

    if plot:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))

    # Step 1: Convert into a binary image.
    # A threshold of 604(-400 HU) is used at all places because it was
    # found in experiments that it works just fine
    binary = image < 604
    if plot:
        print('Step 1: Convert into a binary image.')
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)

    # Step 2: Remove the blobs connected to the border of the image
    cleared = clear_border(binary)
    if plot:
        print('Step 2: Remove the blobs connected to the border of the image.')
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)

    # Step 3: Label the image.
    label_image = label(cleared)
    if plot:
        print('Step 3: Label the image.')
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)

    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot:
        print('Step 4: Keep the labels with 2 largest areas.')
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)

    # Step 5: Erosion operation with a disk of radius 2. This operation is
    # seperate the lung nodules attached to the blood vessels
    # using EROSION_BALL_RADIUS from competition_config.py
    selem = disk(D1['EROSION_BALL_RADIUS'])
    binary = binary_erosion(binary, selem)
    if plot:
        print('Step 5: Erosion operation')
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)

    # Step 6: Closure operation with a disk of radius 10. This operation is
    # to keep nodules attached to the lung wall
    # using CLOSING_BALL_RADIUS from competition_config.py
    selem = disk(D1['CLOSING_BALL_RADIUS'])
    binary = binary_closing(binary, selem)
    if plot:
        print('Step 6: Closure operation')
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)

    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot:
        print('Step 7: Fill in the small holes inside the binary mask of lungs.')
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)

    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    image[get_high_vals] = 0
    if plot:
        print('Step 8: Superimpose the binary mask on the input image.')
        plots[7].axis('off')
        plots[7].imshow(image, cmap=plt.cm.bone)

    return image


def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


def segment_all_ct_scans(input_path, output_path, image):
    '''Iterate through all pickles'''
    # Initialize lenght of previus batch to not iterate through errors
    patients = []
    while True:
        # batch_to_process function from competition_config
        patients, len_prev_batch = \
        batch_to_process(input_path, output_path, AWS), len(patients)
        if len(patients) == 0 or len(patients) == len_prev_batch:
            break

        for patient in patients: # patient has .pickle extension
            i_start_time = time.time()
            # If pickle already exist, skip to the next patient
            if os.path.isfile(output_path + patient):
                print(patient + ' already processed. Skipping...')
                continue

            # Segment
            print('Segmenting ' + patient + '...')
            try:
                if AWS: download_from_s3(input_path + patient)
                # Load pickle with resized ct_scan
                with open(input_path + patient, 'rb') as handle:
                    ct_scan = pickle.load(handle)
                segmented_ct_scan = segment_lung_from_ct_scan(ct_scan)

                # If true, save image as .png with input image and binary mask superimposed
                if image:
                    # patient has .pickle extension, we will remove it to save .png
                    patient_img = patient.split('.')[0] + '.png'
                    plot_ct_scan(segmented_ct_scan)
                    plt.savefig(output_path + patient_img, format='png')
                    if AWS: upload_to_s3(output_path + patient_img)
                    plt.close()

                # Save object as a .pickle
                with open(output_path + patient, 'wb') as handle:
                    pickle.dump(segmented_ct_scan, handle, protocol=PICKLE_PROTOCOL)
                    if AWS: 
                        upload_to_s3(output_path + patient)
                        clean_after_upload(input_path + patient, output_path + patient)

                # Print and time to finish
                i_time = time.time() - i_start_time
                print('Done in ' + \
                      str(time.strftime('%H:%M:%S', time.gmtime(i_time))))

            except KeyboardInterrupt:
                quit()
            except:
                print(patient + ' patient rised a %s Continuing...' % sys.exc_info()[0])

if __name__ == "__main__":
    if not os.path.exists(D1['OUTPUT_DIRECTORY']):
    os.makedirs(D1['OUTPUT_DIRECTORY'])
    # Input data files are available in the D1['INPUT_DIRECTORY'].
    # Any results you write has to be saved in the D1['OUTPUT_DIRECTORY'].
    
    START_TIME = time.time()
    # Turn to False to not save an image with slices and binary mask superimposed
    segment_all_ct_scans(D1['INPUT_DIRECTORY'], D1['OUTPUT_DIRECTORY'], True)

    print("Total elapsed time: " + \
          str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
