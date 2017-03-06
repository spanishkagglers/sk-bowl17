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

from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
from skimage import measure

import sys
sys.path.append("../")
# Import our competition variables, has to be before matplotlib
from competition_config import *
D5 = nodules_roi_dashboard

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
START_TIME = time.time()

if not os.path.exists(D5['OUTPUT_DIRECTORY']):
    os.makedirs(D5['OUTPUT_DIRECTORY'])

# Input data files are available in the D5['INPUT_DIRECTORY'].
# Any results you write has to be saved in the D5['OUTPUT_DIRECTORY'].


def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)


def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def get_nodules(segmented_ct_scan):

    segmented_ct_scan[segmented_ct_scan < 604] = 0
    #plot_ct_scan(segmented_ct_scan)
    #plot_3d(segmented_ct_scan, 604)

    # After filtering, there are still lot of noise because of blood vessels.
    # Thus we further remove the two largest connected component.

    selem = ball(D5['BALL_RADIUS'])
    binary = binary_closing(segmented_ct_scan, selem)

    label_scan = label(binary)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000

        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)

            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]:
            for c in r.coords:
                segmented_ct_scan[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / \
                 (min((max_x - min_x), (max_y - min_y), (max_z - min_z)))

    '''Comment from ArnavJain on Kaggle about index:
    Index is the shape index of the 3D blob. As mentioned in the TODOs, 
    I am working on reducing the generated candidates using shape index 
    and other properties. So, index is one such property.
    I will post the whole part as soon as I complete it.'''

    return segmented_ct_scan


def get_nodules_all_segmented_ct_scans(input_path, output_path, image): # Iterate through all pickles
    # Initialize lenght of previus batch to not iterate through errors
    patients = []
    while True:
        # batch_to_process function from competition_config
        patients, len_prev_batch = \
        batch_to_process(input_path, output_path, True), len(patients)
        if len(patients) == 0 or len(patients) == len_prev_batch:
            break

        for patient in patients: # patient has .pickle extension
            i_start_time = time.time()
            # If pickle already exists, skip to the next patient
            if os.path.isfile(output_path + patient):
                print(patient + ' already processed. Skipping...')
                continue

            # Load pickle with segmented_ct_scan
            with open(input_path + patient, 'rb') as handle:
                segmented_ct_scan = pickle.load(handle)

            # Get nodules
            print('Getting nodules from ' + patient + '...')
            try:
                segmented_nodules_ct_scan = get_nodules(segmented_ct_scan)

                # If true, save image as .png of 3D plotted segmented nodules
                if image:
                    # patient has .pickle extension, we will remove it to save .png
                    patient_img = patient.split('.')[0] + '.png'
                    plot_3d(segmented_nodules_ct_scan, 604)
                    plt.savefig(output_path + patient_img, format='png')
                    if AWS: upload_to_s3(output_path + patient_img)
                    plt.close()

                # Save object as a .pickle
                with open(output_path + patient, 'wb') as handle:
                    pickle.dump(segmented_nodules_ct_scan, handle, protocol=PICKLE_PROTOCOL)
                    if AWS: upload_to_s3(output_path + patient)

                # Print and time to finish
                i_time = time.time() - i_start_time
                print('Done in ' + \
                      str(time.strftime('%H:%M:%S', time.gmtime(i_time))))

            except ValueError:
                print(patient + ' patient rised a ValueError! Continuing...')
            except IndexError:
                print(patient + ' patient rised an IndexError! Continuing...')

# Turn to True to save a 3D segmented nodule image
get_nodules_all_segmented_ct_scans(D5['INPUT_DIRECTORY'], D5['OUTPUT_DIRECTORY'], False)

print("Total elapsed time: " + \
      str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
