from __future__ import print_function
__author__ = 'ajulian'

import pickle
import time
import os
import pandas as pd
import numpy as np
import scipy.ndimage
from def_chunks import worldToVoxelCoord, get_chunk
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
# Import our competition variables
from competition_config import *


def load_LUNA_resized_image(IMG_PATH):
    with open(IMG_PATH, 'rb') as handle:
        resized_scan = pickle.load(handle)
    return resized_scan

"""
http://stackoverflow.com/questions/16170705/drawing-sampling-a-sphere-in-a-3d-numpy-grid
>>> radius = 3
>>> r2 = np.arange(-radius, radius+1)**2
>>> dist2 = r2[:, None, None] + r2[:, None] + r2
>>> volume = np.sum(dist2 <= radius**2)
>>> volume
123
"""

def annot_sphere(annot, chunk):
    h = {}
    chunk_side = chunk.shape[0]
    radio =  min(annot['diameter_mm']/2.0, chunk_side/2.0)
    print(annot['diameter_mm'], chunk.shape[0], radio)
    center = (chunk_side/2.0, chunk_side/2.0, chunk_side/2.0)
    chunk[chunk[i<radio]]
    for i in range(chunk_side):
        for j in range(chunk_side):
            for k in range(chunk_side):
                if (i-center[0])**2 + (j-center[0])**2 + (k-center[0])**2 < radio**2:
                    val = chunk[i, j, k]
    print(h)
    plt.hist(h)
    plt.show()
    exit(0)

# Saves all ann(_ex) chunks from LUNA images in IMG_FOLDER
def save_LUNA_chunks(subfolder, ann_same_size=True, PLOT_ANN=False):

    ORIGIN_OF_IMAGES_FILE = D8['ORIGIN_OF_IMAGES_FILE']
    if not os.path.isfile(ORIGIN_OF_IMAGES_FILE):
        print("No origin of images file", ORIGIN_OF_IMAGES_FILE)
        exit(0)
    else:
        origin_of_images = pd.read_csv(ORIGIN_OF_IMAGES_FILE)

    input_subfolder = D8["INPUT_DIRECTORY"] + subfolder
    if not os.path.exists(input_subfolder):
        print("No resized images input subfolder", input_subfolder)
        exit(0)

    prefix = D8['PREFIX']
    if prefix not in ["ANN", "ANN_EX"]:
        print("LUNA Prefix must be ANN or ANN_EX, but is", prefix)
        exit(0)

    ANNOTS_FILE = D8["ANN_CSV_FILE"]
    OUTPUT_DIRECTORY = D8["OUTPUT_ANN_IMG_DIR"]

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    chunk_dims = D8['CHUNK_DIMS']
    chunk_side = chunk_dims[0]
    NEW_ANN_DIAMETER = D8['NEW_ANN_DIAMETER']
    dist_th2 = D8['DIST_TH2']

    t1 = time.time()
    n_ann = 0
    n_rep = 0
    annots = pd.read_csv(ANNOTS_FILE)
    print("Opening", prefix, ANNOTS_FILE)
    annots = annots[annots.diameter_mm > 3]
    LUNAids = annots.seriesuid.unique() # numpy.ndarray
    for LUNAid in LUNAids:
        img_file = LUNAid + ".pickle"
        if os.path.isfile(input_subfolder + img_file):
            prev_ann_L = []
            image_rows = origin_of_images[origin_of_images['seriesuid']==LUNAid]
            if len(image_rows) == 0:
                print("In Origin of Images could not find image", LUNAid, ". Some error in csv", ORIGIN_OF_IMAGES_FILE)
            image_row = next(image_rows.iterrows())[1] # returns (index, row); [1] is row
            numpyImage = load_LUNA_resized_image(input_subfolder + img_file)
            numpyOrigin = np.array([image_row.origin_z, image_row.origin_y, image_row.origin_x])
            annots_sel = annots[annots["seriesuid"] == LUNAid]
            for index, annot in annots_sel.iterrows():
                diam = float(annot["diameter_mm"])
                #if diam > diam_th:
                #    continue
                PREV_REPET = False
                ann_file = prefix.lower() + str(index) + ".pickle"
                if not os.path.isfile(OUTPUT_DIRECTORY + ann_file):
                    x = float(annot["coordX"])
                    y = float(annot["coordY"])
                    z = float(annot["coordZ"])

                    for prev_ann in prev_ann_L:
                        p_id = prev_ann['id']
                        p_x = prev_ann['coordX']
                        p_y = prev_ann['coordY']
                        p_z = prev_ann['coordZ']
                        p_diam = prev_ann['diameter_mm']
                        dist2 = (x-p_x)**2 + (y-p_y)**2 + (z-p_z)**2
                        if dist2 < dist_th2:
                            #print("Same nodule:", p_id, p_x, p_y, p_z, " and ", index, x, y, z, "Dist2", dist2)
                            if abs(p_diam-diam) > 2: # d_th2 = 1 => 13 with diff(diam) > 2
                                print("nodule diams: previous", p_diam, "current", diam)
                            PREV_REPET=True
                            n_rep += 1
                            break

                    if PREV_REPET:
                        continue

                    n_ann += 1

                    prev_ann_L.append({'id': index, 'coordX': x, 'coordY': y, 'coordZ': z, "diameter_mm" : diam})

                    world_ann_center = np.asarray([z, y, x])
                    # REVISAR
                    numpySpacing = [1, 1, 1]

                    voxel_ann_center = worldToVoxelCoord(world_ann_center, numpyOrigin, numpySpacing)
                    # print(world_ann_center, voxel_ann_center)
                    if ann_same_size: # constant side chunks and constant nodule radio
                        resize_factor = 1.0*diam / NEW_ANN_DIAMETER
                        real_chunk_side = int(round(resize_factor*chunk_side))
                        real_chunk_dims = [real_chunk_side, real_chunk_side, real_chunk_side]
                        real_chunk = get_chunk(numpyImage, voxel_ann_center, real_chunk_dims, show_chunk_out=True, chunk_id=index)
                        #chunk = real_chunk.resize(chunk_dims)
                        chunk = scipy.ndimage.interpolation.zoom(real_chunk, 1.0/resize_factor, order=3)
                        if chunk.shape != (chunk_side, chunk_side, chunk_side):
                            #print(chunk.shape)

                            chunk_0 = np.zeros(shape=(chunk_side, chunk_side, chunk_side), dtype=np.int16)
                            smax = min(chunk_side, chunk.shape[0])
                            rmax = min(chunk_side, chunk.shape[1])
                            cmax = min(chunk_side, chunk.shape[2])
                            chunk_0[:smax, :rmax, :cmax] = chunk[:smax, :rmax, :cmax]
                            chunk = chunk_0

                        if PLOT_ANN:
                            if diam > 16:
                                #annot_sphere(annot, chunk)
                                print("plot", ann_file, "with diameter", diam)
                                f, plots = plt.subplots(1, 2, figsize=(10, 5))
                                plots[0].imshow(real_chunk[int(real_chunk_side/2)], cmap=plt.cm.gray)
                                plots[1].imshow(chunk[int(chunk_side/2)], cmap=plt.cm.gray)
                                plt.show()
                                #exit(0)
                    else: # constant side chunks but variable nodule radio
                        chunk = get_chunk(numpyImage, voxel_ann_center, chunk_dims, show_chunk_out=True, chunk_id=index)

                        if PLOT_ANN:
                            if diam > 20:
                                #annot_sphere(annot, chunk)
                                print("plot", ann_file, "with diameter", diam)
                                plt.imshow(chunk[int(chunk_side/2)], cmap=plt.cm.gray)
                                plt.show()

                    with open(OUTPUT_DIRECTORY + ann_file , 'wb') as handle:
                        pickle.dump(chunk, handle, protocol=PICKLE_PROTOCOL)
                        # the size of each chunk or annotation file is 64x64x64x2 (int16 has 2 bytes per voxel) = 524 KB
    t2 = time.time()
    print ("Annotation chunks pickled:", n_ann, ". Processing time (seconds):", (t2-t1))
    print ("Repeated nodules", n_rep)

D8 = chunks_extraction

if __name__ == "__main__":

    if not os.path.exists(D8['LUNA_ROOT']):
        print("LUNA16 root not found", D8['LUNA_ROOT'])
        exit(0)

    START_TIME = time.time()

    for subfolder in D8['SUBFOLDERS']:
        print("processing subfolder", subfolder)
        save_LUNA_chunks(subfolder, ann_same_size=True, PLOT_ANN=False)

    print("Total elapsed time: " + \
          str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
