from __future__ import print_function
__author__ = 'ajulian'

import pickle
import time
import os
import pandas as pd
import numpy as np
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

# Saves all ann(_ex) chunks from LUNA images in IMG_FOLDER
def save_LUNA_chunks(subfolder):

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
    PLOT_ANN = False

    t1 = time.time()
    n_ann = 0
    annots = pd.read_csv(ANNOTS_FILE)
    print("Opening", prefix, ANNOTS_FILE)
    annots = annots[annots.diameter_mm > 3]
    LUNAids = annots.seriesuid.unique() # numpy.ndarray
    for LUNAid in LUNAids:
        img_file = LUNAid + ".pickle"
        if os.path.isfile(input_subfolder + img_file):
            image_rows = origin_of_images[origin_of_images['seriesuid']==LUNAid]
            image_row = next(image_rows.iterrows())[1] # returns (index, row); [1] is row
            numpyImage = load_LUNA_resized_image(input_subfolder + img_file)
            numpyOrigin = np.array([image_row.origin_z, image_row.origin_y, image_row.origin_x])
            annots_sel = annots[annots["seriesuid"] == LUNAid]
            for index, annot in annots_sel.iterrows():
                ann_file = prefix.lower() + str(index) + ".pickle"
                if not os.path.isfile(OUTPUT_DIRECTORY + ann_file):
                    n_ann += 1
                    world_ann_center = np.asarray([float(annot["coordZ"]),float(annot["coordY"]),float(annot["coordX"])])
                    # REVISAR
                    numpySpacing = [1, 1, 1]

                    voxel_ann_center = worldToVoxelCoord(world_ann_center, numpyOrigin, numpySpacing)
                    # print(world_ann_center, voxel_ann_center)
                    chunk = get_chunk(numpyImage, voxel_ann_center, chunk_dims, show_chunk_out=True, chunk_id=index)

                    if PLOT_ANN:
                        plt.imshow(chunk[chunk_side/2], cmap=plt.cm.gray)
                        plt.show()

                    with open(OUTPUT_DIRECTORY + ann_file , 'wb') as handle:
                        pickle.dump(chunk, handle, protocol=PICKLE_PROTOCOL)
                        # the size of each chunk or annotation file is 64x64x64x2 (int16 has 2 bytes per voxel) = 524 KB
    t2 = time.time()
    print ("Annotation chunks pickled:", n_ann, ". Processing time (seconds):", (t2-t1))

D8 = chunks_extraction

if __name__ == "__main__":

    if not os.path.exists(D8['LUNA_ROOT']):
        print("LUNA16 root not found", D8['LUNA_ROOT'])
        exit(0)

    START_TIME = time.time()

    for subfolder in D8['SUBFOLDERS']:
        print("processing subfolder", subfolder)
        save_LUNA_chunks(subfolder)

    print("Total elapsed time: " + \
          str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
