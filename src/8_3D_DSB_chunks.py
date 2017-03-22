from __future__ import print_function
import pickle
import time
import os
import pandas as pd
import numpy as np
from def_chunks import worldToVoxelCoord, get_chunk
import matplotlib.pyplot as plt
import scipy.ndimage
import sys
sys.path.append("../")
# Import our competition variables
from competition_config import *

def load_DSB_resized_image(IMG_PATH):
    with open(IMG_PATH, 'rb') as handle:
        resized_scan = pickle.load(handle)
    return resized_scan

def save_DSB_chunks(cand_same_size=True, PLOT_CAND=False, unknown_diam=True):

    if unknown_diam and cand_same_size:
        print("Warning: no resize posible with unknown diameter; setting cand_same_size to False")
        cand_same_size = False

    CANDIDATES_FILE = D8b["INPUT_METADATA"]
    if not os.path.exists(CANDIDATES_FILE):
        print("No cluster csv file", CANDIDATES_FILE)
        exit(0)

    with open(CANDIDATES_FILE, 'rb') as handle:
        cands = pickle.load(handle) # pandas dataframe

    print("Opened", CANDIDATES_FILE)

    input_resized_dir = D8b["INPUT_DIRECTORY"]
    if not os.path.exists(input_resized_dir):
        print("No resized images input folder", input_resized_dir)
        exit(0)

    output_chunks_dir = D8b['OUTPUT_DIRECTORY']
    if not os.path.exists(output_chunks_dir):
        os.makedirs(output_chunks_dir)

    chunk_dims = D8b['CHUNK_DIMS']
    chunk_side = chunk_dims[0]
    NEW_CAND_DIAMETER = D8b['NEW_CAND_DIAMETER']
    dist_th2 = D8b['DIST_TH2']
    diam_th = D8b['DIAM_TH']

    t1 = time.time()
    n_cands = 0
    n_rep = 0
    print(len(cands), cands.columns) # 150.000 !!!
    cands = cands[cands['mass_radius'] > 3] # 120.000 > 3, 89.000 > 5, 32.000 > 10, 2859 > 40 !!!, 240 > 100
    print(len(cands))
    #plt.hist(cands["mass_radius"].values, bins=100)
    #plt.show()

    DSBids = cands.ct_scan_id.unique() # numpy.ndarray
    print(DSBids.shape) # 1223 imagenes con nodulos > 40 mm, 209 con nodulos > 100 mm

    for DSBid in DSBids:
        img_file = DSBid + ".pickle"
        if os.path.isfile(input_resized_dir + img_file):
            prev_cand_L = []
            numpyImage = load_DSB_resized_image(input_resized_dir + img_file)
            cands_sel = cands[cands["ct_scan_id"] == DSBid]
            for index, cand in cands_sel.iterrows():
                if unknown_diam==False:
                    diam = cand['mass_radius']*2.0
                    if diam > diam_th:
                        continue

                PREV_REPET = False
                cand_id = cand['id_nodule']
                cand_file = cand_id + ".pickle"
                if not os.path.isfile(output_chunks_dir + cand_file):
                    cand_center = cand['mass_center'] # x, y, z
                    cand_center = cand_center[::-1] # z, y, x; para el get_chunk
                    """
                    x = float(cand_center[2])
                    y = float(cand_center[1])
                    z = float(cand_center[0])
                    diam = float(annot["diameter_mm"])
                    for prev_ann in prev_ann_L:
                        p_id = prev_ann['id']
                        p_x = prev_ann['x']
                        p_y = prev_ann['y']
                        p_z = prev_ann['z']
                        p_diam = prev_ann['diameter_mm']
                        dist2 = (x-p_x)**2 + (y-p_y)**2 + (z-p_z)**2
                        if dist2 < dist_th2:
                            #print("Same nodule:", p_id, p_x, p_y, p_z, " and ", cand_id, x, y, z, "Dist2", dist2)
                            if abs(p_diam-diam) > 2: # d_th2 = 1 => 13 with diff(diam) > 2
                                print("nodule diams: previous", p_diam, "current", diam)
                            PREV_REPET=True
                            n_rep += 1
                            break

                    if PREV_REPET:
                        continue
                    """
                    #print(cand_center, cand_center.shape, cand_center[::-1])

                    n_cands += 1

                    """
                    prev_cand_L.append({'id': cand_id, 'coordX': x, 'coordY': y, 'coordZ': z, "diameter_mm" : diam})
                    """
                    if cand_same_size: # constant side chunks and constant nodule radio
                        resize_factor = 1.0*diam / NEW_CAND_DIAMETER
                        real_chunk_side = int(round(resize_factor*chunk_side))
                        real_chunk_dims = [real_chunk_side, real_chunk_side, real_chunk_side]
                        real_chunk = get_chunk(numpyImage, cand_center, real_chunk_dims, show_chunk_out=True, chunk_id=cand_id)
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

                        if PLOT_CAND:
                            if diam > 16:
                                #annot_sphere(annot, chunk)
                                print("plot", cand_file, "with diameter", diam)
                                f, plots = plt.subplots(1, 2, figsize=(10, 5))
                                plots[0].imshow(real_chunk[int(real_chunk_side/2)], cmap=plt.cm.gray)
                                plots[1].imshow(chunk[int(chunk_side/2)], cmap=plt.cm.gray)
                                plt.show()
                                #exit(0)
                    else: # constant side chunks but variable nodule radio

                        chunk = get_chunk(numpyImage, cand_center, chunk_dims, show_chunk_out=True, chunk_id=cand_id)

                        if PLOT_CAND:
                            if diam > 16:
                                plt.imshow(chunk[chunk_side/2], cmap=plt.cm.gray)
                                plt.show()

                    with open(output_chunks_dir + cand_file , 'wb') as handle:
                        pickle.dump(chunk, handle, protocol=PICKLE_PROTOCOL)
                        # the size of each chunk or annotation file is 64x64x64x2 (int16 has 2 bytes per voxel) = 524 KB
    t2 = time.time()
    print ("Annotation chunks pickled:", n_cands, ". Processing time (seconds):", (t2-t1))

D8b = chunks_DSB_extraction

if __name__ == "__main__":

    START_TIME = time.time()

    save_DSB_chunks(cand_same_size=True, PLOT_CAND=False, unknown_diam=True)

    print("Total elapsed time: " + \
          str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
