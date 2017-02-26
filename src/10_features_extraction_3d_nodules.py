# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:02:42 2017

@author: javier
"""

import pandas as pd
import numpy as np # linear algebra
import os
from glob import glob

import scipy.misc
import pickle

from collections import Counter
from sklearn.cluster import DBSCAN  

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
start_time = time.time()

import sys
sys.path.append("../")
from competition_config import *
d=features_extraction_nodules_3d


from scipy.spatial.distance import *

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import time
start_time = time.time()


if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])
    

def read_roi_mask(ct_scan_id, d):
    with open(d['INPUT_DIRECTORY_1'] + ct_scan_id + '.pickle', 'rb') as handle:
        segmented_ct_scan = pickle.load(handle)
    return segmented_ct_scan

def read_segmented_nodules_info(ct_scan_id, d):
    with open(d['INPUT_DIRECTORY_2'] + ct_scan_id + '.pickle', 'rb') as handle:
        nodules_info = pickle.load(handle)
    return nodules_info       

def feature_sphere_filled_ratio(nodule):
    #center = np.array(nodule.shape)/2
    #scipy.spatial.distance.cdist(center,)
    radius = np.max(nodule.shape)/2
    sphere_volumen = 4 * np.pi * radius**3 / 3
    nodule_volumen = np.sum(nodule>0)
    return(nodule_volumen/sphere_volumen)



file_list=glob(d['INPUT_DIRECTORY_1']+"*.pickle")

for input_filename in tqdm(file_list):
    ct_scan_id = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename=d['OUTPUT_DIRECTORY'] + ct_scan_id + ".pickle"
    if os.path.isfile(output_filename):
        pass
    else:
        roi_mask = read_roi_mask(ct_scan_id, d)
        nodules_info = read_segmented_nodules_info(ct_scan_id, d)
        
        features = []
        for nod in nodules_info:
            
            nodule = roi_mask[nod['min_z']:nod['max_z']+1, nod['min_y']:nod['max_y']+1, nod['min_x']:nod['max_x']+1]

            features.append({
            'ct_scan_id':nod['ct_scan_id'],
            'id':nod['id_nodule'],
            'sphere_filled_ratio':feature_sphere_filled_ratio(nodule),
            'mass_center':nod['center'],
            'mass_radius':nod['radius'],
            })
    
        with open(output_filename, 'wb') as handle:
            output_filename=d['OUTPUT_DIRECTORY'] + ct_scan_id + ".pickle"
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))