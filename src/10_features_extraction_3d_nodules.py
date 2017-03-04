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

from scipy import stats

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

def unit_vector(vector):
    """ Returns the unit vector of the vector  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


file_list=glob(d['INPUT_DIRECTORY_1']+"info_*.pickle")

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
            
            zmin = nod['min_z']
            zmax = nod['max_z']
            ymin = nod['min_y']
            ymax = nod['max_y']
            xmin = nod['min_x']
            xmax = nod['max_x']
            
            cuboid = roi_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
            
            cuboid_shape = cuboid.shape
            cuboid_shape_ordered = np.sort(cuboid_shape)
            cuboid_volume = (zmax-zmin+1)*(ymax-ymin+1)*(xmax-xmin+1)
            cuboid_area = 2*((zmax-zmin+1)*(ymax-ymin+1)+(zmax-zmin+1)*(xmax-xmin+1)+(ymax-ymin+1)*(xmax-xmin+1))
            
            sphere_radius = np.max(cuboid.shape)/2  #scipy.spatial.distance.cdist(center,)
            sphere_volume = 4 * np.pi * sphere_radius**3 / 3
            
            nodule_volume = np.sum(cuboid>0)
            nodule_array = cuboid[cuboid>0]
            
            nodule_geo_center = np.array([zmax,ymax,xmax])
            roi_geo_center = (np.array(roi_mask.shape)-1)/2
            
            roiGC_to_noduleGC_vector = nodule_geo_center-roi_geo_center
            
            features.append({
            'ct_scan_id':nod['ct_scan_id'],
            'id':nod['id_nodule'],
            'cuboid_filled_ratio':nodule_volume/cuboid_volume,
            'sphere_filled_ratio':nodule_volume/sphere_volume,
            'mass_center':nod['center'],
            'mass_radius':nod['radius'],
            'hu_mean':np.mean(nodule_array),
            'hu_sd':np.std(nodule_array),
            'hu_mode': np.asscalar(stats.mode(nodule_array)[0]),
            'nodule_volume':nodule_volume,
            'cuboid_volume': cuboid_volume,
            'cuboid_area_volume_ratio': cuboid_area/cuboid_volume,
            'cuboid_most_asimetric_face_edges_ratio': cuboid_shape_ordered[0]/cuboid_shape_ordered[2],
            'cuboid_diagonal':np.sqrt((zmax-zmin+1)**2+(ymax-ymin+1)**2+(xmax-xmin+1)**2),
            'nodule_center_to_roi_center_distance': np.sqrt(np.sum((roiGC_to_noduleGC_vector)**2)),
            'nodule_center_to_xaxis_angle': angle_between([0,0,1],roiGC_to_noduleGC_vector),
            'nodule_center_to_yaxis_angle': angle_between([0,1,0],roiGC_to_noduleGC_vector),
            'nodule_center_to_zaxis_angle': angle_between([1,0,0],roiGC_to_noduleGC_vector),
            'nodule_xmax_xmaxRoi_ratio': xmax/roi_mask.shape[2],
            'nodule_ymax_ymaxRoi_ratio': ymax/roi_mask.shape[1],
            'nodule_zmax_zmaxRoi_ratio': zmax/roi_mask.shape[0],
            'nodule_xcenter_xcenterRoi_ratio': nodule_geo_center[2]/roi_mask.shape[2],
            'nodule_ycenter_ycenterRoi_ratio': nodule_geo_center[1]/roi_mask.shape[1],
            'nodule_zcenter_zcenterRoi_ratio': nodule_geo_center[0]/roi_mask.shape[0],
            'nodule_fat_ratio':len(nodule_array[(nodule_array>=-100) & (nodule_array<=-50)])/nodule_volume,
            'nodule_csf_ratio':len(nodule_array[nodule_array==15])/nodule_volume,
            'nodule_kidney_ratio':len(nodule_array[nodule_array==30])/nodule_volume,
            'nodule_blood_ratio':len(nodule_array[(nodule_array>=30) & (nodule_array<=45)])/nodule_volume,
            'nodule_muscle_ratio':len(nodule_array[(nodule_array>=10) & (nodule_array<=40)])/nodule_volume,
            'nodule_greymatter_ratio':len(nodule_array[(nodule_array>=37) & (nodule_array<=45)])/nodule_volume,
            'nodule_whitematter_ratio':len(nodule_array[(nodule_array>=20) & (nodule_array<=30)])/nodule_volume,
            'nodule_liver_ratio':len(nodule_array[(nodule_array>=40) & (nodule_array<=60)])/nodule_volume,
            'nodule_soft_ratio':len(nodule_array[(nodule_array>=100) & (nodule_array<=300)])/nodule_volume,
            'nodule_bone_ratio':len(nodule_array[nodule_array>=700])/nodule_volume,
            })
    
        with open(output_filename, 'wb') as handle:
            output_filename=d['OUTPUT_DIRECTORY'] + ct_scan_id + ".pickle"
            pickle.dump(features, handle, protocol=PICKLE_PROTOCOL)
            
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))