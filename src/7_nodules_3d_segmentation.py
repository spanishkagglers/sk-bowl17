# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:08:19 2017

@author: spanish kagglers
"""

import numpy as np # linear algebra
import os
from glob import glob

import scipy.misc
import pickle

from collections import Counter
from sklearn.cluster import DBSCAN  
#import cv2
from scipy.spatial.distance import *
from util.plot_3d import * # It turns AWS variable False...

import time
start_time = time.time()

import sys
sys.path.append("../")
# Import our competition variables, has to be before matplotlib
from competition_config import *
d=nodules_3d_segmentation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import time
start_time = time.time()


if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])

def get_output_filenames(ct_scan_id):
    info_output_filename=d['OUTPUT_DIRECTORY'] + "info_" + ct_scan_id + ".pickle"
    points_output_filename=d['OUTPUT_DIRECTORY'] + "points_" + ct_scan_id + ".pickle"
    
    return info_output_filename, points_output_filename

def nodule_segmentation(ct_scan_id, d):
    
    with open(d['INPUT_DIRECTORY'] + ct_scan_id + '.pickle', 'rb') as handle:
        segmented_ct_scan = pickle.load(handle)
        
    if d['PRINT_IMAGES']: #radius>40:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    information_points = np.nonzero(segmented_ct_scan)
    num_points=information_points[1].shape[0]
    
    X=np.zeros(shape=(num_points,3))
    for i in range(len(information_points[0])):
        X[i][0] = information_points[0][i]
        X[i][1] = information_points[1][i]
        X[i][2] = information_points[2][i]
    
    ###########################################
    
    # DBSCAN
    
    '''
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    
    '''
    
    

    #data = np.random.rand(500,3)
    data=X
    
    db = DBSCAN(eps=5, min_samples=10).fit(data)
    labels = db.labels_
    
    a=Counter(labels)
    labels_set= set(labels)
    
    
    
    
    
    ###########################################
    
    
    
    
    z = information_points[0]
    y = information_points[1]
    x = information_points[2]

    
    
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    clusters_dict={}
    nodules=[]
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    id_nodule=1
    for k, col in zip(unique_labels, colors):
        if k == -1: # garbage points!!!
            break
    
        class_member_mask = (labels == k)
        
        cluster_x=x[class_member_mask]
        cluster_y=y[class_member_mask]
        cluster_z=z[class_member_mask]
        cluster=np.vstack((cluster_x, cluster_y, cluster_z)).T
        
        
        
        center=np.array((int(round(np.mean(cluster_x))), 
                int(round(np.mean(cluster_y))),
                int(round(np.mean(cluster_z)))))
        center=np.reshape(center,(1,3))
        distances=cdist(np.array(center), cluster, metric='euclidean')
        
        center=np.reshape(center,(3))
        radius=np.max(distances)
        
        
        
        if d['PRINT_IMAGES']: #radius>40:
            plot_3d([cluster_x, cluster_y, cluster_z], max_z=max(z), col=col, ax=ax)
            #center:
            ax.plot([center[0]], [center[1]], [max(z)-center[2]], c='b', marker='o')
            #box
            plot_patch(center[0], center[1], max(z)-center[2],radius,ax)
            #print(k)
            
        
        num_points  =cluster_x.shape[0]
        id_nodule_str=ct_scan_id+"_"+str(id_nodule)
        nodules.append({
            'ct_scan_id':ct_scan_id,
            'id_nodule':id_nodule_str,
            'center':center,
            'radius':radius,
            'min_x': min(cluster_x),
            'min_y': min(cluster_y),
            'min_z': min(cluster_z),
            'max_x': max(cluster_x),
            'max_y': max(cluster_y),
            'max_z': max(cluster_z),
            'num_points': num_points,
        })
        
        cluster_dict={
            'x':cluster_x, 
            'y':cluster_y, 
            'z':cluster_z
        }
        clusters_dict[id_nodule_str]=cluster_dict
        
        id_nodule+=1
        
        
    
    if d['PRINT_IMAGES']:
        plt.show()
        
    #print(len(unique_labels))
    info_output_filename, points_output_filename = get_output_filenames(ct_scan_id)
    with open(info_output_filename, 'wb') as handle:
        pickle.dump(nodules, handle, protocol=PICKLE_PROTOCOL)
        if AWS: upload_to_s3(info_output_filename)

    with open(points_output_filename, 'wb') as handle:
        pickle.dump(clusters_dict, handle, protocol=PICKLE_PROTOCOL)
        if AWS: upload_to_s3(points_output_filename)
        

file_list=glob(d['INPUT_DIRECTORY']+"*.pickle")
for input_filename in tqdm(file_list):
    ct_scan_id = os.path.splitext(os.path.basename(input_filename))[0]
    
    info_output_filename, points_output_filename = get_output_filenames(ct_scan_id)
    if os.path.isfile(info_output_filename) and os.path.isfile(points_output_filename):
        #print('Skipping...'+ct_scan_id)
        pass
    else:
        nodule_segmentation(ct_scan_id, d)  

        
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))

