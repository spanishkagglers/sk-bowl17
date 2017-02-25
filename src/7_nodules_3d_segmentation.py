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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
start_time = time.time()

import sys
sys.path.append("../")
from competition_config import *
d=nodules_3d_segmentation

from scipy.spatial.distance import *

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import time
start_time = time.time()

INPUT_DIRECTORY = d['INPUT_DIRECTORY']
OUTPUT_DIRECTORY = d['OUTPUT_DIRECTORY']

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

#from subprocess import check_output
#print(check_output(["ls", INPUT_DIRECTORY]).decode("utf8"))

def plot_patch(x,y,z,r,ax):
    ax.plot([x-r/2,x-r/2,], [y-r/2,y-r/2], [z-r/2,z+r/2], c='r', marker='.')
    ax.plot([x+r/2,x+r/2,], [y-r/2,y-r/2], [z-r/2,z+r/2], c='r', marker='.')
    ax.plot([x-r/2,x-r/2,], [y+r/2,y+r/2], [z-r/2,z+r/2], c='r', marker='.')
    ax.plot([x+r/2,x+r/2,], [y+r/2,y+r/2], [z-r/2,z+r/2], c='r', marker='.')
    
    ax.plot([x-r/2,x+r/2,], [y-r/2,y-r/2], [z+r/2,z+r/2], c='r', marker='.')
    ax.plot([x-r/2,x+r/2,], [y+r/2,y+r/2], [z+r/2,z+r/2], c='r', marker='.')
    ax.plot([x-r/2,x+r/2,], [y-r/2,y-r/2], [z-r/2,z-r/2], c='r', marker='.')
    ax.plot([x-r/2,x+r/2,], [y+r/2,y+r/2], [z-r/2,z-r/2], c='r', marker='.')
    
    ax.plot([x-r/2,x-r/2,], [y-r/2,y+r/2], [z+r/2,z+r/2], c='r', marker='.')
    ax.plot([x+r/2,x+r/2,], [y-r/2,y+r/2], [z+r/2,z+r/2], c='r', marker='.')
    ax.plot([x-r/2,x-r/2,], [y-r/2,y+r/2], [z-r/2,z-r/2], c='r', marker='.')
    ax.plot([x+r/2,x+r/2,], [y-r/2,y+r/2], [z-r/2,z-r/2], c='r', marker='.')

def nodule_segmentation(ct_scan_id, output_filename, d):

    with open(INPUT_DIRECTORY + ct_scan_id + '.pickle', 'rb') as handle:
        segmented_ct_scan = pickle.load(handle)
    
    information_points = np.nonzero(segmented_ct_scan)
    num_points=information_points[1].shape[0]
    
    X=np.zeros(shape=(num_points,3))
    for i in range(len(information_points[0])):
        X[i][0] = information_points[0][i]
        X[i][1] = information_points[1][i]
        X[i][2] = information_points[2][i]
    
    Y=X
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
    
    nodules=[]
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1: # garbage points!!!
            break
    
        class_member_mask = (labels == k)
        
        cluster_x=x[class_member_mask]
        cluster_y=y[class_member_mask]
        cluster_z=max(z)-z[class_member_mask]
        cluster=np.vstack((cluster_x, cluster_y, cluster_z)).T
        
        img=np.zeros(shape=segmented_ct_scan.shape, dtype=np.uint8)
        for i in range(len(cluster_x)):
            img[cluster_z[i]][cluster_y[i]][cluster_x[i]] = 255
        
        
        center=np.array((int(round(np.mean(cluster_x))), 
                int(round(np.mean(cluster_y))),
                int(round(np.mean(cluster_z)))))
        center=np.reshape(center,(1,3))
        distances=cdist(np.array(center), cluster)
        
        center=np.reshape(center,(3))
        radius=np.max(distances)
        
        
        
        if d['PRINT_IMAGES']: #radius>40:
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot(cluster_x, cluster_y, cluster_z, c=col, marker=',', lw = 0, alpha=1)
            #center:
            ax.plot([center[0]], [center[1]], [center[2]], c='b', marker='o')
            #box
            plot_patch(center[0], center[1], center[2],radius,ax)
            #print(k)
        
        center[2]=max(z)-center[2]    
        nodules.append((center,radius))
        plt.show()
    
    #print(len(unique_labels))
    
    with open(output_filename, 'wb') as handle:
        pickle.dump(nodules, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

file_list=glob(INPUT_DIRECTORY+"*.pickle")
for input_filename in tqdm(file_list):
    ct_scan_id = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename=OUTPUT_DIRECTORY + ct_scan_id + ".pickle"
    if os.path.isfile(output_filename):
        #print('Skipping...'+ct_scan_id)
        pass
    else:
        nodule_segmentation(ct_scan_id, output_filename, d)  

        
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))

