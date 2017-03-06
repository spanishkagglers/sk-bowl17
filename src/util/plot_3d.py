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

import sys
sys.path.append("../")
# Import our competition variables, has to be before matplotlib
from competition_config import *
d=nodules_3d_segmentation

#import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
start_time = time.time()



from scipy.spatial.distance import *


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

def plot_3d(data, max_z=None, col=None, ax=None):

           
    if ax is None:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    
    
    if isinstance(data,list) and len(data)==3:
        information_points = np.array(data)
    elif len(data.shape==3):
        information_points = np.nonzero(data)
    else:
        information_points = data
    
    
    num_points=information_points[1].shape[0]
    '''
    X=np.zeros(shape=(num_points,3))
    for i in range(len(information_points[0])):
        X[i][0] = information_points[0][i]
        X[i][1] = information_points[1][i]
        X[i][2] = information_points[2][i]
    
    
    

    #data = np.random.rand(500,3)
    data=X
    
    db = DBSCAN(eps=5, min_samples=10).fit(data)
    labels = db.labels_
    
    a=Counter(labels)
    labels_set= set(labels)
    '''
    
    
    
    z = information_points[2]
    y = information_points[1]
    x = information_points[0]
    
    if max_z==None:
        maz_z=max(z)
    
    z = max_z - z # don't ask me why!
    #z=z_invertido
    
    #z = max(z)-z
    
    ax.plot(x, y, z, c=col, marker=',', lw = 0, alpha=1)
    
    return ax
