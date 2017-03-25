# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:02:42 2017

@author: spanish kagglers
"""
NUM_ROUNDS=7500

import pandas as pd
import numpy as np # linear algebra
import os
from glob import glob

import math

import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
start_time = time.time()

import sys
sys.path.append("../")
from competition_config import *
d=nodule_based_lung_classifier

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

labels__pd=pd.read_csv(d['BOWL_LABELS'])
all_patients=next(os.walk(d['BOWL_PATIENTS']))[1]


if True: #not "test_set_patients" in globals() or test_set_patients is None:
    train_set_patients=list(labels__pd['id'].values)
    Y=np.array(labels__pd['cancer'].values)
    Y=Y.astype(np.float)
    test_set_patients=[x for x in all_patients if x not in train_set_patients]

    #intermediate_ouput_filename=d['INPUT_DIRECTORY'] +  d['CNN']  + '/intermediate_output_' + d['CNN'] + '_fold_1.csv'
    #nodules_features=pd.read_csv(intermediate_ouput_filename)


    cnn_predictions_filename=d['INPUT_DIRECTORY'] +  d['CNN']  + '/test_predictions_' + d['CNN'] + '_fold_1.csv'
    cnn_predictions=pd.read_csv(cnn_predictions_filename)

cnn_predictions['prediction'].hist(bins=100)

cnn_predictions['ct_scan_id']=cnn_predictions['nodule_id'].apply(lambda x: x.split("_")[0])
cnn_predictions['nod_num']=cnn_predictions['nodule_id'].apply(lambda x: x.split("_")[1])

predictions_per_nodule=cnn_predictions.groupby(cnn_predictions['ct_scan_id']).agg({
#    'prediction':np.mean,
    'prediction':np.max,
})

predictions_per_nodule.hist(bins=100)



