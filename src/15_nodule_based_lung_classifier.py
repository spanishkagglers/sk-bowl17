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

from sklearn.metrics import log_loss

from scipy.spatial.distance import *

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import time
start_time = time.time()

if not os.path.exists(d['EXECUTION_OUTPUT_DIRECTORY']):
    os.makedirs(d['EXECUTION_OUTPUT_DIRECTORY'])

labels__pd=pd.read_csv(d['BOWL_LABELS'])
all_patients=next(os.walk(d['BOWL_PATIENTS']))[1]

Y_COL_NAME='cancer'


if True: #not "test_set_patients" in globals() or test_set_patients is None:
    train_set_patients=list(labels__pd['id'].values)
    Y=np.array(labels__pd['cancer'].values)
    Y=Y.astype(np.float)
    test_set_patients=[x for x in all_patients if x not in train_set_patients]

    #intermediate_ouput_filename=d['INPUT_DIRECTORY'] +  d['CNN']  + '/intermediate_output_' + d['CNN'] + '_fold_1.csv'
    #nodules_features=pd.read_csv(intermediate_ouput_filename)


    cnn_predictions_filename=d['INPUT_DIRECTORY'] +  d['CNN']  + '/test_predictions_' + d['CNN'] + '_fold_1.csv'
    cnn_predictions=pd.read_csv(cnn_predictions_filename)

cnn_predictions['1'].hist(bins=100)

cnn_predictions['ct_scan_id']=cnn_predictions['nodule_id'].apply(lambda x: x.split("_")[0])
cnn_predictions['nod_num']=cnn_predictions['nodule_id'].apply(lambda x: x.split("_")[1])

predictions_per_nodule=cnn_predictions.groupby(cnn_predictions['ct_scan_id']).agg({
#    'prediction':np.mean,
    '1':np.max,
})



predictions_per_nodule.hist(bins=100)
predictions_per_nodule.reset_index(inplace=True)
predictions_per_nodule.rename(columns={'1':Y_COL_NAME, 'ct_scan_id':'id'}, inplace=True)
affected_lungs=predictions_per_nodule[predictions_per_nodule[Y_COL_NAME]>=d['AFFECTED_THRESHOLD']]
affected_lungs[Y_COL_NAME]=d['SCORE_AFFECTED']

healthy_lungs=predictions_per_nodule[predictions_per_nodule[Y_COL_NAME]<=d['HEALTHY_THRESHOLD']]
healthy_lungs[Y_COL_NAME]=d['SCORE_HEALTHY']

uncertain_lungs=predictions_per_nodule[
    (predictions_per_nodule[Y_COL_NAME]<d['AFFECTED_THRESHOLD']) 
    & (predictions_per_nodule[Y_COL_NAME]>d['HEALTHY_THRESHOLD'])]
uncertain_lungs[Y_COL_NAME]=d['SCORE_BASE']

preds__pd=pd.concat([affected_lungs,uncertain_lungs,healthy_lungs], axis=0)


test_preds=preds__pd.loc[preds__pd['id'].isin(test_set_patients)]
output_filename=d['EXECUTION_OUTPUT_DIRECTORY']+'class_1_{}_{}_({}).csv'.format(
    d['HEALTHY_THRESHOLD'],
    d['AFFECTED_THRESHOLD'],
    d['CNN'],
)
test_preds.to_csv(output_filename, index=False)


train_preds=preds__pd.loc[preds__pd['id'].isin(train_set_patients)]
train_preds.rename(columns={Y_COL_NAME:'prediction'}, inplace=True)

train_preds=pd.merge(train_preds, labels__pd, on="id")
score=log_loss(train_preds[Y_COL_NAME], train_preds['prediction'])
print("log_loss: "+str(score) + " vs 0.577124126548")