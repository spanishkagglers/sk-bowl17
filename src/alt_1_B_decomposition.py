# This script has been released under the Apache 2.0 open source license.
# Based on: mxnet + xgboost baseline [LB: 0.57] by n01z3
# https://www.kaggle.com/drn01z3/data-science-bowl-2017/mxnet-xgboost-baseline-lb-0-57/run/736291


'''
Download from http://data.dmlc.ml/mxnet/models/imagenet-11k-place365-ch/	
resnet-50-0000.params
resnet-50-symbol.json

and move to ../resnet-50
'''





import sys
sys.path.append("../")
from competition_config import *
d=alternative_model_1b

seed=d['RANDOM_SEED']
if len(sys.argv)>1:
    import hashlib, ctypes
    seed=str(sys.argv[1])
    seed=hashlib.sha224(seed.encode('utf-8')).hexdigest()
    seed=abs(int(hash(str(seed))))
    seed=ctypes.c_uint32(seed).value
print("seed is: ", seed)

ABSOLUTE_VALUES = d['ABSOLUTE_VALUES']
if len(sys.argv)>2:
    ABSOLUTE_VALUES=str(sys.argv[2])=="True"

WHITEN =d['WHITEN']
if len(sys.argv)>3:
    WHITEN=str(sys.argv[3])=="True"
    

SVD_SOLVER =d['SVD_SOLVER']
if len(sys.argv)>4:
    SVD_SOLVER=str(sys.argv[4])

import numpy as np
np.random.seed(seed)
import random
random.seed(seed)

import glob
from matplotlib import pyplot as plt
import os
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb

from glob import glob

from sklearn.decomposition import PCA

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import pickle


if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])

import time
start_time = time.time()

ct_scan_list=[]
resnet_features=[]
file_list=glob(d['INPUT_DIRECTORY']+"*.pickle")
for input_filename in tqdm(file_list):
    ct_scan_id = os.path.splitext(os.path.basename(input_filename))[0]
    
    with open(d['INPUT_DIRECTORY'] + ct_scan_id + '.pickle', 'rb') as handle:
        ct_scan_features= pickle.load(handle)
    
    if d['RELU_AFTER_RESNET']:
        ct_scan_features=np.maximum(ct_scan_features, 0, ct_scan_features)
    
    if d['AVERAGE_RESNET_FEATURES']:
        resnet_features_average=np.mean(ct_scan_features, axis=0)
        resnet_features.append(resnet_features_average)
        ct_scan_list.append(ct_scan_id)
    else:
        resnet_features.append(ct_scan_features)
        for i in range(ct_scan_features.shape[0]):
            ct_scan_list.append(ct_scan_id)
    
resnet_features=np.array(resnet_features)

if d['AVERAGE_RESNET_FEATURES']:
    resnet_features=np.array(resnet_features)
else:
    resnet_features=np.vstack(resnet_features)



pca = PCA(n_components=60,whiten=WHITEN, svd_solver=SVD_SOLVER)
pca_resnet_features=pca.fit_transform(resnet_features)
if d['RELU_AFTER_PCA']:
    pca_resnet_features=np.maximum(pca_resnet_features, 0, pca_resnet_features)

if ABSOLUTE_VALUES:
    pca_resnet_features=abs(pca_resnet_features)


output_base_filename=d['OUTPUT_DIRECTORY']+"pca_resnet_features_average"    
#with open(output_base_filename+".pickle", 'wb') as handle:
#    pickle.dump(pca_resnet_features, handle, protocol=PICKLE_PROTOCOL)
ct_scan_id__pd = pd.DataFrame(ct_scan_list, columns=["ct_scan_id"])
pca_resnet_features__pd=pd.concat([ct_scan_id__pd, pd.DataFrame(pca_resnet_features)], axis=1)
pca_resnet_features__pd=pca_resnet_features__pd.groupby("ct_scan_id", as_index = False).mean()    
pca_resnet_features__pd.to_csv(output_base_filename + ".csv", sep=",", header=True, index=False)
with open(output_base_filename+"_pandas.pickle", 'wb') as handle:
    pickle.dump(pca_resnet_features__pd, handle, protocol=PICKLE_PROTOCOL)

if AWS:
#    upload_to_s3(output_base_filename+".pickle")  
    upload_to_s3(output_base_filename+".csv")
    upload_to_s3(output_base_filename+"_pandas.pickle")
 

'''
file_list=glob(d['INPUT_DIRECTORY']+"*.pickle")
for input_filename in tqdm(file_list):
    ct_scan_id = os.path.splitext(os.path.basename(input_filename))[0]
    
    output_filename=d['OUTPUT_DIRECTORY']+ct_scan_id+".pickle"
    if os.path.isfile(output_filename):
        print('Skipping...'+ct_scan_id)
        pass
    else:
        nodule_segmentation(ct_scan_id, d)  
'''
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))

