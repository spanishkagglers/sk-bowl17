# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:02:42 2017

@author: spanish kagglers
"""

import pandas as pd
import numpy as np # linear algebra
import os
from glob import glob

import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
start_time = time.time()

import sys
sys.path.append("../")
from competition_config import *


if False:#"step_to_be_reduced" in globals() and step_to_be_reduced==6:
    print("REDUCE STEP: 6")
    d=nodules_info_csv_from_step_6
else:
    print("REDUCE STEP: 7")
    d=nodules_info_csv_from_step_7


from scipy.spatial.distance import *

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import time
start_time = time.time()


#if not os.path.exists(d['OUTPUT_DIRECTORY']):
#    os.makedirs(d['OUTPUT_DIRECTORY'])
print(d['INPUT_DIRECTORY']+"info_*.pickle")
file_list=sorted(glob(""+d['INPUT_DIRECTORY']+"info_*.pickle"))


nodules_info=[]
for input_filename in tqdm(file_list):
    ct_scan_id = os.path.splitext(os.path.basename(input_filename))[0]
    #output_filename=d['OUTPUT_DIRECTORY'] + ct_scan_id + ".pickle"
    
    with open(input_filename, 'rb') as handle:
        ct_scan_nodules = pickle.load(handle)
        nodules_info+=ct_scan_nodules

nodules_info__pd = pd.DataFrame(nodules_info)

'''
nodules_info__pd['radius']=nodules_info__pd.apply(lambda x: ( (x['max_x']-x['min_x'])**2 + (x['max_y']-x['min_y'])**2 + (x['max_z']-x['min_z'])**2)**0.5, axis=1)

nodules_info__pd['center']=nodules_info__pd.apply(lambda x: [
    int(x['min_x']+(x['max_x']-x['min_x'])/2),
    int(x['min_y']+(x['max_y']-x['min_y'])/2),
    int(x['min_z']+(x['max_z']-x['min_z'])/2)
], axis=1)

nodules_info__pd['center']=nodules_info__pd['center'].apply(np.array)
'''
nodules_info__pd.rename(columns={
    'center':'mass_center',
    'radius':'mass_radius'
}, inplace=True)

nodules_info__pd.to_csv(d['OUTPUT_FILE'].format("csv"), sep=",", header=True, index=False)

#excel_writer = pd.ExcelWriter(d['OUTPUT_FILE'].format("xslx"), engine='xlsxwriter')
#nodules_info__pd.to_excel(excel_writer, header=True, index=False)

with open(d['OUTPUT_FILE'].format("pickle"), 'wb') as handle:
	pickle.dump(nodules_info__pd, handle, protocol=PICKLE_PROTOCOL)
    
        
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))
