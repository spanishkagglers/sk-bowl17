# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:27:49 2017

@author: spanish kagglers
"""


import os, os.path

import numpy as np # linear algebra
import shutil
from glob import glob

import scipy.misc
import pickle
#import json
import yaml

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
d=lidc_etl

from scipy.spatial.distance import *

import re
from xml.etree import ElementTree
import dicom




try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')



if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])


def get_output_filenames(ct_scan_id):
    info_output_filename=d['OUTPUT_DIRECTORY'] + "info_" + ct_scan_id + ".pickle"
    points_output_filename=d['OUTPUT_DIRECTORY'] + "points_" + ct_scan_id + ".pickle"
    
    return info_output_filename, points_output_filename

def worldToVoxelCoord(worldCoord, origin, spacing):
	stretchedVoxelCoord = np.absolute(worldCoord - origin)
	voxelCoord = stretchedVoxelCoord / spacing
	return voxelCoord


def find_ct_scan_folder(series_instance_uid):
    
    for dirpath, dirnames, filenames in os.walk(d['DICOM_INPUT_DIRECTORY']):
        for dirname in dirnames:
            if dirname==series_instance_uid:
                return dirpath + "/" + dirname + "/"
    return None
    
def get_transformed_points(segmentation_points, ct_scan_folder):
    original_scan = [dicom.read_file(ct_scan_folder + filename, force=True) for filename in sorted(os.listdir(ct_scan_folder)) if filename.endswith('.dcm')]
    original_scan.sort(key=lambda x: int(x.InstanceNumber)) 
    origin = original_scan[0].ImagePositionPatient 
    spacing = np.array([abs(original_scan[0].SliceThickness)] + original_scan[0].PixelSpacing, dtype=np.float32)
    spacing = np.array([spacing[2], spacing[1], spacing[0]]) # cambiamos ZYX a XYZ
#    print("origin: ", origin)
#    print("spacing: " , spacing)
    transformed_points=[]
    for (x,y,z) in segmentation_points:
        transformed_points.append(worldToVoxelCoord(np.array([x,y,z]), origin, spacing))
#    print(transformed_points)
    
    return transformed_points


#get_transformed_points([(364,172,1550.5)], ct_scan_folder)    


#xml_file_list=['/home/virilo/kaggle/lidc/LIDC-XML-only/tcia-lidc-xml/157/158.xml']

xml_file_list=[]
for dirpath, dirnames, filenames in os.walk(d['XML_INPUT_DIRECTORY']):
    for filename in [f for f in filenames if f.endswith(".xml")]:   
        xml_file_list.append(dirpath + "/"  +filename)
    
    
for xml_filename in tqdm(xml_file_list):
    
    with open(xml_filename,'r') as f:
        xml_tree = ElementTree.fromstring( re.sub(' xmlns="[^"]+"', '', f.read()) )  
    
    study_instance_uid  = xml_tree.find('ResponseHeader').find('StudyInstanceUID').text
    series_instance_uid = xml_tree.find('ResponseHeader').find('SeriesInstanceUid').text
    
    ct_scan_folder=find_ct_scan_folder(series_instance_uid)
    
    if ct_scan_folder is None:
        print("ERROR: DICOM folder not found for :" + series_instance_uid)
        break
    
    
   

    nodules=[]
    for session in xml_tree.findall('readingSession'):
        for ann in session.findall('unblindedReadNodule'):
            
            nodule_id=ann.find('noduleID').text
            id_nodule_str=series_instance_uid+"_"+nodule_id
#            print("nodule_id:", nodule_id)
#            print("---------------")

            rois = ann.findall('roi')
            segmentation_points=[]
            x_coords=[]
            y_coords=[]
            z_coords=[]
            for roi in rois:
                if len(roi.findall('edgeMap')) <= 1:
                    continue
                
                z_pos = float(roi.find('imageZposition').text) 
                edgeMaps = roi.findall('edgeMap')
#                print("Z=",z_pos)
                for edgeMap in edgeMaps:
                    x_pos = float(edgeMap.find('xCoord').text)
                    y_pos = float(edgeMap.find('yCoord').text)
#                    print(x_pos, ", " , y_pos)
                    segmentation_points.append((x_pos,y_pos,z_pos))
                '''
                    x_coords.append(x_pos)
                    y_coords.append(y_pos)
                    z_coords.append(z_pos)
                
                center=np.array((int(round(np.mean(x_coords))), 
                    int(round(np.mean(y_coords))),
                    int(round(np.mean(z_coords)))))
                '''
                
                transformed_segmentation_points=get_transformed_points(segmentation_points, ct_scan_folder)
                
                tsp_array=np.array(transformed_segmentation_points)
                
                cluster_x = tsp_array[:,0]
                cluster_y = tsp_array[:,1]
                cluster_z = tsp_array[:,2]
                
                nodules.append({
                    'ct_scan_id':series_instance_uid,
                    'id_nodule':id_nodule_str,
#                    'center':center,
#                    'radius':radius,
                    'min_x': int(np.floor(min(cluster_x))),
                    'min_y': int(np.floor(min(cluster_y))),
                    'min_z': int(np.floor(min(cluster_z))),
                    'max_x': int(np.ceil(max(cluster_x))),
                    'max_y': int(np.ceil(max(cluster_y))),
                    'max_z': int(np.ceil(max(cluster_z))),
                    'num_points': len(cluster_x),
                })
                
                '''cluster_dict={
                    'x':cluster_x, 
                    'y':cluster_y, 
                    'z':cluster_z
                } '''   
                
                info_output_filename, points_output_filename = get_output_filenames(series_instance_uid)
                with open(info_output_filename, 'wb') as handle:
                    pickle.dump(nodules, handle, protocol=PICKLE_PROTOCOL)
                    if AWS: upload_to_s3(info_output_filename)
                
#                print(1324)
                
        #asdf
#        print("="*20)








