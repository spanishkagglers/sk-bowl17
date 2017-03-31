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

features_list =\
['calcification',
'internalStructure',
'lobulation',
'malignancy',
'margin',
'sphericity',
'spiculation',
'subtlety',
'texture']


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
    
def get_transformed_points(segmentation_points, ct_scan_folder, use_just_one_slice=True):
    
    slices_file_list=[filename for filename in sorted(os.listdir(ct_scan_folder)) if filename.endswith('.dcm')]
    
    if use_just_one_slice and len(slices_file_list)>0:
        slices_file_list=[slices_file_list[0]]
    
    original_scan = [dicom.read_file(ct_scan_folder + filename, force=True) for filename in slices_file_list]
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
    
    return transformed_points, origin, spacing

def etree2dict(t):
    dictionary = {t.tag : map(etree2dict, t.iterchildren())}
    dictionary.update(('@' + k, v) for k, v in t.attrib.iteritems())
    dictionary['text'] = t.text
    return dictionary

#get_transformed_points([(364,172,1550.5)], ct_scan_folder)    


#xml_file_list=['/home/virilo/kaggle/lidc/LIDC-XML-only/tcia-lidc-xml/157/158.xml']

xml_file_list=[]
for dirpath, dirnames, filenames in os.walk(d['XML_INPUT_DIRECTORY']):
    for filename in [f for f in filenames if f.endswith(".xml")]:   
        xml_file_list.append(dirpath + "/"  +filename)
    
num_nodules=0    
for xml_filename in tqdm(sorted(xml_file_list)):
    #print(xml_filename)
    
    with open(xml_filename,'r') as f:
        xml_tree = ElementTree.fromstring( re.sub(' xmlns="[^"]+"', '', f.read()) )  
    '''
    if(xml_tree.find('ResponseHeader') is None or xml_tree.find('ResponseHeader').find('StudyInstanceUID') is None or xml_tree.find('ResponseHeader').find('SeriesInstanceUid') is None):
        print("ERROR: StudyInstanceUID and SeriesInstanceUid required.  Skipped: " + xml_filename)
        continue
    '''
    try:
        study_instance_uid  = xml_tree.find('ResponseHeader').find('StudyInstanceUID').text
        
        series_node = xml_tree.find('ResponseHeader').find('SeriesInstanceUid')
        
        if series_node is not None:
            ct_type='normal'
            series_instance_uid=series_node.text
        else:
            
            series_node = xml_tree.find('ResponseHeader').find('CTSeriesInstanceUid')
            
            if series_node is not None:
                ct_type='CT'
                series_instance_uid=series_node.text
            else:
                series_node = xml_tree.find('ResponseHeader').find('CXRSeriesInstanceUid')
                series_instance_uid=series_node.text                        
    except Exception as e:
        print(e)
        print(xml_filename)
        continue
        
    #ct_scan_id=study_instance_uid
    if "161-resubmitted-correction-3-9-12.xml" in filename:
        ct_scan_id = "161-3-9-12"
    else:
        ct_scan_id=xml_filename.split('/')[-2] + "_" + xml_filename[:-len(".xml")].split('/')[-1]
    
    
    info_output_filename, points_output_filename = get_output_filenames(ct_scan_id)
    
#    if os.path.isfile(info_output_filename): #BUG: skipping already done files affects on id_nodule
        #print("Skipped "+ xml_filename)
#        continue
    
    ct_scan_folder=find_ct_scan_folder(series_instance_uid)
    
    if ct_scan_folder is None:
        print("ERROR: DICOM folder not found for :" + series_instance_uid)
        continue
    
    clusters_dict={}
    nodules=[]
    for session in xml_tree.findall('readingSession'):
        for ann in session.findall('unblindedReadNodule'):
            
            original_nodule_id=ann.find('noduleID').text
            id_nodule_str=ct_scan_id+"_"+str(num_nodules+1)
#            print("nodule_id:", nodule_id)
#            print("---------------")

            rois = ann.findall('roi')
            segmentation_points=[]
            x_coords=[]
            y_coords=[]
            z_coords=[]
            for roi in rois:
                
#                if len(roi.findall('edgeMap')) <= 0:
#                    continue
                
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
                
                '''cluster_dict={
                    'x':cluster_x, 
                    'y':cluster_y, 
                    'z':cluster_z
                } '''
                
                
            transformed_segmentation_points, origin, spacing,center,radius=1,1,[1,1,1],1,1
                            
            transformed_segmentation_points, origin, spacing=get_transformed_points(segmentation_points, ct_scan_folder)
            
            tsp_array=np.array(transformed_segmentation_points)
            
            cluster_x = tsp_array[:,0]
            cluster_y = tsp_array[:,1]
            cluster_z = tsp_array[:,2]
            
            center=np.array((int(round(np.mean(cluster_x))), 
                int(round(np.mean(cluster_y))),
                int(round(np.mean(cluster_z)))))
            
            center=np.reshape(center,(1,3))
            distances=cdist(np.array(center), tsp_array, metric='euclidean')
            
            center=np.reshape(center,(3))
            radius=np.max(distances)
            
            
                
            nodule={
                'ct_scan_id':ct_scan_id,
                'ct_transformation_origin': origin,
                'ct_transformation_spacing': spacing,
                'ct_transformation_spacing_z': spacing[2],
                'ct_original_nodule_id':original_nodule_id,
                'ct_type':ct_type,
                'series_instance_uid':series_instance_uid,
                'study_instance_uid':study_instance_uid,
                'id_nodule':id_nodule_str,
                'center':center,
                'radius':radius,
                'min_x': int(np.floor(min(cluster_x))),
                'min_y': int(np.floor(min(cluster_y))),
                'min_z': int(np.floor(min(cluster_z))),
                'max_x': int(np.ceil(max(cluster_x))),
                'max_y': int(np.ceil(max(cluster_y))),
                'max_z': int(np.ceil(max(cluster_z))),
                'num_points': len(cluster_x),
            }
            
            
            features = ann.find('characteristics')
            if features is not None:
                for feature_name in features_list:
                    feature_node=features.find(feature_name)
                    if feature_node is not None:
                        nodule[feature_name]= int(feature_node.text)
                    else:
                        nodule[feature_name]= -1
                
                nodules.append(nodule)
            else:
#                print("Skipped: nodule hasn't any features ", id_nodule_str, "(", series_instance_uid, " ", study_instance_uid, ")")
#                print(original_nodule_id)
#                print(xml_filename)
                continue
            
            
            cluster_dict={
                'x':cluster_x, 
                'y':cluster_y, 
                'z':cluster_z
            }
            clusters_dict[id_nodule_str]=cluster_dict
            
            num_nodules+=1
                
                   
                
                
    with open(info_output_filename, 'wb') as handle:
        pickle.dump(nodules, handle, protocol=PICKLE_PROTOCOL)
        if AWS: upload_to_s3(info_output_filename)
    
    with open(points_output_filename, 'wb') as handle:
        pickle.dump(clusters_dict, handle, protocol=PICKLE_PROTOCOL)
        if AWS: upload_to_s3(points_output_filename)
                        
#                1/0
                
#                print(1324)
                
        #asdf
#        print("="*20)

print("TOTAL: ", num_nodules, " nodules")






