from __future__ import print_function
import os
from skimage.morphology import ball, binary_closing
from skimage.measure import label,regionprops
from skimage import measure
import pickle
import boto3

import sys
sys.path.append("../")
# Import our competition variables
from competition_config import *
d=nodules_roi_dashboard
import time
start_time = time.time()

s3 = boto3.client('s3')
#Create output directory
os.makedirs('/tmp/output/')

#TODO if 5_pickle already exist, skip?

def get_nodules(segmented_ct_scan):
    
    segmented_ct_scan[segmented_ct_scan < 604] = 0

    #After filtering, there are still lot of noise because of blood vessels.
    #Thus we further remove the two largest connected component.

    selem = ball(d['BALL_RADIUS'])
    binary = binary_closing(segmented_ct_scan, selem)

    label_scan = label(binary)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000
        
        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)
            
            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in r.coords:
                segmented_ct_scan[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / \
                 (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
    
    '''Comment from ArnavJain on Kaggle about index: 
    Index is the shape index of the 3D blob. As mentioned in the TODOs, 
    I am working on reducing the generated candidates using shape index 
    and other properties. So, index is one such property.
    I will post the whole part as soon as I complete it.'''
            
    return segmented_ct_scan

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/' + key
        upload_path = '/tmp/output/' + key
        
        #print('key', key, 'bucket', bucket,)
        #print('download', download_path, 'upload', upload_path)
        s3.download_file(bucket, key, download_path)
        with open(download_path, 'rb') as handle:
            segmented_ct_scan = pickle.load(handle)
        print('downloaded!')
        segmented_nodules_ct_scan = get_nodules(segmented_ct_scan)

        with open(upload_path, 'wb') as handle:
            pickle.dump(segmented_nodules_ct_scan, handle, protocol=2)
        print('Uploading', segmented_nodules_ct_scan)
        s3.upload_file(upload_path, bucket, 'output/5_nodules_roi/' + key)

        print('Gooood!')