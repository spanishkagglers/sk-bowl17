#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:37:58 2017
@author: virilo
"""

global COMPETITION_HOME, COMPETITION_DATASET_DIRECTORY


# All routes must be ended by /

COMPETITION_HOME = '../'

COMPETITION_DATASET_DIRECTORY = COMPETITION_HOME + 'stage1/'
# Comment previous and uncomment to use sample_images or mini_stage1 folders for testing
#COMPETITION_DATASET_DIRECTORY = COMPETITION_HOME + 'sample_images/' # 20 patients
#COMPETITION_DATASET_DIRECTORY = COMPETITION_HOME + 'mini_stage1/' # optional number of patients

import pickle
# Used protocol 2 but there were coding ascii errors. Similar processing time 2 vs 4
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL # protocol 4

####################### 0 - RESIZE DICOMS
#
# Part of Full Preprocessing Tutorial to resample DICOMs to a certain isotropic
# resolution. Pixel spacing varies between pacients, this will resize the dataset.
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/notebook

global resize_dashboard

resize_dashboard={
    'INPUT_DIRECTORY' : COMPETITION_DATASET_DIRECTORY,
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/0_resize_dicoms/',
    'NEW_SPACING': [1, 1, 1],
}

global LUNA_ROOT, LUNA_DATASET_DIRECTORY, LUNA_RESIZED_DIRECTORY, LUNA_SUBFOLDERS, LUNA_CSV_DIRECTORY

LUNA_ROOT = COMPETITION_HOME + "LUNA16/" # OFFICIAL
# LUNA_ROOT = "../../../LUNA16/" # TEST AJR
LUNA_DATASET_DIRECTORY = LUNA_ROOT + "IMG_ORIG/"
LUNA_RESIZED_DIRECTORY = LUNA_ROOT + "IMG_111/"
LUNA_SUBFOLDERS = ["subset" + str(i) + "/" for i in range(10)]
LUNA_CSV_DIRECTORY = LUNA_ROOT + "CSVFILES/"
LUNA_ORIGIN_OF_IMAGES_FILE = LUNA_CSV_DIRECTORY + "origin_of_images.csv"


####################### 0 - RESIZE LUNA

global resize_LUNA_dashboard

resize_LUNA_dashboard={
    'LUNA_ROOT' : LUNA_ROOT,
    'INPUT_DIRECTORY' : LUNA_DATASET_DIRECTORY,
    'OUTPUT_DIRECTORY' : LUNA_RESIZED_DIRECTORY,
    'SUBFOLDERS' : LUNA_SUBFOLDERS,
    'NEW_SPACING': [1, 1, 1],
    'ORIGIN_OF_IMAGES_FILE': LUNA_ORIGIN_OF_IMAGES_FILE
}

####################### 1 - ARNAV'S LUNGS ROI WAY
#
# Candidate Generation and LUNA16 preprocessing (part 1)
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

global arnavs_lugns_roi_dashboard

arnavs_lugns_roi_dashboard={
    'INPUT_DIRECTORY' : resize_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/1_lungs_roi_arnav/',
    'EROSION_BALL_RADIUS': 2,
    'CLOSING_BALL_RADIUS': 10,
}

####################### 2 - Watershed LUNGS ROI WAY
#
# Improved Lung Segmentation using Watershed (Ankasor)
# https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed

global watershed_lugns_roi_dashboard

watershed_lugns_roi_dashboard={
    'INPUT_DIRECTORY' : resize_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/2_lungs_roi_watershed/',
}

####################### 3 - Best Lung ROI selector
#
# Improved Lung Segmentation using Watershed (Ankasor)
# https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed

global best_lug_roi_selector_dashboard

best_lug_roi_selector_dashboard={
    'INPUT_DIRECTORY_1' : arnavs_lugns_roi_dashboard['OUTPUT_DIRECTORY'],
    'INPUT_DIRECTORY_2' : watershed_lugns_roi_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/3_best_lung_roi_selector/',
}

####################### 4 - Inverted Lung Detector
#
# Improved Lung Segmentation using Watershed (Ankasor)
# https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed

global inverted_lung_detector_dashboard

inverted_lung_detector_dashboard={
    'INPUT_DIRECTORY' : best_lug_roi_selector_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/4_inverted_lung_detector/',
}


####################### 5 - Nodules ROI
#
# Candidate Generation and LUNA16 preprocessing (part 2)
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

global nodules_roi_dashboard

nodules_roi_dashboard={
    #'INPUT_DIRECTORY' : inverted_lung_detector_dashboard['INPUT_DIRECTORY'],
    'INPUT_DIRECTORY' : arnavs_lugns_roi_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/5_nodules_roi/',
    'BALL_RADIUS':  2,
}

####################### 6 - Nodules 3D segmentation (LUNA-version)
#
# Candidate Generation and LUNA16 preprocessing (part 2)
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

global nodules_3d_segmentation_luna

nodules_3d_segmentation_luna={
    #'INPUT_DIRECTORY' : inverted_lung_detector_dashboard['INPUT_DIRECTORY'],
    'INPUT_DIRECTORY' : nodules_roi_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/6_nodules_3D_segmentation_LUNA"016/',
}

####################### 7 - Nodules 3D segmentation
#
# Candidate Generation and LUNA16 preprocessing (part 2)
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

global nodules_3d_segmentation

nodules_3d_segmentation={
    #'INPUT_DIRECTORY' : inverted_lung_detector_dashboard['INPUT_DIRECTORY'],
    'INPUT_DIRECTORY' : nodules_roi_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/7_nodules_3D_segmentation/',
    'PRINT_IMAGES': False,
}

####################### 7B - reduce CSV
#
# reduce: joins all centerradius info from step 7 pickles, into a sligle CSV


global nodules_info_csv_from_step_7

nodules_info_csv_from_step_7={
    #'INPUT_DIRECTORY' : inverted_lung_detector_dashboard['INPUT_DIRECTORY'],
    'INPUT_DIRECTORY' : nodules_3d_segmentation['OUTPUT_DIRECTORY'],
    'OUTPUT_FILE' : COMPETITION_HOME + 'output/step-7-nodules.{}',
}

####################### 8 - 3D chunks extraction
#
CHUNK_SIDE = 64

LUNA_PREFIX = "ANN" # LUNA Annotations
#LUNA_PREFIX = "ANN_EX" # LUNA Annotations excluded

if LUNA_PREFIX == "ANN":
    LUNA_ANN_CSV_FILE = LUNA_CSV_DIRECTORY + "annotations.csv"
else:
    LUNA_ANN_CSV_FILE = LUNA_CSV_DIRECTORY + "annotations_excluded.csv"

LUNA_ANN = LUNA_ROOT + LUNA_PREFIX + str(CHUNK_SIDE) + "/"

chunks_extraction={
    'LUNA_ROOT' : LUNA_ROOT,
    'INPUT_DIRECTORY' : LUNA_RESIZED_DIRECTORY,
    'CHUNK_DIMS': (CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE),
    'PREFIX': LUNA_PREFIX,
    'ANN_CSV_FILE': LUNA_ANN_CSV_FILE,
    'OUTPUT_ANN_IMG_DIR' : LUNA_ANN,
    'SUBFOLDERS' : LUNA_SUBFOLDERS,
    'ORIGIN_OF_IMAGES_FILE': LUNA_ORIGIN_OF_IMAGES_FILE
}

####################### 8 - 3D DSB chunks extraction
#
DSB_ANN_CSV_FILE = nodules_info_csv_from_step_7['OUTPUT_FILE']


####################### 10 - Features Extraction Nodules 3D
#
# 
# 

global features_extraction_nodules_3d

features_extraction_nodules_3d={
    'INPUT_DIRECTORY_1' : nodules_roi_dashboard['OUTPUT_DIRECTORY'],
    'INPUT_DIRECTORY_2' : nodules_3d_segmentation['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/10_features_extraction_3D_nodules/',
}

####################### 13 - Features Extraction Lungs
#
# 
# 

global features_extraction_lungs_3d

features_extraction_lungs_3d={
    'INPUT_DIRECTORY'  : features_extraction_nodules_3d['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/13_features_extraction_lungs/',
}


####################### Extras and Cloud
#
#

AWS = False # To download form and upload to the AWS S3 bucket
BUCKET = 'kaggle-adri'

if AWS:
    import boto3
    s3 = boto3.client('s3')
    from matplotlib import use as pltuse
    pltuse('Agg', warn=False) # Avoid no display name and environment variable error on AWS

from random import shuffle
import os
import shutil

# Get what patients are left to process, pick randomnly a batch
BATCH_SIZE = 100

def batch_to_process(input_path, output_path, aws=False, input_is_folder=False):
    '''Take all files/folders to process, forget about the ones already processed'''
    # If input folder have folders with dicoms
    if input_is_folder:
        if aws:
            patients = read_from_s3(input_path, True)
            processed = [p.split('.')[0] for p in read_from_s3(output_path) if p.endswith('.pickle')]
        else:
            patients = os.listdir(input_path)
            processed = [p.split('.')[0] for p in os.listdir(output_path) if p.endswith('.pickle')]
    # If both folders have pickles. Forget images and other files
    else:
        if aws:
            patients = read_from_s3(input_path)
            processed = read_from_s3(output_path)
        else:
            patients = [p for p in os.listdir(input_path) if p.endswith('.pickle')]
            processed = [p for p in os.listdir(output_path) if p.endswith('.pickle')]
    
    to_process = [f for f in patients if f not in processed]
    
    print('There are ' + str(len(patients)) + ' patients. ' \
       + str(len(to_process)) + ' left to process')
    # Several python scripts can run in parallel. We will make shuffled batches
    # of a BATCH_SIZE until finalized.
    shuffle(to_process)
    if len(to_process) > BATCH_SIZE:
        return to_process[0:BATCH_SIZE]
    else:
        return to_process

def download_from_s3(input_path, input_is_folder=False):
    '''Download file from S3, if its a dicoms folder, download content'''
    s3_path = input_path[3:] # Don't need the '../' part
    if input_is_folder:
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        ls_objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path)
        to_download = [p['Key'].split('/')[-1] for p in ls_objects['Contents'] \
                      if p['Key'].split('/')[-1].endswith('.dcm')]
        for file in to_download:
            if os.path.isfile(input_path + '/' + file): continue
            download_from_s3(input_path + '/' + file)
    else:
        if os.path.isfile(input_path): return
        s3.download_file(BUCKET, s3_path, input_path)
        print(input_path, 'downloaded from S3')
        

def upload_to_s3(local_path):
    '''Upload to S3 bucket giving a local file path'''
    print('Uploading to S3', local_path)
    # local_path example '../output/0_resize_dicoms/', removing '../'
    s3.upload_file(local_path, BUCKET, local_path[3:])


def clean_after_upload(input_path, output_path, input_is_folder=False):
    '''Delete files/folders from input and output after successful uplaod to S3'''
    if input_is_folder:
        shutil.rmtree(input_path)
        print('Folder deleted:', input_path[3:])
    else:
        os.remove(input_path)
        print('Input file deleted:', input_path)
    os.remove(output_path)
    print('Output file deleted:', output_path)


def read_from_s3(path, input_is_folder=False):
    '''Read files (.pickles) or folders from an S3 path'''
    s3_path = path[3:] # Don't need the '../' part
    if input_is_folder:
        # We only want subfolders with dicoms, not dicoms, using '/' Delimiter
        ls_objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path, Delimiter='/')
        final_list = [f.get('Prefix').split('/')[1] for f in ls_objects['CommonPrefixes']]
        # If there are more than 1000 and less than 2000 files/folders
        if ls_objects['IsTruncated']:
            next_ls_objects = \
            s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path, Delimiter='/', \
                               ContinuationToken=ls_objects['NextContinuationToken'])
            final_list += \
            [f.get('Prefix').split('/')[1] for f in next_ls_objects['CommonPrefixes']]
        # List will contain paths like ['example/patient/', ...], we only want patient
        return final_list 
    
    else:
        ls_objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path)
        final_list = [p['Key'].split('/')[-1] for p in ls_objects['Contents'] \
                      if p['Key'].split('/')[-1].endswith('.pickle')]
        # If there are more than 1000 and less than 2000 files/folders
        if ls_objects['IsTruncated']:
                next_ls_objects = \
                s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path, Delimiter='/', \
                                   ContinuationToken=ls_objects['NextContinuationToken'])
                final_list += \
                [f.get('Prefix').split('/')[1] for f in next_ls_objects['CommonPrefixes']]
        # List will contain paths like ['example/patient.pickle', ...],
        # we only want patients.pickle, not images or other files
        return final_list
