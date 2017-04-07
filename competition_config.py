#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:37:58 2017
@author: virilo
"""

global COMPETITION_HOME, COMPETITION_DATASET_DIRECTORY

AWS = False # To download form and upload to the AWS S3 bucket

RANDOM_SEED = 1234
# All routes must be ended by /

COMPETITION_HOME = '../'

CHUNK_VARIATION_SEP="#"

COMPETITION_DATASET_DIRECTORY = COMPETITION_HOME + 'stage1/'
# Comment previous and uncomment to use sample_images or mini_stage1 folders for testing
#COMPETITION_DATASET_DIRECTORY = COMPETITION_HOME + 'sample_images/' # 20 patients
#COMPETITION_DATASET_DIRECTORY = COMPETITION_HOME + 'mini_stage1/' # optional number of patients

import pickle
# Used protocol 2 but there were coding ascii errors. Similar processing time 2 vs 4
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL # protocol 4

import datetime


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

LUNA_ROOT = COMPETITION_HOME + "LUNA16/"
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

####################### 6 - LIDC ETL
#
# https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
# http://www.via.cornell.edu/lidc/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2176079/

global lidc_etl

lidc_etl={
    'DICOM_INPUT_DIRECTORY' : '../lidc/DOI/',
    'XML_INPUT_DIRECTORY' : '../lidc/LIDC-XML-only/',
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/6_lidc_etl/',
}

####################### 6B - reduce CSV
#
# reduce: joins all centerradius info from step 7 pickles, into a sligle CSV


global nodules_info_csv_from_step_6

nodules_info_csv_from_step_6={
    'INPUT_DIRECTORY' : lidc_etl['OUTPUT_DIRECTORY'],
    'OUTPUT_FILE' : COMPETITION_HOME + 'output/step-6-nodules.{}',
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
CHUNK_SIDE = 24
NEW_ANN_DIAMETER = 15.0
DIST_TH2 = 3**2 # hay ann y ann_ex de LUNA que son el mismo nodulo en distintos algoritmos; dist**2 < DIST_TH2

#LUNA_PREFIX = "ANN" # LUNA Annotations
#LUNA_PREFIX = "ANN_EX" # LUNA Annotations excluded
LUNA_PREFIX = "CAND" # LUNA candidates from non-nodule/healthy areas

if LUNA_PREFIX == "ANN":
    LUNA_ANN_CSV_FILE = "annotations.csv"
elif LUNA_PREFIX == "ANN_EX":
    LUNA_ANN_CSV_FILE = "annotations_excluded.csv" # deberia ser UNIQUE!!!
else:
    LUNA_ANN_CSV_FILE = "candidates_processed.csv"

LUNA_ANN_CSV_FILE_PATH = LUNA_CSV_DIRECTORY + LUNA_ANN_CSV_FILE
LUNA_ANN = LUNA_ROOT + LUNA_PREFIX + str(CHUNK_SIDE) + "/"

chunks_extraction={
    'LUNA_ROOT' : LUNA_ROOT,
    'INPUT_DIRECTORY' : LUNA_RESIZED_DIRECTORY,
    'CHUNK_DIMS': (CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE),
    'PREFIX': LUNA_PREFIX,
    'ANN_CSV_FILE': LUNA_ANN_CSV_FILE_PATH,
    'OUTPUT_ANN_IMG_DIR' : LUNA_ANN,
    'SUBFOLDERS' : LUNA_SUBFOLDERS,
    'ORIGIN_OF_IMAGES_FILE': LUNA_ORIGIN_OF_IMAGES_FILE,
    'NEW_ANN_DIAMETER' : NEW_ANN_DIAMETER,
    'DIST_TH2' : DIST_TH2
}

####################### 8 - 3D DSB chunks extraction
#
#DSB_CANDS_CSV_FILE = nodules_info_csv_from_step_7['OUTPUT_FILE']
DSB_CANDS_CSV_FILE = COMPETITION_HOME + 'output/step-7-nodules.pickle'
NEW_CAND_DIAMETER = 15.0
DIAM_TH = 50

chunks_DSB_extraction={
    'INPUT_DIRECTORY' : resize_dashboard['OUTPUT_DIRECTORY'],
    'INPUT_METADATA' : DSB_CANDS_CSV_FILE,
    'CHUNK_DIMS': (CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE),
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/8B-1_Chunks_for_CNN/',
    'NEW_CAND_DIAMETER' : NEW_CAND_DIAMETER,
    'DIST_TH2' : DIST_TH2,
    'DIAM_TH': DIAM_TH
}


DSB_ANN_CSV_FILE = nodules_info_csv_from_step_7['OUTPUT_FILE']

####################### 9 - 3D nodules Augmentation 
#

augment_luna = {
    'INPUT_DIRECTORY' : '../ANN24/', #'../luna/ANN24/',
    'OUTPUT_DIRECTORY' : '../ANN24rot/', #'../luna/ANN24rot/',
}


augment_dsbowl = {
    'INPUT_DIRECTORY' : chunks_DSB_extraction['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' :  COMPETITION_HOME + 'output/9_augmented-chunks_for_CNN/',
}

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


####################### 14 - Nodule 3D classifier
#
# 
# 

global nodule_3D_classifier

nodule_3D_classifier={
'i':'''
      _       _          _____     ______  
     | |     | |        |_   _|   / / __ \ 
   __| | __ _| |_ __ _    | |    / / |  | |
  / _` |/ _` | __/ _` |   | |   / /| |  | |
 | (_| | (_| | || (_| |  _| |_ / / | |__| |
  \__,_|\__,_|\__\__,_| |_____/_/   \____/
  
''',

    'INPUT_LUNA_NODULES_METADATA' : '../luna/annotations.csv',
    'LUNA_INPUT_DIRECTORY' : augment_luna['OUTPUT_DIRECTORY'],
    'LUNA_NON_NODULES_INPUT_DIRECTORY' : '../ANN_EX24/',
    'LUNA_OTHER_TISSUES_INPUT_DIRECTORY' : '../CAND24/',
    'BOWL_INPUT_DIRECTORY' : chunks_DSB_extraction['OUTPUT_DIRECTORY'],
    'BOWL_LABELS' : '../stage1_labels.csv',
    #'BOWL_PATIENTS': COMPETITION_DATASET_DIRECTORY,
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/14_Nodule_3D_classifier/',
    'TEMP_DIRECTORY': '/media/ramdisk1/',
    'USE_RAMDISK': True,
    
'i':'''                                            _             
                                                  (_)            
  _ __  _ __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
 | '_ \| '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
 | |_) | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
 | .__/|_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
 | |            | |                                         __/ |
 |_|            |_|                                        |___/ 

''',   
    'RADIUS_THRESHOLD_MM': 10, # heuristic: considers any nodule with radius bigger than RADIUS_THRESHOLD_MM as affected nodule 
    'CHUNK_SIZE': 24,
    'MAX_CHUNKS_PER_CLASS': 1200,
    'MAX_CHUNKS_TO_PREDICT': None, #50, # for testing only  ¡¡¡ MUST BE None !!!
    
    


'i':'''
                      _      _   _             _       _             
                     | |    | | | |           (_)     (_)            
  _ __ ___   ___   __| | ___| | | |_ _ __ __ _ _ _ __  _ _ __   __ _ 
 | '_ ` _ \ / _ \ / _` |/ _ \ | | __| '__/ _` | | '_ \| | '_ \ / _` |
 | | | | | | (_) | (_| |  __/ | | |_| | | (_| | | | | | | | | | (_| |
 |_| |_| |_|\___/ \__,_|\___|_|  \__|_|  \__,_|_|_| |_|_|_| |_|\__, |
                                                                __/ |
                                                               |___/ 
''',
    
    #'RESUME_TRAINING': None, #'CNN-5_classes_50_epochs_5_early_stopping_2017-03-27_07.09.12', #'CNN-20_epochs_5_early_stopping_2017-03-24_17.48.58',
    'USE_CLASSES': [
        'CLASS_LOWER_DIAMETER_NODULES',  # [ANN24.ZIP] LUNA chunks with original diameter <2cm (candidates.csv)
        'CLASS_HIGHER_DIAMETER_NODULES', # [ANN24.ZIP] LUNA chunks with original diameter >2cm (candidates.csv)
        'CLASS_SEGMENTED_FROM_NON_AFFECTED_LUNGS', # [] BOWL chunks from non-cancer lungs (7 - Nodules 3D segmentation)
        'CLASS_NON_NODULES', # [ANN_EX24.ZIP] from LUNA annotations_excluded.csv
        'CLASS_OTHER_TISSUES' # [CAND24.ZIP] from LUNA
    ],
    
    # MODEL_TRAINING
    'NUM_FOLDS': None,# (2,2), #(5,5),  # (2,5) means execute only 2 folds of 5
    #'NUM_CLASSES': autodetected, see bellow
    'BATCH_SIZE':100,
    'EPOCHS':55,
    'EARLY_STOPPING_ROUNDS':5,
    
    
    #OUTPUT_FILTER
    'CLASS_1_THRESHOLD':0.90,  # confussion matrix threeshold
}

del(nodule_3D_classifier['i']) #comments to be ingnored

i=0
nodule_3D_classifier['CLASSES_TXT']=""
for x in nodule_3D_classifier['USE_CLASSES']:
    nodule_3D_classifier[x]=i
    nodule_3D_classifier['CLASSES_TXT']+=str(x) + ": " + str(i) + "\n"
    i+=1

nodule_3D_classifier['CLASSES']=sorted(set([nodule_3D_classifier[x] for x in nodule_3D_classifier['USE_CLASSES']]))


if 'NUM_CLASSES' not in nodule_3D_classifier:
    nodule_3D_classifier['NUM_CLASSES']=len(nodule_3D_classifier['CLASSES'])


nodule_3D_classifier['USE_K_FOLD']=nodule_3D_classifier['NUM_FOLDS'] is not None and nodule_3D_classifier['NUM_FOLDS']!=(1,1) and nodule_3D_classifier['NUM_FOLDS']!=(0,0)



'''
            _          
           (_)         
  _ __ ___  _ ___  ___ 
 | '_ ` _ \| / __|/ __|
 | | | | | | \__ \ (__ 
 |_| |_| |_|_|___/\___|
                       
                     
'''

nodule_3D_classifier['DASHBOARD_ID']="CNN-{}_classes_{}_epochs_{}_early_stopping_{}".format(
    nodule_3D_classifier['NUM_CLASSES'],
    nodule_3D_classifier['EPOCHS'],
    nodule_3D_classifier['EARLY_STOPPING_ROUNDS'],
    str(datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
)

nodule_3D_classifier['EXECUTION_OUTPUT_DIRECTORY']=nodule_3D_classifier['OUTPUT_DIRECTORY']+nodule_3D_classifier['DASHBOARD_ID']+"/"
if not 'TEMP_DIRECTORY' in nodule_3D_classifier:
    nodule_3D_classifier['TEMP_DIRECTORY']=nodule_3D_classifier['EXECUTION_OUTPUT_DIRECTORY']


####################### 15 - Nodule 3D classifier
#
# 
# 

global nodule_based_lung_classifier

nodule_based_lung_classifier={
    'CNN':'CNN-3_classes_15_epochs_5_early_stopping_2017-03-25_19.31.23',
    'INPUT_DIRECTORY' : nodule_3D_classifier['OUTPUT_DIRECTORY'],
    'BOWL_LABELS' : '../stage1_labels.csv',
    'BOWL_PATIENTS': COMPETITION_DATASET_DIRECTORY,
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/15_nodule_based_lung_classifier/',
    'SCORE_BASE':0.2878788,
    'SCORE_AFFECTED':0.3878788,
    'SCORE_HEALTHY':0.1878788,
    'AFFECTED_THRESHOLD': 0.935,
    'HEALTHY_THRESHOLD': 0.07,
    
    
    
}

nodule_based_lung_classifier['EXECUTION_OUTPUT_DIRECTORY']=nodule_based_lung_classifier['OUTPUT_DIRECTORY']+nodule_based_lung_classifier['CNN']+"/"


####################### ALT-1: RESNET + XGBOOST
#
# 
# 


'''
Download from http://data.dmlc.ml/mxnet/models/imagenet-11k-place365-ch/
	
    resnet-50-0000.params
    
    resnet-50-symbol.json

and move to ../resnet-50
'''

global alternative_model_1a, alternative_model_1b, alternative_model_1c



alternative_model_1a={
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/alt_1_A_resnet_features/',
    'PRETRAINED_RESNET_50':'../resnet-50/resnet-50',
    'BOWL_LABELS' : '../stage1_labels.csv',
    'BOWL_CT_SCANS': COMPETITION_DATASET_DIRECTORY,
    #'BOWL_CT_SCANS_NPY':alternative_model_Z['OUTPUT_DIRECTORY'],
    'USE_GPU': True,
}

alternative_model_1b={
    'INPUT_DIRECTORY':alternative_model_1a['OUTPUT_DIRECTORY'], # resnet features
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/alt_1_B_decomposition/',
    'RANDOM_SEED': RANDOM_SEED,
    'RELU_AFTER_RESNET':False,
    'RELU_AFTER_PCA':False,
    'AVERAGE_RESNET_FEATURES':False, 
    'SLICES_PERCENTAGES':(None, None),
    'ABSOLUTE_VALUES':False,
    'WHITEN':False,
    'SVD_SOLVER':'auto',
}
alternative_model_1b['AVERAGE_PCA_FEATURES']=not alternative_model_1b['AVERAGE_RESNET_FEATURES']

alternative_model_1c={
    'INPUT_DIRECTORY':alternative_model_1b['OUTPUT_DIRECTORY'], # resnet features
    'BOWL_LABELS' : '../stage1_labels.csv',
    'BOWL_PATIENTS': COMPETITION_DATASET_DIRECTORY,
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/alt_1_C_xgboost/',
    'RANDOM_SEED': RANDOM_SEED,
}

'''
alternative_model_Z={
    'BOWL_CT_SCANS': COMPETITION_DATASET_DIRECTORY,
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/alt_Z_dicom_to_npy/',
}
'''

####################### Extras and Cloud
#
#

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
        # If there are more than 1000 files/folders, break after 5 iterations
        truncated = ls_objects['IsTruncated']
        cont_token = ls_objects.get('NextContinuationToken')
        i = 0
        while truncated:
            i += 1
            if i > 5: break
            next_ls_objects = \
            s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path, Delimiter='/', \
                               ContinuationToken=cont_token)
            final_list += \
            [f.get('Prefix').split('/')[1] for f in next_ls_objects['CommonPrefixes']]
            truncated = next_ls_objects['IsTruncated']
            cont_token = next_ls_objects.get('NextContinuationToken')
            # List will contain paths like ['example/patient/', ...], we only want patient

    else:
        ls_objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path)
        final_list = [p['Key'].split('/')[-1] for p in ls_objects['Contents'] \
                      if p['Key'].split('/')[-1].endswith('.pickle')]
        # If there are more than 1000 files/folders, break after 5 iterations
        truncated = ls_objects['IsTruncated']
        cont_token = ls_objects.get('NextContinuationToken')
        i = 0
        while truncated:
            i += 1
            if i > 5: break
            next_ls_objects = \
            s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_path, \
                               ContinuationToken=cont_token)
            final_list += \
            [p['Key'].split('/')[-1] for p in next_ls_objects['Contents'] \
             if p['Key'].split('/')[-1].endswith('.pickle')]
            truncated = next_ls_objects['IsTruncated']
            cont_token = next_ls_objects.get('NextContinuationToken')
            # List will contain paths like ['example/patient.pickle', ...],
            # we only want patients.pickle, not images or other files
    return final_list


#import pickle
#with open('src/alt_1_xgboost_resnet_features.pickle', 'rb') as handle:
#    z = pickle.load(handle)
