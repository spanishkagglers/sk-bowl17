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


####################### 0 - RESIZE DICOMS
#
# Part of Full Preprocessing Tutorial to resample DICOMs to a certain isotropic
# resolution. Pixel spacing varies between pacients, this will resize the dataset.
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/notebook

global resize_dashboard

resize_dashboard={
    'INPUT_DIRECTORY' : COMPETITION_DATASET_DIRECTORY,
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/0_resize/',
    'NEW_SPACING': [1, 1, 1],
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
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/3_Best_Lung_ROI_selector/',
}

####################### 4 - Inverted Lung Detector
#
# Improved Lung Segmentation using Watershed (Ankasor)
# https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed

global inverted_lung_detector_dashboard

inverted_lung_detector_dashboard={
    'INPUT_DIRECTORY' : best_lug_roi_selector_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/4_Inverted_Lung_Detector/',
}


####################### 5 - Nodules ROI
#
# Candidate Generation and LUNA16 preprocessing (part 2)
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

global nodules_roi_dashboard

nodules_roi_dashboard={
    #'INPUT_DIRECTORY' : inverted_lung_detector_dashboard['INPUT_DIRECTORY'],
    'INPUT_DIRECTORY' : arnavs_lugns_roi_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/5_Nodules_ROI/',
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
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/6_Nodules_3D_segmentation_LUNA"016/',
}

####################### 7 - Nodules 3D segmentation
#
# Candidate Generation and LUNA16 preprocessing (part 2)
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

global nodules_3d_segmentation

nodules_3d_segmentation={
    #'INPUT_DIRECTORY' : inverted_lung_detector_dashboard['INPUT_DIRECTORY'],
    'INPUT_DIRECTORY' : nodules_roi_dashboard['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/7_Nodules_3D_segmentation/',
    'PRINT_IMAGES': False,
}

####################### 10 - Features Extraction Nodules 3D
#
# 
# 

global features_extraction_nodules_3d

features_extraction_nodules_3d={
    'INPUT_DIRECTORY_1' : nodules_roi_dashboard['OUTPUT_DIRECTORY'],
    'INPUT_DIRECTORY_2' : nodules_3d_segmentation['OUTPUT_DIRECTORY'],
    'OUTPUT_DIRECTORY' : COMPETITION_HOME + 'output/10_Features_Extraction_3D_Nodules/',
}

