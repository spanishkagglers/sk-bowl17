#################################################################################
## https://www.kaggle.com/c/data-science-bowl-2017
## Data Science Bowl 2017
## Santiago Mota
## santiago_mota@yahoo.es
## https://es.linkedin.com/in/santiagomota/en


# Can you improve lung cancer detection?

# In the United States, lung cancer strikes 225,000 people every year, and 
# accounts for $12 billion in health care costs. Early detection is critical to 
# give patients the best chance at recovery and survival.

# One year ago, the office of the U.S. Vice President spearheaded a bold new 
# initiative, the Cancer Moonshot, to make a decade's worth of progress in 
# cancer prevention, diagnosis, and treatment in just 5 years.

# In 2017, the Data Science Bowl will be a critical milestone in support of the 
# Cancer Moonshot by convening the data science and medical communities to 
# develop lung cancer detection algorithms.

# Using a data set of thousands of high-resolution lung scans provided by the 
# National Cancer Institute, participants will develop algorithms that 
# accurately determine when lesions in the lungs are cancerous. This will 
# dramatically reduce the false positive rate that plagues the current detection 
# technology, get patients earlier access to life-saving interventions, and give 
# radiologists more time to spend with their patients.

# This year, the Data Science Bowl will award $1 million in prizes to those who 
# observe the right patterns, ask the right questions, and in turn, create 
# unprecedented impact around cancer screening care and prevention. The funds 
# for the prize purse will be provided by the Laura and John Arnold Foundation.

# Visit DataScienceBowl.com to:
# • Sign up to receive news about the competition
# • Learn about the history of the Data Science Bowl and past competitions
# • Read our latest insights on emerging analytics techniques

# DSB 2017
# Acknowledgments

# The Data Science Bowl is presented by

# BAH and Kaggle
# Competition Sponsors

# Laura and John Arnold Foundation
# The Cancer Imaging Program of NCI
# American College of Radiology
# Amazon Web Services
# NVIDIA
# Data Support Providers

# National Lung Screening Trial
# The Cancer Imaging Archive
# Dr. Bram van Ginneken, Professor of Functional Image Analysis and his team at Radboud University Medical Center in Nijmegen
# Lahey Hospital & Medical Center
# University of Copenhagen
# Nicholas Petrick, Ph.D., Acting Director Division of Imaging, Diagnostics and Software Reliability Office of Science and Engineering Laboratories Center for Devices and Radiological Health U.S. Food and Drug Administration
# Supporting Organizations 

# Bayes Impact
# Black Data Processng Associates
# Code the Change
# Data Community DC
# DataKind
# Galvanize
# Great Minds in STEM
# Hortonworks
# INFORMS
# Lesbians Who Tech
# NSBE
# Society of Asian Scientists & Engineers
# Society of Women Engineers
# University of Texas Austin, Business Analytics Program,
# McCombs School of Business
# US Dept. of Health and Human Services
# US Food and Drug Administration
# Women in Technology
# Women of Cyberjutsu

# Started: 2:00 pm, Thursday 12 January 2017 UTC
# Ends   : 11:59 pm, Wednesday 12 April 2017 UTC (90 total days)
# Points : this competition awards standard ranking points
# Tiers  : this competition counts towards tiers


# Data Files
# File Name 	                  Available Formats
# stage1_labels.csv 	            .zip (27.17 kb)
# stage1_sample_submission.csv 	.zip (4.03 kb)
# data_password.txt 	            .zip (597 b)
# stage1 	                        .7z (66.88 gb)
#                                   .torrent (669.97 kb)
# sample_images 	                  .7z (781.39 mb)

# You only need to download one format of each file.
# Each has the same contents but use different packaging methods.

# In this dataset, you are given over a thousand low-dose CT images from 
# high-risk patients in DICOM format. Each image contains a series with multiple 
# axial slices of the chest cavity. Each image has a variable number of 
# 2D slices, which can vary based on the machine taking the scan and patient.

# The DICOM files have a header that contains the necessary information about 
# the patient id, as well as scan parameters such as the slice thickness.

# The competition task is to create an automated method capable of determining 
# whether or not the patient will be diagnosed with lung cancer within one year 
# of the date the scan was taken. The ground truth labels were confirmed by 
# pathology diagnosis.

# The images in this dataset come from many sources and will vary in quality. 
# For example, older scans were imaged with less sophisticated equipment. You 
# should expect the stage 2 data to be, on the whole, more recent and higher 
# quality than the stage 1 data (generally having thinner slice thickness). 
# Ideally, your algorithm should perform well across a range of image quality.

# Notes

# Use of external data is permitted in this competition, provided the data is 
# freely available. If you are using a source of external data, you must post 
# the source to the official external data forum thread no later than one week 
# prior to the deadline of the first stage.

# This is a two-stage competition. In order to appear on the final competition 
# leaderboard and receive ranking points, your team must make a submission 
# during both stages of the competition.

# Due to the large file size, Kaggle is beta testing use of BitTorrent as an 
# alternate means of download. The image archives are encrypted in order to 
# prevent outside access. Please do not share the decryption password. The large 
# stage1.7z archive hosted on BitTorrent is the same as the version available 
# for direct download.

# File Descriptions

# Each patient id has an associated directory of DICOM files. The patient id is 
# found in the DICOM header and is identical to the patient name. The exact 
# number of images will differ from case to case, varying according in the 
# number of slices. Images were compressed as .7z files due to the large size of 
# the dataset.

# stage1.7z                    - contains all images for the first stage of the 
#                                competition, including both the training and 
#                                test set. This is file is also hosted on 
#                                BitTorrent.
# stage1_labels.csv            - contains the cancer ground truth for the stage 
#                                1 training set images
# stage1_sample_submission.csv - shows the submission format for stage 1. You 
#                                should also use this file to determine which 
#                                patients belong to the leaderboard set of 
#                                stage 1.
# sample_images.7z             - a smaller subset set of the full dataset, 
#                                provided for people who wish to preview the 
#                                images before downloading the large file.
# data_password.txt            - contains the decryption key for the image files

# The DICOM standard is complex and there are a number of different tools to 
# work with DICOM files. You may find the following resources helpful for 
# managing the competition data:

# The lite version of OsiriX is useful for viewing images on OSX.
# pydicom: A package for working with images in python.
# oro.dicom: A package for working with images in R.
# Mango: A useful DICOM viewer for Windows users.

# The Hounsfield scale applies to medical-grade CT scans but not to cone beam 
# computed tomography (CBCT) scans.[1]
# Substance 	            HU
# Air 	                  −1000
# Lung 	                  −500
# Fat 	                  −100 to −50
# Water 	                   0
# CSF (Cerebrospinal fluid)    15
# Kidney 	                   30
# Blood 	                  +30 to +45
# Muscle 	                  +10 to +40
# Grey matter 	            +37 to +45
# White matter 	            +20 to +30
# Liver 	                  +40 to +60
# Soft Tissue, Contrast 	+100 to +300
# Bone 	                  +700 (cancellous bone) to +3000 (cortical bone)

################################################################################
## LIDC-IDRI
## https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

# Number of Studies : 1,308
# Number of Series  : 1,018 CT
#                     290 CR/DX
# Number of Patients: 1,010
# Number of Images  : 244,527
# Modalities        : CT (computed tomography)
#                     DX (digital radiography) 
#                     CR (computed radiography) 
# Image Size (GB)   : 124

library(xlsx)

# Read csv files
nodule <- read.xlsx("./data/LIDC-IDRI/lidc-idri nodule counts (6-23-2015).xlsx", 
                    sheetName="Sheet1")

patient_list <- read.xlsx("./data/LIDC-IDRI/list3.2.xls", sheetName="list3.2")

diagnosis <- read.xlsx("./data/LIDC-IDRI/tcia-diagnosis-data-2012-04-20.xls", 
                       sheetName="Diagnosis Truth")

## ETL csv files

# Explanations on nodule csv
# *   total number of lesions that received either a "nodule < 3mm" mark or a 
#     "nodule >= 3mm" mark from at least one of the four LIDC radiologists
# **  total number of lesions that received a "nodule >= 3mm" mark from at least 
#     one of the four LIDC radiologists (regardless of how the other 
#     radiologists marked the lesion)
# *** total number of lesions that received a "nodule < 3mm" mark from at least 
#     one of the four LIDC radiologists (with no radiologist assigning the 
#     lesion a "nodule >= 3mm" mark)

# Leave only first 4 columns
nodule <- nodule[, 1:4]

# Delete last row (total)
nodule <- nodule[1:108,]

# Change variables names
names(nodule)
# [1] "TCIA.Patent.ID"            "Total.Number.of.Nodules.." "Number.of.Nodules...3mm.."
# [4] "Number.of.Nodules..3mm..."

names(nodule) <- c('patient_id', 'total_nodules', 'nodules_bigger_3mm', 
                   'nodules_smaller_3mm')

# CHECK: Is it rigth 3mm? I think it could be 3 cm

# Change id to numeric
nodule$patient_id <- as.numeric(substr(as.character(nodule$patient_id), 11, 15))

summary(nodule)

# Change variables names
names(patient_list)
# [1] "case"      "scan"      "roi"       "volume"    "eq..diam." "x.loc."    "y.loc."    "slice.no."
# [9] "nodIDs"    "NA."       "NA..1"     "NA..2"     "NA..3"     "NA..4"     "NA..5" 

names(patient_list) <- c("patient_id", "scan", "roi", "volume", "eq_diam", 
                         "location_x", "location_y", "slice_number", 
                         "nodule_id_1", "nodule_id_2", "nodule_id_3", 
                         "nodule_id_4", "nodule_id_5", "nodule_id_6", 
                         "nodule_id_7")

# Change variables types
patient_list$patient_id  <- as.numeric(patient_list$patient_id)
patient_list$nodule_id_1 <- as.character(patient_list$nodule_id_1)
patient_list$nodule_id_2 <- as.character(patient_list$nodule_id_2)
patient_list$nodule_id_3 <- as.character(patient_list$nodule_id_3)
patient_list$nodule_id_4 <- as.character(patient_list$nodule_id_4)
patient_list$nodule_id_5 <- as.character(patient_list$nodule_id_5)
patient_list$nodule_id_6 <- as.character(patient_list$nodule_id_6)
patient_list$nodule_id_7 <- as.character(patient_list$nodule_id_7)

# Explanations on diagnosis csv
# TCIA Patient ID	"Diagnosis at the Patient Level
# 0=Unknown
# 1=benign or non-malignant disease
# 2= malignant, primary lung cancer
# 3 = malignant metastatic
# "	"Diagnosis Method
# 0 = unknown
# 1 = review of radiological images to show 2 years of stable nodule
# 2 = biopsy
# 3 = surgical resection
# 4 = progression or response"	Primary tumor site for metastatic disease	"Nodule 1
# Diagnosis at the Nodule Level 
# 0=Unknown
# 1=benign or non-malignant disease
# 2= malignant, primary lung cancer
# 3 = malignant metastatic)
# "	"Nodule 1
# Diagnosis Method at the Nodule Level
# 0 = unknown
# 1 = review of radiological images to show 2 years of stable nodule
# 2 = biopsy
# 3 = surgical resection
# 4 = progression or response
# "	"Nodule 2
# Diagnosis at the Nodule Level 
# 0=Unknown
# 1=benign or non-malignant disease
# 2= malignant, primary lung cancer
# 3 = malignant metastatic)
# "	"Nodule 2
# Diagnosis Method at the Nodule Level
# 0 = unknown
# 1 = review of radiological images to show 2 years of stable nodule
# 2 = biopsy
# 3 = surgical resection
# 4 = progression or response
# "	"Nodule 3
# Diagnosis at the Nodule Level 
# 0=Unknown
# 1=benign or non-malignant disease
# 2= malignant, primary lung cancer
# 3 = malignant metastatic)
# "	"Nodule 3
# Diagnosis Method at the Nodule Level
# 0 = unknown
# 1 = review of radiological images to show 2 years of stable nodule
# 2 = biopsy
# 3 = surgical resection
# 4 = progression or response
# "	"Nodule 4
# Diagnosis at the Nodule Level 
# 0=Unknown
# 1=benign or non-malignant disease
# 2= malignant, primary lung cancer
# 3 = malignant metastatic)
# "	"Nodule 4
# Diagnosis Method at the Nodule Level
# 0 = unknown
# 1 = review of radiological images to show 2 years of stable nodule
# 2 = biopsy
# 3 = surgical resection
# 4 = progression or response
# "	"Nodule 5
# Diagnosis at the Nodule Level 
# 0=Unknown
# 1=benign or non-malignant disease
# 2= malignant, primary lung cancer
# 3 = malignant metastatic)
# "	"Nodule 5
# Diagnosis Method at the Nodule Level
# 0 = unknown
# 1 = review of radiological images to show 2 years of stable nodule
# 2 = biopsy
# 3 = surgical resection
# 4 = progression or response
# "

names(diagnosis) <- c('patient_id', 'diagnosis', 'diagnosis_method', 
                      'primary_tumor_site',
                      'nodule_1_diagnosis', 'nodule_1_method', 
                      'nodule_2_diagnosis', 'nodule_2_method',
                      'nodule_3_diagnosis', 'nodule_3_method', 
                      'nodule_4_diagnosis', 'nodule_4_method', 
                      'nodule_5_diagnosis', 'nodule_5_method')

# Change variables types
diagnosis$patient_id <- as.numeric(substr(as.character(diagnosis$patient_id), 11, 15))

table(diagnosis$diagnosis)
#  0  1  2  3 
# 27 36 43 51

table(diagnosis$diagnosis_method)
#  0  1  2  3  4 
# 28 28 46 39 16 

# Some ETL on primary tumor site
diagnosis$primary_tumor_site <- as.character(diagnosis$primary_tumor_site)
diagnosis$primary_tumor_site <- tolower(diagnosis$primary_tumor_site)
diagnosis$primary_tumor_site <- gsub('prostate ', 'prostate', diagnosis$primary_tumor_site)
diagnosis$primary_tumor_site <- gsub('melanoma ', 'melanoma', diagnosis$primary_tumor_site)

table(diagnosis$primary_tumor_site)
