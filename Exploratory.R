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

# Some initial work
Sys.setenv(LANGUAGE="en")
set.seed(1967)

# Info session
sessionInfo()

# system("uptime")
system("w") 

# What Linux flavor are we using 
system("cat /etc/*release*")

# System CPU
system("lscpu")

# System Memory
system("cat /proc/meminfo")

# System Partitions
system("cat /proc/partitions")

# System kernel
system("cat /proc/version")

# Benchmark drive
system("dd if=/dev/zero of=test bs=64k count=16k conv=fdatasync; rm test")

# Drive organization
system("lsblk")

# Disk usage
system("df")

# Linux packages installed
# system("dpkg --list")

## The R installed packages
print(installed.packages()[, c(3)])

# The kernel configuration?
# system("/sbin/sysctl -a")


# Show elements working directory
ls()

# Gets you the current working directory
getwd()                    

# Lists all the files present in the current working directory
dir()

# Updates all packages
update.packages() 


################################################################################
library(ggplot2)
library(readr)

stage1_labels <- read_csv('./data/stage1_labels.csv')
summary(stage1_labels)
class(stage1_labels)

# Number of files: 1397
head(stage1_labels)

# Convert to data frame
stage1_labels <- as.data.frame(stage1_labels)

table(stage1_labels$cancer)
#    0    1 
# 1035  362 

# 26% of cases


### Directories analysis ###

# First directory
dir_base <- '.'

dir_base_names <- dir(dir_base)
length(dir_base_names)
# [1] 3
dir_base_names
# [1] "data" "Exploratory001.R" "Full_Preprocessing_Tutorial.ipynb"

# Second directory
dir_data <- './data'

dir_data_names <- dir(dir_data)
length(dir_data_names)
# [1] 7
dir_data_names
# [1] "data_password.txt"            "data_password.txt.zip"        "DSB3Tutorial-master"         
# [4] "sample_images"                "stage1"                       "stage1_labels.csv"           
# [7] "stage1_sample_submission.csv"

# Another directory
dir_sample_images <- './data/sample_images'

dir_sample_images_names <- dir(dir_sample_images)
length(dir_sample_images_names)
# [1] 20
dir_sample_images_names

# Another directory
dir_stage1 <- './data/stage1'

dir_stage1_names <- dir(dir_stage1)
length(dir_stage1_names)
# [1] 1595
dir_stage1_names

stage1_labels_id <- stage1_labels$id

# Test set
# ids in files without labels: 198
# Train: 1397 / Test: 198 / Total: 1595
test_ids  <- dir_stage1_names[!(dir_stage1_names %in% stage1_labels_id)]
train_ids <- dir_stage1_names[(dir_stage1_names %in% stage1_labels_id)]

# Sample submission
stage1_sample_submission <- read_csv('./data/stage1_sample_submission.csv')

library(oro.dicom)
library(oro.nifti)
setwd("./data/stage1")

# One file "026470d51482c93efc18b9803159c960"
dicom_0001 <- readDICOM(test_ids[1])

str(dicom_0001$hdr)

dicom_0001$hdr[1]
# hdr_0001 <- as.data.frame(unlist(dicom_0001$hdr))

dim(dicom_0001$img[[11]])
# [1] 512 512

class(dicom_0001)

str(dicom_0001$hdr)
str(dicom_0001$img)

summary(dicom_0001$img)

hdr_0001 <- dicom_0001$hdr[[11]]
hdr_0001[hdr_0001$name == "PixelSpacing", "value"]
# [1] "0.781250 0.781250"

summary(hdr_0001)
str(hdr_0001)
hdr_0001$name
hdr_0001$value

dicom_0001$img

range(dicom_0001$img)
# [1] -1024  2575

dicom_0001_2 <- dicom_0001

# From DICOM to NIfTI
nii_0001   <- dicom2nifti(dicom_0001)

# Dimension
d <- dim(nii_0001)
d
# [1] 512 512 175

class(nii_0001)
# [1] "nifti”

image(1:d[1], 1:d[2], nii_0001[,, 11], col=gray(0:64/64), xlab="", ylab="")
image(1:d[1], 1:d[2], nii_0001[,, 100], col=gray(0:64/64), xlab="", ylab="")
image(1:d[1], 1:d[2], nii_0001[,, 150], col=gray(0:64/64), xlab="", ylab="")

# image(nii_0001)

# All Planes: Coronal, Sagittal, Axial
orthographic(nii_0001, xyz=c(330, 215, 124))

# Visualizing All Slices
par(mfrow=c(1, 2))
o <- par(mar=c(4,4,0,0))
hist(nii_0001, breaks = 75, prob=T, xlab="T1 intensities", 
     col=rgb(0,0,1,1/2), main="")
hist(nii_0001[nii_0001 > 20], breaks = 75, prob=T, xlab="T1 intensities > 20",
     col=rgb(0,0,1,1/2), main="")


# Backmapping One Slice
is_btw_300_400 <- ((nii_0001>300) & (nii_0001<400))
nii_0001_mask  <- nii_0001
nii_0001_mask[!is_btw_300_400] <- NA
overlay(nii_0001, nii_0001_mask, z=124, plot.type="single")

# Backmapping All Slices
overlay(nii_0001, nii_0001_mask)

# Back Mapping Orthographic
orthographic(nii_0001, nii_0001_mask, xyz=c(330, 215, 124), 
             text="Image overlaid with mask", text.cex = 1.5)


# Define a different transfer function
# This defines a linear spline. Other definitions are possible
linear.spline <- function(x, knots, slope) {   # lin.sp
      knots   <- c(min(x), knots, max(x))
      slopeS  <- slope[1]
      for(j in 2:length(slope)){slopeS <- c(slopeS, slope[j]-sum(slopeS))}
      rvals  <- numeric(length(x))
      for(i in 2:length(knots))
      {rvals<-ifelse(x>=knots[i-1], slopeS[i-1]*(x-knots[i-1])+rvals, rvals)}
      return(rvals)}
# Define a spline with two knots and three slopes
knot.vals <- c(.3,.6)
slp.vals  <- c(1,.5,.25)

# Plot the spline transfer function
par(new = TRUE)
curve(linear.spline(x,knot.vals, slp.vals), axes=FALSE, xlab="", ylab="", col=2, lwd=3)
axis(side=4, at = pretty(range(im_hist$mids))/max(nii_0001), 
     labels=pretty(range(im_hist$mids)))
mtext("Transformed Intensity", side=4, line=2)

# Plot the spline transfer function
trans_T1 <- linear.spline(nii_0001, knot.vals*max(nii_0001), slp.vals)
image(nii_0001, z=150, plot.type='single', main="Original Image")
image(trans_T1, z=150, plot.type='single', main="Transformed Image")

# Smoothing
# Smooth the image with a Gaussian smoother (~ 1 minute)
library(AnalyzeFMRI)
smooth_T1 <- GaussSmoothArray(nii_0001, voxdim=c(1,1,1), ksize=11, sigma=diag(3,3),
                              mask=NULL, var.norm=FALSE)
orthographic(smooth_T1)

# brainR
library(brainR)
imgs <- paste("Visit_", 1:5, "_8mm.nii.gz", sep="")
imgs
files <- sapply(imgs, system.file, package='brainR')
files
class(files)
scene_4d <- scene4d(files, levels=rep(0.99, length(files)), 
                    color= rep("blue", length(files)), useTemp=TRUE,
                    MNITemp = "8mm", alpha = rep(1, length(files)), 
                    rescale=TRUE)  
scene_4d

# Brain Template from Copyright (C) 1993-2009 Louis Collins,
# McConnell Brain Imaging Centre,
# Montreal Neurological Institute, McGill University
# 6th generation non-linear symmetric brain
# Downsampled to 8mm using FSL fslmaths -subsamp2

template <- readNIfTI(system.file("MNI152_T1_8mm_brain.nii.gz", package="brainR")
                      , reorient=FALSE)
dtemp <- dim(template)
### 4500 - value that empirically value that presented a brain with gyri
### lower values result in a smoother surface
brain <- contour3d(template, x=1:dtemp[1], y=1:dtemp[2],
                   z=1:dtemp[3], level = 4500, alpha = 0.8, draw = FALSE)

### Example data courtesy of Daniel Reich
### Each visit is a binary mask of lesions in the brain
imgs  <- paste("Visit_", 1:5, "_8mm.nii.gz", sep="")
imgs
files <- sapply(imgs, system.file, package='brainR')
files
scene <- list(brain)
scene
## loop through images and thresh
nimgs <- length(imgs)
cols  <- rainbow(nimgs)
for (iimg in 1:nimgs) {
      mask <- readNIfTI(files[iimg], reorient=FALSE)
      if (length(dim(mask)) > 3) mask <- mask[,,,1]
      ### use 0.99 for level of mask - binary
      activation <- contour3d(mask, level = c(0.99), alpha = 1,
                              add = TRUE, color=cols[iimg], draw=FALSE)
      ## add these triangles to the list
      scene <- c(scene, list(activation))
}
## make output image names from image names
fnames  <- c("brain.stl", gsub(".nii.gz", ".stl", imgs, fixed=TRUE))
outfile <- "index_4D_stl.html"
write4D(scene=scene, fnames=fnames, outfile=outfile, standalone=TRUE, rescale=TRUE)
browseURL(outfile)


################################################################################

setwd("./data/stage1")

# One file "0015ceb851d7251b8f399e39779d1e7d"
# dicom_0001 <- readDICOM(test_ids[4])

# dicom_names  <- dicom_0001$hdr[[11]]$name
# dicom_values <- dicom_0001$hdr[[11]]$value

library(stringr)
detect.spaces <- function(x) {str_detect(x, ' ')}

# test_dicom_hdr <- data.frame(dicom_values, stringsAsFactors = FALSE)
# test_dicom_hdr <- transpose(test_dicom_hdr)
# names(test_dicom_hdr) <- dicom_names

# test_dicom_names  <- data.frame(names='', stringsAsFactors = FALSE)
# test_dicom_values <- data.frame(values='', stringsAsFactors = FALSE)

test_dicom_names  <- list()
test_dicom_values <- list()

for (i in 1:198) {
      print(i)
      temp_dicom         <- readDICOM(test_ids[i])
      temp_dicom_names   <- temp_dicom$hdr[[11]]$name
      temp_dicom_values  <- temp_dicom$hdr[[11]]$value
      # single_names  <- dicom_names[!(detect.spaces(dicom_values))]
      # single_values <- dicom_values[single_index]
      # temp_test_dicom_hdr <- data.frame(temp_dicom_values, 
      #                                   stringsAsFactors = FALSE)
      # temp_test_dicom_hdr <- transpose(temp_test_dicom_hdr)
      # names(temp_test_dicom_hdr) <- temp_dicom_names
      # test_dicom_hdr[i,] <- temp_dicom_values
      # test_dicom_hdr <- merge(test_dicom_hdr, temp_test_dicom_hdr, 
      #                         by = 'PatientID')
      test_dicom_names[[i]]  <- temp_dicom_names
      test_dicom_values[[i]] <- temp_dicom_values
}

save(test_dicom_hdr, file='../test_dicom_hdr.RData')
save(test_dicom_names, file='../test_dicom_names.RData')
save(test_dicom_values, file='../test_dicom_values.RData')

# single_names    <- dicom_names[!(detect.spaces(dicom_values))]
# single_values   <- dicom_values[!(detect.spaces(dicom_values))]
# single_index    <- (1:47)[!(detect.spaces(dicom_values))]

train_dicom_names  <- list()
train_dicom_values <- list()

for (i in 1:1397) {
      # for (i in 1:5) {
      print(i)
      temp_dicom         <- readDICOM(train_ids[i])
      temp_dicom_names   <- temp_dicom$hdr[[11]]$name
      temp_dicom_values  <- temp_dicom$hdr[[11]]$value
      train_dicom_names[[i]]  <- temp_dicom_names
      train_dicom_values[[i]] <- temp_dicom_values
}

# save(train_dicom_hdr, file='../train_dicom_hdr.RData')
save(train_dicom_names, file='../train_dicom_names.RData')
save(train_dicom_values, file='../train_dicom_values.RData')

save(train_dicom_hdr, file='../train_dicom_hdr.RData')
summary(train_dicom_hdr)
str(train_dicom_hdr)

################################################################################
load("./data/train_dicom_names.RData")
load("./data/train_dicom_values.RData")
load("./data/test_dicom_names.RData")
load("./data/test_dicom_values.RData")


# Number of values in test
table(as.data.frame(summary(test_dicom_names))$Freq, useNA = 'ifany')
# 41        42        43        44        46        47        52 
#  8        79        40        14        18        36         2 
# 53 character    -none- 
#  1       198       198 

# Number of values in train
table(as.data.frame(summary(train_dicom_names))$Freq, useNA = 'ifany')
# 41        42        43        44        45        46        47 
# 61       590       235        68        21       138       270 
# 48        49        52        53        54 character    -none- 
#  2         1         8         2         1      1397      1397 

train_dicom_names_unique <- unique(unlist(train_dicom_names))
test_dicom_names_unique  <- unique(unlist(test_dicom_names))
total_dicom_names_unique <- unique(c(train_dicom_names_unique, test_dicom_names_unique))

total_dicom_names_unique[!(total_dicom_names_unique %in% train_dicom_names_unique)]
# character(0)
total_dicom_names_unique[!(total_dicom_names_unique %in% test_dicom_names_unique)]
# [1] "ContrastAllergies"          "LossyImageCompressionRatio"

# library(data.table)
# test_dicom_hdr <- data.frame(names=rep('', 68), stringsAsFactors = FALSE)
# test_dicom_hdr <- transpose(test_dicom_hdr)
# names(test_dicom_hdr) <- total_dicom_names_unique

# temp_GroupLength <- rep('', 198)
# for (i in 1:198) {temp_GroupLength[i] <- unlist(test_dicom_values[[i]])[unlist(test_dicom_names[[i]]) == 'GroupLength']}
# temp_GroupLength

variables_to_ignore <- c('GroupLength')
total_names_procesed <- total_dicom_names_unique[!(total_dicom_names_unique %in% variables_to_ignore)]

test_hdr <- matrix(data = '', nrow = 198, ncol = length(total_names_procesed))
test_hdr <- data.frame(test_hdr, stringsAsFactors = FALSE)
names(test_hdr) <- total_names_procesed

for (j in 1:length(total_names_procesed)) {
      for (i in 1:198) {
            temp <- unlist(test_dicom_values[[i]])[unlist(test_dicom_names[[i]]) == total_names_procesed[j]]
            if(length(temp)>0) {test_hdr[i, j] <- temp} 
      }
      # test_hdr
}
      

train_hdr <- matrix(data = '', nrow = 1397, ncol = length(total_names_procesed))
train_hdr <- data.frame(train_hdr, stringsAsFactors = FALSE)
names(train_hdr) <- total_names_procesed

for (j in 1:length(total_names_procesed)) {
      for (i in 1:1397) {
            temp <- unlist(train_dicom_values[[i]])[unlist(train_dicom_names[[i]]) == total_names_procesed[j]]
            if(length(temp)>0) {train_hdr[i, j] <- temp} 
      }
      # train_hdr
}

test_hdr[, 1]

# Include Train/Test set
test_hdr$set  <- "TEST"
train_hdr$set <- "TRAIN"

save(test_hdr, file = './data/test_hdr.RData')
save(train_hdr, file = './data/train_hdr.RData')

# Create a total set
total_hdr <- rbind(train_hdr, test_hdr)

str(total_hdr)

# Change to factor
total_hdr$set <- as.factor(total_hdr$set)

total_hdr$RescaleType[total_hdr$RescaleType==''] <- NA
total_hdr$RescaleType <- as.factor(total_hdr$RescaleType)
table(total_hdr$RescaleType, useNA = 'ifany')

total_hdr$TransferSyntaxUID[total_hdr$TransferSyntaxUID==''] <- NA
total_hdr$TransferSyntaxUID <- as.factor(total_hdr$TransferSyntaxUID)
table(total_hdr$TransferSyntaxUID, useNA = 'ifany')
# 1.2.840.10008.1.2 1.2.840.10008.1.2.1 
#               302                1293

total_hdr$ImplementationClassUID[total_hdr$ImplementationClassUID==''] <- NA
total_hdr$ImplementationClassUID <- as.factor(total_hdr$ImplementationClassUID)
table(total_hdr$ImplementationClassUID, useNA = 'ifany')
# 1.2.276.0.7230010.3.0.3.6.0           1.2.40.0.13.1.1.1     1.3.6.1.4.1.22213.1.143 
#                          14                        1552                          29

total_hdr$ImplementationVersionName[total_hdr$ImplementationVersionName==''] <- NA
total_hdr$ImplementationVersionName <- as.factor(total_hdr$ImplementationVersionName)
table(total_hdr$ImplementationVersionName, useNA = 'ifany')
# 0.5  dcm4che-1.4.31 OFFIS_DCMTK_360 
#  29            1552              14

total_hdr$MediaStorageSOPInstanceUID[total_hdr$MediaStorageSOPInstanceUID==''] <- NA
total_hdr$MediaStorageSOPInstanceUID <- as.factor(total_hdr$MediaStorageSOPInstanceUID)
table(total_hdr$MediaStorageSOPInstanceUID, useNA = 'ifany')

total_hdr$SOPInstanceUID[total_hdr$SOPInstanceUID==''] <- NA
total_hdr$SOPInstanceUID <- as.factor(total_hdr$SOPInstanceUID)
table(total_hdr$SOPInstanceUID, useNA = 'ifany')

total_hdr$StudyInstanceUID[total_hdr$StudyInstanceUID==''] <- NA
total_hdr$StudyInstanceUID <- as.factor(total_hdr$StudyInstanceUID)
table(total_hdr$StudyInstanceUID, useNA = 'ifany')

total_hdr$SeriesInstanceUID[total_hdr$SeriesInstanceUID==''] <- NA
total_hdr$SeriesInstanceUID <- as.factor(total_hdr$SeriesInstanceUID)
table(total_hdr$SeriesInstanceUID, useNA = 'ifany')

total_hdr$SeriesNumber[total_hdr$SeriesNumber==''] <- NA
total_hdr$SeriesNumber <- as.factor(total_hdr$SeriesNumber)
table(total_hdr$SeriesNumber, useNA = 'ifany')

total_hdr$AcquisitionNumber[total_hdr$AcquisitionNumber==''] <- NA
total_hdr$AcquisitionNumber <- as.factor(total_hdr$AcquisitionNumber)
table(total_hdr$AcquisitionNumber, useNA = 'ifany')

total_hdr$InstanceNumber[total_hdr$InstanceNumber==''] <- NA
total_hdr$InstanceNumber <- as.factor(total_hdr$InstanceNumber)
table(total_hdr$InstanceNumber, useNA = 'ifany')

total_hdr$FrameOfReferenceUID[total_hdr$FrameOfReferenceUID==''] <- NA
total_hdr$FrameOfReferenceUID <- as.factor(total_hdr$FrameOfReferenceUID)
table(total_hdr$FrameOfReferenceUID, useNA = 'ifany')

total_hdr$PixelPaddingValue[total_hdr$PixelPaddingValue==''] <- NA
total_hdr$PixelPaddingValue <- as.factor(total_hdr$PixelPaddingValue)
table(total_hdr$PixelPaddingValue, useNA = 'ifany')
#         0 -1024 -2000 63536  8240 
# 671    77    14    73   756     4 

total_hdr$BurnedInAnnotation[total_hdr$BurnedInAnnotation==''] <- NA
total_hdr$BurnedInAnnotation <- as.factor(total_hdr$BurnedInAnnotation)
table(total_hdr$BurnedInAnnotation, useNA = 'ifany')
#      NO 
# 988 607

total_hdr$LongitudinalTemporalInformationModified[total_hdr$LongitudinalTemporalInformationModified==''] <- NA
total_hdr$LongitudinalTemporalInformationModified <- as.factor(total_hdr$LongitudinalTemporalInformationModified)
table(total_hdr$LongitudinalTemporalInformationModified, useNA = 'ifany')
#     MODIFIED 
# 973      622

total_hdr$'WindowCenter&WidthExplanation'[total_hdr$'WindowCenter&WidthExplanation'==''] <- NA
total_hdr$'WindowCenter&WidthExplanation' <- as.factor(total_hdr$'WindowCenter&WidthExplanation')
table(total_hdr$'WindowCenter&WidthExplanation', useNA = 'ifany')
#      WINDOW1 WINDOW2 
# 1087             508

total_hdr$PatientOrientation[total_hdr$PatientOrientation==''] <- NA
total_hdr$PatientOrientation <- as.factor(total_hdr$PatientOrientation)
table(total_hdr$PatientOrientation, useNA = 'ifany')
#       L P 
# 1494  101

total_hdr$PlanarConfiguration[total_hdr$PlanarConfiguration==''] <- NA
total_hdr$PlanarConfiguration <- as.factor(total_hdr$PlanarConfiguration)
table(total_hdr$PlanarConfiguration, useNA = 'ifany')
#         0 
# 1549   46 

total_hdr$LossyImageCompression[total_hdr$LossyImageCompression==''] <- NA
total_hdr$LossyImageCompression <- as.factor(total_hdr$LossyImageCompression)
table(total_hdr$LossyImageCompression, useNA = 'ifany')
#        00   01 
# 1571   23    1 

total_hdr$NumberOfFrames[total_hdr$NumberOfFrames==''] <- NA
total_hdr$NumberOfFrames <- as.factor(total_hdr$NumberOfFrames)
table(total_hdr$NumberOfFrames, useNA = 'ifany')
#         1 
# 1577   18

total_hdr$ImageDimensions[total_hdr$ImageDimensions==''] <- NA
total_hdr$ImageDimensions <- as.factor(total_hdr$ImageDimensions)
table(total_hdr$ImageDimensions, useNA = 'ifany')
#         2 
# 1436  159

total_hdr$ImageFormat[total_hdr$ImageFormat==''] <- NA
total_hdr$ImageFormat <- as.factor(total_hdr$ImageFormat)
table(total_hdr$ImageFormat, useNA = 'ifany')
#      RECT 
# 1436  159 

total_hdr$CompressionCode[total_hdr$CompressionCode==''] <- NA
total_hdr$CompressionCode <- as.factor(total_hdr$CompressionCode)
table(total_hdr$CompressionCode, useNA = 'ifany')
#      NONE 
# 1436  159

total_hdr$ImageLocation[total_hdr$ImageLocation==''] <- NA
total_hdr$ImageLocation <- as.factor(total_hdr$ImageLocation)
table(total_hdr$ImageLocation, useNA = 'ifany')
#      32736 
# 1436   159

total_hdr$PixelAspectRatio[total_hdr$PixelAspectRatio==''] <- NA
total_hdr$PixelAspectRatio <- as.factor(total_hdr$PixelAspectRatio)
table(total_hdr$PixelAspectRatio, useNA = 'ifany')
#       1 1 
# 1552   43 

total_hdr$SourceImageSequence[total_hdr$SourceImageSequence==''] <- NA
total_hdr$SourceImageSequence <- as.factor(total_hdr$SourceImageSequence)
table(total_hdr$SourceImageSequence, useNA = 'ifany')
#      Sequence 
# 1581       14 

total_hdr$ReferencedSOPClassUID[total_hdr$ReferencedSOPClassUID==''] <- NA
total_hdr$ReferencedSOPClassUID <- as.factor(total_hdr$ReferencedSOPClassUID)
table(total_hdr$ReferencedSOPClassUID, useNA = 'ifany')
#      1.2.840.10008.5.1.4.1.1.2 
# 1581                        14 

total_hdr$VolumetricProperties[total_hdr$VolumetricProperties==''] <- NA
total_hdr$VolumetricProperties <- as.factor(total_hdr$VolumetricProperties)
table(total_hdr$VolumetricProperties, useNA = 'ifany')
#      VOLUME 
# 1581     14

change.to.factor <- function(x) {
      x[x==''] <- NA
      x <- as.factor(x)
      # print(table(x, useNA = 'ifany'))
}

# codetools::findGlobals(change.to.factor)



total_hdr$TemporalPositionIndex[total_hdr$TemporalPositionIndex==''] <- NA
total_hdr$TemporalPositionIndex <- as.factor(total_hdr$TemporalPositionIndex)
table(total_hdr$TemporalPositionIndex, useNA = 'ifany')
#         1 
# 1581   14 
# total_hdr$TemporalPositionIndex <- change.to.factor(total_hdr$TemporalPositionIndex)

total_hdr$FrameIncrementPointer[total_hdr$FrameIncrementPointer==''] <- NA
total_hdr$FrameIncrementPointer <- as.factor(total_hdr$FrameIncrementPointer)
table(total_hdr$FrameIncrementPointer, useNA = 'ifany')
#      \030\005 
# 1581       14
# total_hdr$FrameIncrementPointer <- change.to.factor(total_hdr$FrameIncrementPointer)

total_hdr$Unknown[total_hdr$Unknown==''] <- NA
total_hdr$Unknown <- as.factor(total_hdr$Unknown)
table(total_hdr$Unknown, useNA = 'ifany')
#      524300 
# 1581     14
# total_hdr$Unknown <- change.to.factor(total_hdr$Unknown)

total_hdr$SourceApplicationEntityTitle[total_hdr$SourceApplicationEntityTitle==''] <- NA
total_hdr$SourceApplicationEntityTitle <- as.factor(total_hdr$SourceApplicationEntityTitle)
table(total_hdr$SourceApplicationEntityTitle, useNA = 'ifany')
#      POSDA 
# 1566    29
# total_hdr$SourceApplicationEntityTitle <- change.to.factor(total_hdr$SourceApplicationEntityTitle)

total_hdr$WindowCenter[total_hdr$WindowCenter==''] <- NA
total_hdr$WindowCenter <- as.factor(total_hdr$WindowCenter)
table(total_hdr$WindowCenter, useNA = 'ifany')
# total_hdr$WindowCenter <- change.to.factor(total_hdr$WindowCenter)

total_hdr$WindowWidth[total_hdr$WindowWidth==''] <- NA
total_hdr$WindowWidth <- as.factor(total_hdr$WindowWidth)
table(total_hdr$WindowWidth, useNA = 'ifany')
# total_hdr$WindowWidth <- change.to.factor(total_hdr$WindowWidth)



# Change to numeric
total_hdr$LargestImagePixelValue <- as.numeric(total_hdr$LargestImagePixelValue)
table(total_hdr$LargestImagePixelValue, useNA = 'ifany')

total_hdr$SmallestImagePixelValue <- as.numeric(total_hdr$SmallestImagePixelValue)
table(total_hdr$SmallestImagePixelValue, useNA = 'ifany')

total_hdr$RescaleSlope <- as.numeric(total_hdr$RescaleSlope)
table(total_hdr$RescaleSlope, useNA = 'ifany')

total_hdr$RescaleIntercept <- as.numeric(total_hdr$RescaleIntercept)
table(total_hdr$RescaleIntercept, useNA = 'ifany')

# total_hdr$WindowWidth <- as.numeric(total_hdr$WindowWidth)
# table(total_hdr$WindowWidth, useNA = 'ifany')

# total_hdr$WindowCenter <- as.numeric(total_hdr$WindowCenter)
# table(total_hdr$WindowCenter, useNA = 'ifany')

total_hdr$BitsStored <- as.numeric(total_hdr$BitsStored)
table(total_hdr$BitsStored, useNA = 'ifany')
#  12  16 
# 639 956

total_hdr$HighBit <- as.numeric(total_hdr$HighBit)
table(total_hdr$HighBit, useNA = 'ifany')
#  11  15 
# 639 956

total_hdr$PixelRepresentation <- as.numeric(total_hdr$PixelRepresentation)
table(total_hdr$PixelRepresentation, useNA = 'ifany')
#   0   1 
# 639 956 






# To eliminate (not information)
table(total_hdr$PixelData, useNA = 'ifany')
# PixelData 
#      1595
total_hdr$PixelData <- NULL

table(total_hdr$kVp, useNA = 'ifany')
# 
# 1595 
total_hdr$kVp <- NULL

table(total_hdr$Item, useNA = 'ifany')
# 
# 1595 
total_hdr$Item <- NULL

table(total_hdr$Laterality, useNA = 'ifany')
# 
# 1595
total_hdr$Laterality <- NULL

table(total_hdr$ContrastAllergies, useNA = 'ifany')
# 
# 1595
total_hdr$ContrastAllergies <- NULL

table(total_hdr$PregnancyStatus, useNA = 'ifany')
# 
# 1595
total_hdr$PregnancyStatus <- NULL

table(total_hdr$FileMetaInformationVersion, useNA = 'ifany')
# \001 
# 1595
total_hdr$FileMetaInformationVersion <- NULL

table(total_hdr$PatientsBirthDate, useNA = 'ifany')
# 19000101 
#     1595
total_hdr$PatientsBirthDate <- NULL

table(total_hdr$LossyImageCompressionRatio, useNA = 'ifany')
#      5.5935388 
# 1594         1
total_hdr$LossyImageCompressionRatio <- NULL

table(total_hdr$MediaStorageSOPClassUID, useNA = 'ifany')
# 1.2.840.10008.5.1.4.1.1.2 
#                      1595
total_hdr$MediaStorageSOPClassUID <- NULL

table(total_hdr$SpecificCharacterSet, useNA = 'ifany')
# ISO_IR 100 
#       1595
total_hdr$SpecificCharacterSet <- NULL

table(total_hdr$SOPClassUID, useNA = 'ifany')
# 1.2.840.10008.5.1.4.1.1.2 
#                      1595
total_hdr$SOPClassUID <- NULL

table(total_hdr$Modality, useNA = 'ifany')
#   CT 
# 1595
total_hdr$Modality <- NULL

table(total_hdr$SeriesDescription, useNA = 'ifany')
# Axial 
#  1595 
total_hdr$SeriesDescription <- NULL

table(total_hdr$ImageOrientationPatient, useNA = 'ifany')
# 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000                               1.0 0.0 0.0 0.0 1.0 0.0 
#                                                  1016                                                     1 
# 1 0 0 0 1 0 
#         578 
total_hdr$ImageOrientationPatient <- NULL

table(total_hdr$PositionReferenceIndicator, useNA = 'ifany')
#      SN 
# 671 924
total_hdr$PositionReferenceIndicator <- NULL

table(total_hdr$SamplesperPixel, useNA = 'ifany')
#    1 
# 1595
total_hdr$SamplesperPixel <- NULL

table(total_hdr$PhotometricInterpretation, useNA = 'ifany')
# MONOCHROME2 
#        1595
total_hdr$PhotometricInterpretation <- NULL

table(total_hdr$Rows, useNA = 'ifany')
#  512 
# 1595 
total_hdr$Rows <- NULL

table(total_hdr$Columns, useNA = 'ifany')
#  512 
# 1595 
total_hdr$Columns <- NULL

table(total_hdr$BitsAllocated, useNA = 'ifany')
#   16 
# 1595
total_hdr$BitsAllocated <- NULL



########
# Work
table(total_hdr$SliceLocation, useNA = 'ifany')
total_hdr$SliceLocation[c(51, 102, 149, 152, 266, 417, 687, 708, 891, 1308, 1338, 1465, 1506, 1536)]
# Checks NAs
(1:1595)[is.na(total_hdr$SliceLocation)]
total_hdr$SliceLocation <- gsub('\\+', '', total_hdr$SliceLocation)
total_hdr$SliceLocation <- as.numeric(total_hdr$SliceLocation)
(1:1595)[is.na(total_hdr$SliceLocation)]

total_hdr[c(51, 102, 149, 152, 266, 417, 687, 708, 891, 1308, 1338, 1465, 1506, 1536), ]


table(total_hdr$PixelSpacing, useNA = 'ifany')
# Number of ' ' (two values per variable)
library(stringr)
detect.spaces <- function(x) {str_detect(x, ' ')}
sum(detect.spaces(total_hdr$PixelSpacing))
pixel_spacing <- unlist(strsplit(total_hdr$PixelSpacing, ' '))

head(pixel_spacing)
# [1] "0.693359"   "0.693359"   "0.582031"   "0.582031"   "0.80859375" "0.80859375"

pixel_spacing_x <- pixel_spacing[rep(c(TRUE, FALSE), 1595)]
pixel_spacing_y <- pixel_spacing[rep(c(FALSE, TRUE), 1595)]

total_hdr$PixelSpacingX <- as.numeric(pixel_spacing_x)
total_hdr$PixelSpacingY <- as.numeric(pixel_spacing_y)
total_hdr$PixelSpacing  <- NULL


table(total_hdr$ImagePositionPatient, useNA = 'ifany')
sum(detect.spaces(total_hdr$ImagePositionPatient))
# [1] 1595
image_position_patient <- unlist(strsplit(total_hdr$ImagePositionPatient, ' '))

head(total_hdr$ImagePositionPatient)

image_position_patient_x <- image_position_patient[rep(c(TRUE, FALSE, FALSE), 1595)]
image_position_patient_y <- image_position_patient[rep(c(FALSE, TRUE, FALSE), 1595)]
image_position_patient_z <- image_position_patient[rep(c(FALSE, FALSE, TRUE), 1595)]

total_hdr$ImagePositionPatientX <- as.numeric(image_position_patient_x)
total_hdr$ImagePositionPatientY <- as.numeric(image_position_patient_y)
total_hdr$ImagePositionPatientZ <- as.numeric(image_position_patient_z)
total_hdr$ImagePositionPatient  <- NULL

summary(total_hdr)
library(Hmisc)
describe_total <- describe(total_hdr)
describe_total

# save(total_hdr, file = './data/total_hdr.RData')
load("./data/total_hdr.RData")

################################################################################
# One file "0015ceb851d7251b8f399e39779d1e7d"
dicom_0001 <- readDICOM(train_ids[1])

str(dicom_0001$hdr)

dicom_0001$hdr[1]
# hdr_0001 <- as.data.frame(unlist(dicom_0001$hdr))

dim(dicom_0001$img[[11]])
# [1] 512 512

class(dicom_0001)

str(dicom_0001$hdr)
str(dicom_0001$img)

hdr_0001 <- dicom_0001$hdr[[11]]

