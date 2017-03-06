## https://www.kaggle.com/yoshcakes/data-science-bowl-2017/full-preprocessing-in-r-with-3d-visualizations/code
## Version 13

library(oro.dicom)
library(oro.nifti) # for orthographic
library(plyr)
library(ggplot2)
library(imager) # for resize, dilate and erode
library(misc3d) # for contour3d
# library(raster)
library(SDMTools) # for function ConnCompLabel

# Here is a preprocessing tutorial in R, following the tutorial available online 
# as well as Guido Zuidhof's kernel for preprocessing in Python

# General
input_path <- file.path("./data/sample_images")
patients <- setdiff(basename(list.dirs(input_path)), c("sample_images"))

# Load the scans in given folder path and create slice thickness variable
load_scan <- function(path) {
  slices <- lapply(list.files(path, full.names = T), readDICOMFile)
  imagePositionPatientZ <- ldply(slices, function(slice) attr(create3D(slice), "ipp")[3])
  slices <- slices[order(imagePositionPatientZ)] # Reorder the slices from bottom to top
  
  slice_thickness <- try(abs(attr(create3D(slices[[1]]), "ipp")[3] - 
                               attr(create3D(slices[[2]]), "ipp")[3]
  ))
  if (class(slice_thickness) == "try-error") {
    slice_thickness <- abs(as.numeric(slices[[1]]$hdr$value[slices[[1]]$hdr$name == "SliceLocation"]) - as.numeric(slices[[2]]$hdr$value[slices[[2]]$hdr$name == "SliceLocation"]))
  }
  
  for (s in seq_along(slices)) {
    attr(slices[[s]], "SliceThickness") <- slice_thickness
  }
  slices
}

# Transform data in DICOM files in Hounsfield Units
get_image_hu <- function(scan) {
  image <- lapply(scan, function(x) x$img)
  # Set outside-of-scan pixels to 0
  # The intercept is usually -1024, so air is approximately 0
  # Convert to Hounsfield units (HU)
  for (slice_number in seq_along(scan)) {
    image[[slice_number]][image[[slice_number]] == -2000] <- 0
    intercept <- as.numeric(scan[[slice_number]]$hdr$value[scan[[slice_number]]$hdr$name == "RescaleIntercept"])
    slope <- as.numeric(scan[[slice_number]]$hdr$value[scan[[slice_number]]$hdr$name == "RescaleSlope"])
    
    if (slope != 1) {
      image[[slice_number]] = as.integer(slope * image[[slice_number]])
    }
    image[[slice_number]] <- image[[slice_number]] + intercept
  }
  ar <- array(dim = c(dim(image[[1]]), length(image)))
  for (i in seq_along(image)) {
    ar[,,i] <- image[[i]]
  }
  ar # image 3D array
}

# Our first patient
path <- file.path(input_path, patients[1])
patient_scan <- load_scan(path)
patient_images <- get_image_hu(patient_scan)

# Only take a sample for faster computation
df <- data.frame(pix=sample(unlist(patient_images), 10000))

ggplot(df, aes(x=pix)) + 
  geom_histogram(binwidth = 20) +
  xlab("Hounsfield Units (HU)") +
  ggtitle("Sample Distribution of Hounsfield Units in Patient #1")

# Show some slice in the middle
image(patient_images[,,80], col=grey(0:1024/1024), main="Slice")

# Orthographic view using oro.nifti package
orthographic(patient_images, crosshairs = F)

# Change the resoltuion of the images for homogenous inputs
change_resolution <- function(image, scan, new_spacing=c(1,1,1)) {
  # Determine current pixel spacing from scan metadata
  pixelspacing <- scan[[1]]$hdr$value[scan[[1]]$hdr$name == "PixelSpacing"]
  pixelspacing <- as.numeric(strsplit(pixelspacing, " ")[[1]])
  current_spacing <- c(pixelspacing, attr(scan[[1]], "SliceThickness"))
  image_shape <- dim(image)
  new_size <- as.integer(current_spacing/new_spacing*image_shape)
  
  new_scan <- resize(as.cimg(image), size_x=new_size[1],
                     size_y=new_size[2],
                     size_z=new_size[3],
                     interpolation_type = 3) # allows for linear interpolation
  
  as.array(new_scan)[,,,1]
}

patient_images_newres <- change_resolution(patient_images, patient_scan)
# Let's replot the same slice as before (with the new resolution) to compare
image(patient_images_newres[,,200], col=grey(0:10/10), main="Slice")
# Orthographic view using oro.nifti package
orthographic(patient_images_newres, crosshairs = F)

dim(patient_images) # Shape before resampling
dim(patient_images_newres) # Shape after resampling

# 3D plot of bones
contour3d(patient_images_newres, 400, engine = "standard")
contour3d(patient_images_newres[dim(patient_images_newres)[1]:1,,], 400, 
          engine = "standard")
# Use aperm(pix_resampled, c(2,1,3)) to get other views of the chest cavity
# by permuting the 3 dimensions of the arrays
# Or use engine = "rgl" to have an interactive view

contour3d(patient_images_newres[dim(patient_images_newres)[1]:1,,], 400, 
          engine = "rgl")

get_background_labels <- function(labels) {
  corners <- c(labels[1,1], 
               labels[1,nrow(labels)], 
               labels[ncol(labels),1], 
               labels[nrow(labels), ncol(labels)])
  corners
}

# Given a 3D image array, returns a binary array of same size with 1 for lungs
# Lungs have HU = -500, so choose a treshold above that, -320 is good
# brush_size allows use to broaden the mask in order to catch nodules
get_lung_mask <- function(image, threshold=-320, fill_lung = F, brush_size = 13) {
  binary_image <- image < threshold
  # binary_image has 1 for lungs but also 1 for air... let's change that
  for (slice_number in 1:dim(binary_image)[3]) {
    labeling <- ConnCompLabel(binary_image[,,slice_number])
    background_labels <- get_background_labels(labeling)
    binary_image[,,slice_number][labeling %in% background_labels] <- 0
    if (fill_lung) {
      binary_image[,,slice_number] <- dilate_square(as.cimg(binary_image[,,slice_number]), brush_size)
      binary_image[,,slice_number] <- erode_square(as.cimg(binary_image[,,slice_number]), brush_size)
    }
  }
  binary_image
}

# Let's see the lungs of our first patient
patient_lungs <- get_lung_mask(patient_images_newres)
patient_lungs_fill <- get_lung_mask(patient_images_newres, fill_lung = T)
contour3d(patient_lungs, 1, alpha = 0.1, engine = "standard")
contour3d(patient_lungs_fill, 1, alpha = 0.1, engine = "standard")

# Difference between the two masks
contour3d(patient_lungs_fill - patient_lungs, 1, alpha = 0.1, engine = "standard")

contour3d(patient_lungs_fill - patient_lungs, 1, alpha = 0.1, 
          engine = "rgl")

# Normalization of the data
MIN_BOUND <- -1000.0
MAX_BOUND <- 400.0

normalize <- function(image) {
  image <- (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
  image[image > 1] <- 1
  image[image < 0] <- 0
  image
}

# Zero center
PIXEL_MEAN <- 0.25

zero_center <- function(image) {
  image <- image - PIXEL_MEAN
  image
}