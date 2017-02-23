# Based on:
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook

import numpy as np # linear algebra
import os
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label,regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import dicom
import pickle

import sys
sys.path.append("../")
# Import our competition variables
from competition_config import *
d=arnavs_lugns_roi_dashboard

import time
start_time = time.time()

INPUT_DIRECTORY = d['INPUT_DIRECTORY']
OUTPUT_DIRECTORY = d['OUTPUT_DIRECTORY']

if not os.path.exists(OUTPUT_DIRECTORY):
	os.makedirs(OUTPUT_DIRECTORY)

# Input data files are available in the INPUT_DIRECTORY.
# Any results you write has to be saved in the OUTPUT_DIRECTORY.


def read_ct_scan(folder_name):
	# Read the slices from the dicom file
	slicesL = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]
	
	# Sort the dicom slices in their respective order
	slicesL.sort(key=lambda x: int(x.InstanceNumber))
	
	slice_thickness = slicesL[0].ImagePositionPatient[2] - \
	slicesL[1].ImagePositionPatient[2] # Will be < 0 on inverted images
		
	#slice_thickness1 = slicesL[0].SliceLocation - slicesL[1].SliceLocation
	#print(folder_name, slice_thickness, slice_thickness1)
		
	# Get the pixel values for all the slices
	slicesA = np.stack([s.pixel_array for s in slicesL])
	slicesA = slicesA.astype(np.int16)
	
	# Convert to Hounsfield units (HU)
	intercept = slicesL[0].RescaleIntercept
	slope = slicesL[0].RescaleSlope
	
	if slope != 1:
		slicesA = slope * slicesA.astype(np.float64)
		slicesA = slicesA.astype(np.int16)

	slicesA += np.int16(intercept)
	
	# Saw -2048 instead of -2000, original has == -2000, changed for <= -2000
	# Instead of 0 (water), we use -1000 (air)
	slicesA[slicesA <= -2000] = -1000
	slicesA[slicesA <= -2000] = -1000

	slicesA += 1000 # So it will show
	
	# For inverted lungs:
	if slice_thickness < 0:
		slicesA = slicesA[::-1]

	slicesA = np.array(slicesA, dtype=np.int16)

	return slicesA


# Now we will plot a few more images of the slices using the plot_ct_scan function.

def plot_ct_scan(scan):
	f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
	for i in range(0, scan.shape[0], 5):
		plots[int(i / 20), int((i % 20) / 5)].axis('off')
		plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 


# This funtion segments the lungs from the given 2D slice.

def get_segmented_lungs(im, plot=False):
	
	if plot == True:
		f, plots = plt.subplots(8, 1, figsize=(5, 40))

	# Step 1: Convert into a binary image.
	# A threshold of 604(-400 HU) is used at all places because it was
	# found in experiments that it works just fine
	binary = im < 604
	if plot == True:
		print('Step 1: Convert into a binary image.')
		plots[0].axis('off')
		plots[0].imshow(binary, cmap=plt.cm.bone) 

	# Step 2: Remove the blobs connected to the border of the image.
	cleared = clear_border(binary)
	if plot == True:
		print('Step 2: Remove the blobs connected to the border of the image.')
		plots[1].axis('off')
		plots[1].imshow(cleared, cmap=plt.cm.bone) 
	
	# Step 3: Label the image.
	label_image = label(cleared)
	if plot == True:
		print('Step 3: Label the image.')
		plots[2].axis('off')
		plots[2].imshow(label_image, cmap=plt.cm.bone) 

	# Step 4: Keep the labels with 2 largest areas.
	areas = [r.area for r in regionprops(label_image)]
	areas.sort()
	if len(areas) > 2:
		for region in regionprops(label_image):
			if region.area < areas[-2]:
				for coordinates in region.coords:
					label_image[coordinates[0], coordinates[1]] = 0
	binary = label_image > 0
	if plot == True:
		print('Step 4: Keep the labels with 2 largest areas.')
		plots[3].axis('off')
		plots[3].imshow(binary, cmap=plt.cm.bone) 

	# Step 5: Erosion operation with a disk of radius 2. This operation is 
	# seperate the lung nodules attached to the blood vessels
	# using EROSION_BALL_RADIUS from competition_config.py
	selem = disk(arnavs_lugns_roi_dashboard['EROSION_BALL_RADIUS'])
	binary = binary_erosion(binary, selem)
	if plot == True:
		print('Step 5: Erosion operation')
		plots[4].axis('off')
		plots[4].imshow(binary, cmap=plt.cm.bone) 
	
	# Step 6: Closure operation with a disk of radius 10. This operation is 
	# to keep nodules attached to the lung wall
	# using CLOSING_BALL_RADIUS from competition_config.py
	selem = disk(arnavs_lugns_roi_dashboard['CLOSING_BALL_RADIUS'])
	binary = binary_closing(binary, selem)
	if plot == True:
		print('Step 6: Closure operation')
		plots[5].axis('off')
		plots[5].imshow(binary, cmap=plt.cm.bone) 

	# Step 7: Fill in the small holes inside the binary mask of lungs.
	edges = roberts(binary)
	binary = ndi.binary_fill_holes(edges)
	if plot == True:
		print('Step 7: Fill in the small holes inside the binary mask of lungs.')
		plots[6].axis('off')
		plots[6].imshow(binary, cmap=plt.cm.bone) 

	# Step 8: Superimpose the binary mask on the input image.
	get_high_vals = binary == 0
	im[get_high_vals] = 0
	if plot == True:
		print('Step 8: Superimpose the binary mask on the input image.')
		plots[7].axis('off')
		plots[7].imshow(im, cmap=plt.cm.bone) 
		
	return im


def segment_lung_from_ct_scan(ct_scan):
	return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


def segment_all_ct_scans(path): # Iterate through all folders
	all_folders = os.listdir(path)
	print('Found ' + str(len(all_folders)) + ' scans. Segmenting...')
	i = 0
	for folder in all_folders:
		i += 1
		i_start_time = time.time()
		# If pickle already exist, skip to the next folder
		if os.path.isfile(OUTPUT_DIRECTORY + folder + '.pickle'):
			print(folder + ' already segmented. Skipping...')
			continue
	
		# Segment
		ct_scan = read_ct_scan(path + folder + '/') 
		segmented_ct_scan = segment_lung_from_ct_scan(ct_scan)
		
		# Save image as .png with input image and binary mask superimposed
		plot_ct_scan(segmented_ct_scan)
		plt.savefig(OUTPUT_DIRECTORY + folder + '.png', format='png')
		
		# Save object as a .pickle
		with open(OUTPUT_DIRECTORY + folder + ".pickle", 'wb') as handle:
			pickle.dump(segmented_ct_scan, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		# Print and time to finish
		time_finish = round(time.time() - i_start_time) * (len(all_folders) - i)
		print(folder + ' segmentation created. About ' + \
			str(time.strftime('%H:%M:%S', time.gmtime(time_finish))) + \
			' left.')

segment_all_ct_scans(INPUT_DIRECTORY)

print("Total elapsed time: {} seconds".format(round((time.time() - start_time), 2)))