# Based on:
# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/notebook


import os
from skimage.morphology import ball, binary_closing
from skimage.measure import label,regionprops
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import pickle
import boto3

import sys
sys.path.append("../")

import time
start_time = time.time()

s3_client = boto3.client('s3')


#def plot_ct_scan(scan):
#	f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
#	for i in range(0, scan.shape[0], 5):
#		plots[int(i / 20), int((i % 20) / 5)].axis('off')
#		plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
#
#
#def plot_3d(image, threshold=-300):
#	
#	# Position the scan upright, 
#	# so the head of the patient would be at the top facing the camera
#	p = image.transpose(2,1,0)
#	p = p[:,:,::-1]
#	
#	verts, faces = measure.marching_cubes(p, threshold)
#
#	fig = plt.figure(figsize=(10, 10))
#	ax = fig.add_subplot(111, projection='3d')
#
#	# Fancy indexing: `verts[faces]` to generate a collection of triangles
#	mesh = Poly3DCollection(verts[faces], alpha=0.1)
#	face_color = [0.5, 0.5, 1]
#	mesh.set_facecolor(face_color)
#	ax.add_collection3d(mesh)
#
#	ax.set_xlim(0, p.shape[0])
#	ax.set_ylim(0, p.shape[1])
#	ax.set_zlim(0, p.shape[2])
#
#	plt.show()
	

#def get_nodules(segmented_ct_scan):
#	
#	segmented_ct_scan[segmented_ct_scan < 604] = 0
#	#plot_ct_scan(segmented_ct_scan)
#	#plot_3d(segmented_ct_scan, 604)
#
#	# After filtering, there are still lot of noise because of blood vessels.
#	# Thus we further remove the two largest connected component.
#
#	selem = ball(d['BALL_RADIUS'])
#	binary = binary_closing(segmented_ct_scan, selem)
#
#	label_scan = label(binary)
#
#	areas = [r.area for r in regionprops(label_scan)]
#	areas.sort()
#
#	for r in regionprops(label_scan):
#		max_x, max_y, max_z = 0, 0, 0
#		min_x, min_y, min_z = 1000, 1000, 1000
#		
#		for c in r.coords:
#			max_z = max(c[0], max_z)
#			max_y = max(c[1], max_y)
#			max_x = max(c[2], max_x)
#			
#			min_z = min(c[0], min_z)
#			min_y = min(c[1], min_y)
#			min_x = min(c[2], min_x)
#		if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
#			for c in r.coords:
#				segmented_ct_scan[c[0], c[1], c[2]] = 0
#		else:
#			index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / \
#				 (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
#	
#	'''Comment from ArnavJain on Kaggle about index: 
#	Index is the shape index of the 3D blob. As mentioned in the TODOs, 
#	I am working on reducing the generated candidates using shape index 
#	and other properties. So, index is one such property.
#	I will post the whole part as soon as I complete it.'''
#			
#	return segmented_ct_scan


# TODO if 5_pickle already exist, skip


def lambda_handler(event, context):
	for record in event['Records']:
		bucket = record['s3']['bucket']['name']
		key = record['s3']['object']['key'] 
		download_path = '/tmp/input/{}'.format( key)
		upload_path = '/tmp/output/{}'.format(key)

		s3_client.download_file(bucket, key, download_path)
		# Load pickle with segmented_ct_scan
		with open(download_path, 'rb') as handle:
			segmented_ct_scan = pickle.load(handle)

		segmented_nodules_ct_scan = get_nodules(segmented_ct_scan)
		
		# Save object as a .pickle
		with open(upload_path, 'wb') as handle:
			pickle.dump(segmented_nodules_ct_scan, handle, protocol=2)	
		s3_client.upload_file(upload_path, bucket, 'output/5_nodules_roi' + key)