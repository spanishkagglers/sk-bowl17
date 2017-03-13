# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:02:42 2017

@author: javier
"""

import numpy as np
import os
from glob import glob
import csv

import pickle

import time
start_time = time.time()

import sys
sys.path.append("../")
from competition_config import *
d=features_extraction_nodules_3d

from scipy import stats

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import time
start_time = time.time()


if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])
    

def read_roi(ct_scan_id, d):
    with open(d['INPUT_DIRECTORY_1'] + ct_scan_id + '.pickle', 'rb') as handle:
        segmented_ct_scan = pickle.load(handle)
    return segmented_ct_scan

def read_segmented_nodules_info(ct_scan_id, d):
    with open(d['INPUT_DIRECTORY_2'] + 'info_' + ct_scan_id + '.pickle', 'rb') as handle:
        nodules_info = pickle.load(handle)
    return nodules_info  

def read_segmented_nodules_coords(ct_scan_id, d):
    with open(d['INPUT_DIRECTORY_2'] + 'points_' + ct_scan_id + '.pickle', 'rb') as handle:
        nodules_coords = pickle.load(handle)
    return nodules_coords        
    
def unit_vector(vector):
    """ Returns the unit vector of the vector  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return [z, y, x]

def sph2cartPoints(az_arr,el_arr,r):
    cart_coords = []
    for i in range(len(az_arr)):
        cart_coords.append(sph2cart(az_arr[i], el_arr[i], r))
    return cart_coords

def sphGrid2cartGrid(n,r,zc,yc,xc):
    # n = num points,  r = radius
    theta = np.linspace(0+np.pi/(n*2), np.pi-np.pi/(n*2), n, endpoint=True)
    phi = np.linspace(0 + 2*np.pi/(n*2), 2*np.pi - 2*np.pi/(n*2), n, endpoint=True)
    theta, phi = np.meshgrid(theta, phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x = x + xc
    y = y + yc
    z = z + zc
    return z, y, x


def WriteDictToCSV(csv_file,csv_columns,dict_data):

    with open(csv_file, 'w') as csvfile:
        #writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
    return     

def num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,nodule_mass_radius,f,n):
    # f factor division del radio
    # n^2 el número de puntos del grid de la esfera

    z2i,y2i,x2i = np.round(sphGrid2cartGrid(n,nodule_mass_radius/f,nodule_geo_center[0],nodule_geo_center[1],nodule_geo_center[2]),0)
    sphere_grid_coords=[]
    for i in range(n):
        for j in range(n):
            sphere_grid_coords.append([z2i[i,j],y2i[i,j],x2i[i,j]])           
    sphere_grid_coords = np.array(sphere_grid_coords)
    
    cont=0
    for ni in range(nod_coords.shape[0]):
        if(any(np.equal(sphere_grid_coords,nod_coords[ni]).all(1))):
            cont=cont+1
    return cont



file_list=glob(d['INPUT_DIRECTORY_1']+"*.pickle")

for input_filename in tqdm(file_list):
    ct_scan_id = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename=d['OUTPUT_DIRECTORY'] + ct_scan_id + ".pickle"
    if os.path.isfile(output_filename):
        pass
    else:
        roi = read_roi(ct_scan_id, d)
        nodules_info = read_segmented_nodules_info(ct_scan_id, d)
        nodules_coords = read_segmented_nodules_coords(ct_scan_id, d)
        

        features = []
        for nod in nodules_info:
            zmin = nod['min_z']; zmax = nod['max_z']
            ymin = nod['min_y']; ymax = nod['max_y']
            xmin = nod['min_x']; xmax = nod['max_x']
            
            nod_coords_dict = nodules_coords[nod['id_nodule']]
            nodule_values_array = roi[nod_coords_dict['z'],nod_coords_dict['y'],nod_coords_dict['x']]-1000
            nodule_volume = len(nodule_values_array)
            
            nodule_mass_radius = nod['radius']
            nodule_mass_center = np.array([nod['center'][2],nod['center'][1],nod['center'][0]]),
            nodule_geo_center = np.array([zmin+(zmax-zmin)/2,ymin+(ymax-ymin)/2,xmin+(xmax-xmin)/2])
            roi_geo_center = (np.array(roi.shape)-1)/2
            
            roiGC_to_noduleGC_vector = nodule_geo_center-roi_geo_center
            
            nodule_cuboid_2 = roi[max([0,int(nodule_geo_center[0]-2)]):max([0,int(nodule_geo_center[0]+3)]),
                                    max([0,int(nodule_geo_center[1]-2)]):max([0,int(nodule_geo_center[1]+3)]),
                                    max([0,int(nodule_geo_center[2]-2)]):max([0,int(nodule_geo_center[2]+3)])]
            
            nodule_cuboid_2 = nodule_cuboid_2[nodule_cuboid_2>0]
            
            nodule_cuboid_4 = roi[max([0,int(nodule_geo_center[0]-4)]):max([0,int(nodule_geo_center[0]+5)]),
                                    max([0,int(nodule_geo_center[1]-4)]):max([0,int(nodule_geo_center[1]+5)]),
                                    max([0,int(nodule_geo_center[2]-4)]):max([0,int(nodule_geo_center[2]+5)])]
          
            nodule_cuboid_4 = nodule_cuboid_4[nodule_cuboid_4>0]
            
            nodule_cuboid_8 = roi[max([0,int(nodule_geo_center[0]-8)]):max([0,int(nodule_geo_center[0]+9)]),
                                    max([0,int(nodule_geo_center[1]-8)]):max([0,int(nodule_geo_center[1]+9)]),
                                    max([0,int(nodule_geo_center[2]-8)]):max([0,int(nodule_geo_center[2]+9)])]
            
            nodule_cuboid_8 = nodule_cuboid_8[nodule_cuboid_8>0]
            
            nod_coords = np.concatenate(([nod_coords_dict['z']],[nod_coords_dict['y']],[nod_coords_dict['x']]),axis=0).T

            #np.percentile(nodule_values_array,10)
            
            '''
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import axes3d
            zi,yi,xi = np.round(sphGrid2cartGrid(10,100,500,0,0),0)
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
            ax.plot_wireframe(xi, yi, zi, color='k', rstride=1, cstride=1)
            ax.scatter(xi, yi, zi, s=10, c='r', zorder=10)
            '''
            
            cuboid = roi[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
            cuboid_shape = cuboid.shape
            cuboid_shape_ordered = np.sort(cuboid_shape)
            cuboid_volume = (zmax-zmin+1)*(ymax-ymin+1)*(xmax-xmin+1)
            cuboid_area = 2*((zmax-zmin+1)*(ymax-ymin+1)+(zmax-zmin+1)*(xmax-xmin+1)+(ymax-ymin+1)*(xmax-xmin+1))
            
            cuboid_nodules_volume = np.sum(cuboid>0)
            cuboid_nodules_array = cuboid[cuboid>0]
            
            nodule_volume_vs_cuboid_nodules_volume = nodule_volume/cuboid_nodules_volume
            
            sphere_volume = 4 * np.pi * nodule_mass_radius**3 / 3
            
            roi_values_array = roi[roi>0]
            
            
            features.append({
            'ct_scan_id':nod['ct_scan_id'],
            'id':nod['id_nodule'],
            'nod_volume':nodule_volume,
            'nod_sphere_volume':sphere_volume,
            'nod_cuboid_volume': cuboid_volume,
            'nod_cuboid_filled_ratio':nodule_volume/cuboid_volume,
            'nod_sphere_filled_ratio':nodule_volume/sphere_volume,
            'nod_vol_vs_cuboid_nods_vol': nodule_volume_vs_cuboid_nodules_volume,
            'nod_sphere_radius':nodule_mass_radius,
            'nod_hu_mean':np.mean(nodule_values_array),
            'nod_hu_sd':np.std(nodule_values_array),
            'nod_hu_mode': np.asscalar(stats.mode(nodule_values_array)[0]),
            'nod_hu_mean_vs_cuboid_hu_mean':np.mean(nodule_values_array)/np.mean(cuboid_nodules_array),
            'nod_hu_mean_vs_roi_hu_mean':np.mean(nodule_values_array)/np.mean(roi_values_array),
            'nod_lung_ratio': len(nodule_values_array[(nodule_values_array>=-949) & (nodule_values_array<=-120)])/nodule_volume,
            'nod_lung_ratio_90':len(nodule_values_array[(nodule_values_array>=-949) & (nodule_values_array<=-900)])/nodule_volume,
            'nod_lung_ratio_85':len(nodule_values_array[(nodule_values_array>=-899) & (nodule_values_array<=-850)])/nodule_volume,
            'nod_lung_ratio_80':len(nodule_values_array[(nodule_values_array>=-849) & (nodule_values_array<=-800)])/nodule_volume,
            'nod_lung_ratio_75':len(nodule_values_array[(nodule_values_array>=-799) & (nodule_values_array<=-750)])/nodule_volume,
            'nod_lung_ratio_70':len(nodule_values_array[(nodule_values_array>=-749) & (nodule_values_array<=-700)])/nodule_volume,
            'nod_lung_ratio_65':len(nodule_values_array[(nodule_values_array>=-699) & (nodule_values_array<=-650)])/nodule_volume,
            'nod_lung_ratio_60':len(nodule_values_array[(nodule_values_array>=-649) & (nodule_values_array<=-600)])/nodule_volume,
            'nod_lung_ratio_55':len(nodule_values_array[(nodule_values_array>=-599) & (nodule_values_array<=-550)])/nodule_volume,
            'nod_lung_ratio_50':len(nodule_values_array[(nodule_values_array>=-549) & (nodule_values_array<=-500)])/nodule_volume,
            'nod_lung_ratio_45':len(nodule_values_array[(nodule_values_array>=-499) & (nodule_values_array<=-450)])/nodule_volume,
            'nod_lung_ratio_40':len(nodule_values_array[(nodule_values_array>=-449) & (nodule_values_array<=-400)])/nodule_volume,
            'nod_lung_ratio_35':len(nodule_values_array[(nodule_values_array>=-399) & (nodule_values_array<=-350)])/nodule_volume,
            'nod_lung_ratio_30':len(nodule_values_array[(nodule_values_array>=-349) & (nodule_values_array<=-300)])/nodule_volume,
            'nod_lung_ratio_25':len(nodule_values_array[(nodule_values_array>=-299) & (nodule_values_array<=-250)])/nodule_volume,
            'nod_lung_ratio_20':len(nodule_values_array[(nodule_values_array>=-249) & (nodule_values_array<=-200)])/nodule_volume,
            'nod_lung_ratio_15':len(nodule_values_array[(nodule_values_array>=-199) & (nodule_values_array<=-150)])/nodule_volume,
            'nod_lung_ratio_10':len(nodule_values_array[(nodule_values_array>=-149) & (nodule_values_array<=-100)])/nodule_volume,
            'nod_fat_ratio':len(nodule_values_array[(nodule_values_array>=-100) & (nodule_values_array<=-50)])/nodule_volume,
            'nod_water_ratio':len(nodule_values_array[nodule_values_array==0])/nodule_volume,
            'nod_csf_ratio':len(nodule_values_array[nodule_values_array==15])/nodule_volume,
            'nod_kidney_ratio':len(nodule_values_array[nodule_values_array==30])/nodule_volume,
            'nod_blood_ratio':len(nodule_values_array[(nodule_values_array>=30) & (nodule_values_array<=45)])/nodule_volume,
            'nod_muscle_ratio':len(nodule_values_array[(nodule_values_array>=10) & (nodule_values_array<=40)])/nodule_volume,
            'nod_greymatter_ratio':len(nodule_values_array[(nodule_values_array>=37) & (nodule_values_array<=45)])/nodule_volume,
            'nod_whitematter_ratio':len(nodule_values_array[(nodule_values_array>=20) & (nodule_values_array<=30)])/nodule_volume,
            'nod_liver_ratio':len(nodule_values_array[(nodule_values_array>=40) & (nodule_values_array<=60)])/nodule_volume,
            'nod_soft_ratio':len(nodule_values_array[(nodule_values_array>=100) & (nodule_values_array<=300)])/nodule_volume,
            'nod_bone_ratio':len(nodule_values_array[nodule_values_array>=700])/nodule_volume,
            'nod_cuboid_area_volume_ratio': cuboid_area/cuboid_volume,
            'nod_cuboid_most_asimetric_face_edges_ratio': cuboid_shape_ordered[0]/cuboid_shape_ordered[2],
            'nod_cuboid_diagonal':np.sqrt((zmax-zmin+1)**2+(ymax-ymin+1)**2+(xmax-xmin+1)**2),
            'nod_center_to_roi_center_distance': np.sqrt(np.sum((roiGC_to_noduleGC_vector)**2)),
            'nod_center_to_xaxis_angle': angle_between([0,0,1],roiGC_to_noduleGC_vector),
            'nod_center_to_yaxis_angle': angle_between([0,1,0],roiGC_to_noduleGC_vector),
            'nod_center_to_zaxis_angle': angle_between([1,0,0],roiGC_to_noduleGC_vector),
            'nod_xmax_xmaxRoi_ratio': xmax/roi.shape[2],
            'nod_ymax_ymaxRoi_ratio': ymax/roi.shape[1],
            'nod_zmax_zmaxRoi_ratio': zmax/roi.shape[0],
            'nod_xcenter_xcenterRoi_ratio': nodule_geo_center[2]/roi.shape[2],
            'nod_ycenter_ycenterRoi_ratio': nodule_geo_center[1]/roi.shape[1],
            'nod_zcenter_zcenterRoi_ratio': nodule_geo_center[0]/roi.shape[0],
            'nod_sph_intersect_radiusNod1':num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,nodule_mass_radius,1.5,10),
            'nod_sph_intersect_radiusNod2':num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,nodule_mass_radius,2,10),
            'nod_sph_intersect_radiusNod4':num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,nodule_mass_radius,4,10),
            'nod_sph_intersect_radius2':num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,2,1,10),
            'nod_sph_intersect_radius4':num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,4,1,10),
            'nod_sph_intersect_radius8':num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,8,1,10),
            'nod_sph_intersect_radius16':num_points_nod_and_sphere_intersection(nod_coords,nodule_geo_center,16,1,10),
            'nod_cuboid_2_hu_mean':np.mean(nodule_cuboid_2),
            'nod_cuboid_4_hu_mean':np.mean(nodule_cuboid_4),
            'nod_cuboid_8_hu_mean':np.mean(nodule_cuboid_8),
            'nod_cuboid_2_hu_sd':np.std(nodule_cuboid_2),
            'nod_cuboid_4_hu_sd':np.std(nodule_cuboid_4),
            'nod_cuboid_8_hu_sd':np.std(nodule_cuboid_8),
            'nod_cuboid_4_vs_2_hu_mean':np.mean(nodule_cuboid_4)/np.mean(nodule_cuboid_2),
            'nod_cuboid_8_vs_2_hu_mean':np.mean(nodule_cuboid_8)/np.mean(nodule_cuboid_2),
            'raw_nod_center_z':nodule_geo_center[0],
            'raw_nod_center_y':nodule_geo_center[1],
            'raw_nod_center_x':nodule_geo_center[2],
            'raw_roi_center_z':roi_geo_center[0],
            'raw_roi_center_y':roi_geo_center[1],
            'raw_roi_center_x':roi_geo_center[2],
            'raw_nod_masscenter_z':nod['center'][2],
            'raw_nod_masscenter_y':nod['center'][1],
            'raw_nod_masscenter_x':nod['center'][0],
            'raw_cuboid_zmin':zmin,
            'raw_cuboid_zmax':zmax,
            'raw_cuboid_ymin':ymin,
            'raw_cuboid_ymax':ymax,
            'raw_cuboid_xmin':xmin,
            'raw_cuboid_xmax':xmax,
            'raw_roi_zmax':roi.shape[0]-1,
            'raw_roi_ymax':roi.shape[1]-1,
            'raw_roi_xmax':roi.shape[2]-1,
            })
    

    
        with open(output_filename, 'wb') as handle:
            output_filename=d['OUTPUT_DIRECTORY'] + ct_scan_id + ".pickle"
            pickle.dump(features, handle, protocol=PICKLE_PROTOCOL)
        
        csv_columns=sorted(list(features[0].keys()))
        csv_file = d['OUTPUT_DIRECTORY'] + ct_scan_id + ".csv"
        WriteDictToCSV(csv_file,csv_columns,features)
            
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))