from __future__ import print_function
__author__ = 'ajulian'
import numpy as np
import SimpleITK as sitk
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
import scipy.ndimage

import os
import time
import sys
sys.path.append("../")
# Import our competition variables
from competition_config import *

def load_itk_image(filename, filterBound=False, transform2realHU=False, resizeCT=False, correct_inv=False):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    if filterBound:
        numpyImage[numpyImage <= -2000] = -1000

    if transform2realHU:
        numpyImage += 1000

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    if resizeCT:
        numpyImage, numpySpacing = resize(numpyImage, old_spacing=numpySpacing, showTime=True)

    return numpyImage, numpyOrigin, numpySpacing

def resize(image, old_spacing, new_spacing=[1, 1, 1], showTime=False):
    t0 = time.time()
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    #print (image.shape, old_spacing)
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor) # 57 sec
    # image = scipy.ndimage.zoom(image, real_resize_factor) # 57 sec
    # image.resize(np.array(new_shape, dtype=np.int16))
    #print (image.shape)

    t1 = time.time()
    if showTime:
        print ("Resize takes", (t1-t0))

    return image, new_spacing

def tmpLUNAfiles(folder):
    LUNAfiles = [file for file in os.listdir(folder) if file[-4:] == ".mhd"]
    return LUNAfiles

def save_LUNA_origins(subfolder):
    ORIGIN_OF_IMAGES_FILE = D0['ORIGIN_OF_IMAGES_FILE']
    if not os.path.isfile(ORIGIN_OF_IMAGES_FILE):
        print("Origin of Images csv not found. Creating", ORIGIN_OF_IMAGES_FILE)
        origin_of_images = pd.DataFrame(columns=('seriesuid', 'origin_z', 'origin_y', 'origin_x'))
    else:
        print("Opening Origin of Images csv", ORIGIN_OF_IMAGES_FILE)
        origin_of_images = pd.read_csv(ORIGIN_OF_IMAGES_FILE)

    print("Opened Origin of Images csv")

    if not os.path.exists(D0["INPUT_DIRECTORY"]):
        print("LUNA images ROOT directory not found", D0["INPUT_DIRECTORY"])
        exit(0)

    input_subfolder = D0["INPUT_DIRECTORY"] + subfolder
    if not os.path.exists(input_subfolder):
        print("LUNA images subfolder not found", input_subfolder)
        exit(0)

    LUNAfiles = tmpLUNAfiles(input_subfolder)
    for img_file in LUNAfiles:
        if img_file[:-4] not in origin_of_images['seriesuid'].values:
            print("Reading image and origin", img_file)
            itkimage = sitk.ReadImage(input_subfolder + img_file)
            print("Already read image", input_subfolder + img_file)
            #numpyOrigin = np.array([new_row, new_row+1, new_row+2])
            numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
            new_row = origin_of_images.shape[0]+1
            # my understaing is numpy origin is z-y-x since it reversed itkimage.GetOrigin()
            origin_of_images.loc[new_row] = [img_file[:-4], numpyOrigin[0], numpyOrigin[1], numpyOrigin[2]]

    # index=True by default, which inserts a semicolon (,) at the beginning of the columns names
    origin_of_images.to_csv(ORIGIN_OF_IMAGES_FILE, index=False)
    print("Updated Origin of Images")

def resize_save_LUNA_imgs(subfolder):

    input_subfolder = D0["INPUT_DIRECTORY"] + subfolder
    output_subfolder = D0["OUTPUT_DIRECTORY"] + subfolder

    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    resized_images = 0
    LUNAfiles = tmpLUNAfiles(input_subfolder)
    for img_file in LUNAfiles:
        LUNApickle = img_file[:-4]+ ".pickle"
        if not os.path.isfile(output_subfolder + LUNApickle):
            resized_images +=1
            print("Resizing", resized_images, img_file)
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(input_subfolder + img_file,
                         resizeCT=True, filterBound=True, transform2realHU=True, correct_inv=True)
            with open(output_subfolder + LUNApickle , 'wb') as handle:
                        pickle.dump(numpyImage, handle, protocol=PICKLE_PROTOCOL)

# if D0 inside main, not visible when import 0_resize_LUNA from another file
D0 = resize_LUNA_dashboard # imported from competition_config.py

if __name__ == "__main__":

    if not os.path.exists(D0['LUNA_ROOT']):
        print("LUNA16 root not found", D0['LUNA_ROOT'])
        exit(0)

    if not os.path.exists(D0['OUTPUT_DIRECTORY']):
        os.makedirs(D0['OUTPUT_DIRECTORY'])

    START_TIME = time.time()

    for subfolder in D0['SUBFOLDERS']: # all Origin of images inserts should be made ASAP
        save_LUNA_origins(subfolder)
        #resize_save_LUNA_imgs(subfolder)
 
    subfolder = "subset0/"
    #save_LUNA_origins(subfolder)
    resize_save_LUNA_imgs(subfolder)
    
    print("Total elapsed time: " + \
          str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
