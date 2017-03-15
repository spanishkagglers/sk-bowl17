#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:06:37 2017

@author: spanishkagglers
"""

import sys
import os
import numpy as np # Important v1.12.0 or greater is needed. Maybe new environment?
# conda create -n augment python=3 numpy=1.12.0 matplotlib scipy spyder -y
import scipy.ndimage

import sys
sys.path.append("../")
# Import our competition variables, has to be before matplotlib
from competition_config import *
import matplotlib.pyplot as plt


def plt_chunk(chunk):
    '''Plot half the 3D chunk in 2D'''
    plt.imshow(chunk[int(len(chunk)/2)], cmap=plt.cm.gray)
    plt.show()

def save_chunk(chunk, name):
    '''Save chunk with name like [name, 'axis', rotation] = name_axis_rotation'''
    output = augment['OUTPUT_DIRECTORY'] + '_'.join(name)
    with open(output, 'wb') as handle:
        pickle.dump(chunk, handle, protocol=PICKLE_PROTOCOL)

def rotate90(chunk, name, axis):
    '''Will rotate chunk 90º clockwise 0, 1, 2 and 3 times for each wall.
    Rotations are like head to left shoulder'''
    rotations = ['0', '90', '180', '270']
    print('rotate90', name)
    for rot in range(4):

        rot_chunk = np.rot90(chunk, rot, axis)
        save_chunk(rot_chunk, name+[rotations[rot]])
        plt_chunk(rot_chunk)
        print('Rotation', rotations[rot])

def rotate_chunk(chunk, name):
    '''If you were lyin in a bed, you would rotate 4 times facing each
    wall of the room, floor and celling. Rotations are like head to left shoulder
    z = head to feet, x = left to right, y = back to chest'''

    # Patient is lying looking at the celling 'y' (original position)
    # Rotate bed clockwise
    rotate90(chunk, name+['y'], (0,2))
    # Turn upside down looking at the floor '-y' and rotate bed clockwise
    rotate90(np.rot90(chunk, 2, (2,1)), name+['-y'], (2,0))
    # Patient get out of bed on the right side of bed, standing up
    chunk = np.rot90(chunk, 1, (1,0)) # rises
    chunk = np.rot90(chunk, 1, (0,2)) # now right side of bed
    # Patient does a cart-wheel!
    rotate90(chunk, name+['x'], (1,0))
    # Turns around and another cart-wheel
    rotate90(np.rot90(chunk, 2, (2,0)), name+['-x'], (0,1))
    # Patient, standing up on the right of the bed, now faces the end of bed
    # The bed can be touched by his/her left hand
    chunk = np.rot90(chunk, 1, (2,0))
    # Cart-wheeh! (removing the bed so it does not interfere)
    rotate90(chunk, name+['z'], (2,1))
    # Turns around and another cart-wheel
    rotate90(np.rot90(chunk, 2, (2,0)), name+['-z'], (1,2))

def rotate_all_chunks(input_path, output_path):
    '''Rotate all input directory chunks and save them on output directory'''
    if not os.path.exists(augment['OUTPUT_DIRECTORY']):
        os.makedirs(augment['OUTPUT_DIRECTORY'])
    if not os.path.exists(augment['INPUT_DIRECTORY']):
        os.makedirs(augment['INPUT_DIRECTORY'])

    chunks = os.listdir(input_path)

    for c in chunks:
        with open(augment['INPUT_DIRECTORY'] + c, 'rb') as handle:
            # ANN LUNA pickles throw ascii compatibility issues, encoding has to be latin1
            chunk = pickle.load(handle, encoding='latin1')
        # Chunk name has to be a list, we use the text before .pickle
        rotate_chunk(chunk, [c.split('.')[0]])


if __name__ == "__main__":
    START_TIME = time.time()

    rotate_all_chunks(augment['INPUT_DIRECTORY'], augment['OUTPUT_DIRECTORY'])

    print('Total elapsed time: ' + \
          str(time.strftime('%H:%M:%S', time.gmtime((time.time() - START_TIME)))))
