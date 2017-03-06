__author__ = 'ajulian'

import numpy as np

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def get_chunk(img_set, ann_center, chunk_dims=(64, 64, 64), show_chunk_out=True, chunk_id=-1):
    # img_set: 3D ndarray

    # ann_center
    #   slice: closer slice to annotation center
    #   side: x (== side == col) in the slice
    #   front: y (== front == row) in the slice

    # chunk_dims: default (64, 64, 64)
    #   thickness: distance between chunk limit (upper and lower) slices in real world coordinates
    #   width: distance between chunk left and right sides
    #   depth: distance between chunk front and back sides

    # show_chunk_out: if True, print info when chunk exceeds CT

    # chunk_id: annotation index will be the chunk file id

    # return value: chunk (numpy 3d array); 0-pad if bigger than img_set limits

    CHUNK_OUT = False
    half_chunk_slice = chunk_dims[0]/2.0
    half_chunk_side = chunk_dims[1]/2.0
    half_chunk_front = chunk_dims[2]/2.0

    img_max_slice = img_set.shape[0]
    img_max_side = img_set.shape[1]
    img_max_front = img_set.shape[2]

    offset_min = np.array([0.0, 0.0, 0.0])
    offset_max = np.array([0.0, 0.0, 0.0])

    # check if chunk is inside slice limits
    min_slice = ann_center[0]-half_chunk_slice
    max_slice = ann_center[0]+half_chunk_slice
    if (min_slice < 0):
        offset_min[0] = -min_slice
        min_slice = 0.0
        CHUNK_OUT = True
    elif (max_slice > img_max_slice):
        offset_max[0] = max_slice - img_max_slice
        max_slice = img_max_slice
        CHUNK_OUT = True

    # check if chunk is inside side limits
    min_side = ann_center[1]-half_chunk_side
    max_side = ann_center[1]+half_chunk_side
    if (min_side < 0):
        offset_min[1] = -min_side
        min_side = 0.0
        CHUNK_OUT = True
    elif (max_side > img_max_side):
        offset_max[1] = max_side - img_max_side
        max_side = img_max_side
        CHUNK_OUT = True

    # check if chunk is inside front limits
    min_front = ann_center[2]-half_chunk_front
    max_front = ann_center[2]+half_chunk_front
    if (min_front < 0):
        offset_min[2] = -min_front
        min_front = 0.0
        CHUNK_OUT = True
    elif (max_front > img_max_front):
        offset_max[2] = max_front - img_max_front
        max_front = img_max_front
        CHUNK_OUT = True

    """
    print chunk_dims, max_slice-min_slice, max_side-min_side, max_front-min_front
    print min_slice, max_slice
    print min_side, max_side
    print min_front, max_front
    """

    min_slice = int(min_slice)
    max_slice = int(max_slice)
    min_side = int(min_side)
    max_side = int(max_side)
    min_front = int(min_front)
    max_front = int(max_front)

    if CHUNK_OUT:
        offset = np.array(np.round((offset_min+offset_max)/2), dtype=np.int16)
        if show_chunk_out:
            print ("PART OF CHUNK IS OUT!")
            print ("offset", offset_min, offset_max, offset)
            print ("chunk id", chunk_id)

        new_chunk = np.zeros(shape=chunk_dims, dtype=np.int16)
        new_chunk[offset[0]:offset[0]+(max_slice-min_slice),
                  offset[1]:offset[1]+(max_side-min_side),
                  offset[2]:offset[2]+(max_front-min_front)] = \
            img_set[min_slice:max_slice, min_side:max_side, min_front:max_front]
        return new_chunk
    else:
        return img_set[min_slice:max_slice, min_side:max_side, min_front:max_front].copy()
