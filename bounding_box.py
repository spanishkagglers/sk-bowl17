import numpy as np

def boundingBox3D(ct):
    # ct: 3D numpy array
    # returns limits of bounding box
    rows = np.any(ct, axis=(0, 2))
    cols = np.any(ct, axis=(0, 1))
    slices = np.any(ct, axis=(1, 2))

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    smin, smax = np.where(slices)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, smin, smax

