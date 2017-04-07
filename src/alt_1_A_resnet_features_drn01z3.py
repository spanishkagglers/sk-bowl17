# This script has been released under the Apache 2.0 open source license.
# Based on: mxnet + xgboost baseline [LB: 0.57] by n01z3
# https://www.kaggle.com/drn01z3/data-science-bowl-2017/mxnet-xgboost-baseline-lb-0-57/run/736291


'''
Download from http://data.dmlc.ml/mxnet/models/imagenet-11k-place365-ch/	
resnet-50-0000.params
resnet-50-symbol.json

and move to ../resnet-50
'''

import sys
sys.path.append("../")
from competition_config import *
d=alternative_model_1a


import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import os
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb

import pickle

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x


if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])

with open('alt_1_xgboost_resnet_features.pickle', 'rb') as handle:
    z=pickle.load(handle)


if d['USE_GPU']:
    DEVICE_TYPE='gpu'
else:
    DEVICE_TYPE='cpu'



def get_extractor():
    model = mx.model.FeedForward.load(d['PRETRAINED_RESNET_50'], 0, ctx=mx.Context(DEVICE_TYPE), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.Context(DEVICE_TYPE), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_data_id(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0
    # f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

        # if cnt < 20:
        #     plots[cnt // 5, cnt % 5].axis('off')
        #     plots[cnt // 5, cnt % 5].imshow(np.swapaxes(tmp, 0, 2))
        # cnt += 1

    # plt.show()
    batch = np.array(batch)
    return batch


def calc_features():
    net = get_extractor()
    for folder in tqdm(glob.glob(d['BOWL_CT_SCANS']+'*')):
        ct_scan_id=folder.split("/")[-1]
        output_filename=d['OUTPUT_DIRECTORY']+ct_scan_id+".pickle"
        if os.path.isfile(output_filename):
            print(str(folder) + ' already processed. Skipped')
            continue
        try:
            
#            print(ct_scan_id)
            batch = get_data_id(folder)
            feats = net.predict(batch)
#            print(feats.shape)
            
            with open(output_filename, 'wb') as handle:
                pickle.dump(feats, handle, protocol=PICKLE_PROTOCOL)
            if AWS:
                upload_to_s3(output_filename)
        except KeyboardInterrupt:
            print ("KeyboardInterrupt.  Deleting " + outfile)
            os.remove(outfile)
            print ("deleted")


if __name__ == '__main__':
    calc_features()
