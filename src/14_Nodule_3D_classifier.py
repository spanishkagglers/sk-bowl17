# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:08:19 2017

@author: spanish kagglers
"""

import os
#os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO"]="device=gpu,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic;"
import keras.backend as K
K.set_image_dim_ordering('th') 

import numpy as np # linear algebra
import os
import shutil
from glob import glob

import scipy.misc
import pickle
#import json
import yaml

from collections import Counter
from sklearn.cluster import DBSCAN  
#import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
start_time = time.time()

import sys
sys.path.append("../")
from competition_config import *
d=nodule_3D_classifier

from scipy.spatial.distance import *



from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, History, ModelCheckpoint 


from keras.models import Model
from keras.layers import Input,  BatchNormalization

#import theano
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import cv2
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import cross_validation
from sklearn import preprocessing

from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

import pandas as pd

try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x

import time
start_time = time.time()

from util.plot_3d import *

import seaborn as sns

import numpy as np
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
    
def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

def plot(x):
    fig, axes = plt.subplots(nrows=2)
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier]):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    axes[0].set_title('Percentile-based Outliers', **kwargs)
    axes[1].set_title('MAD-based Outliers', **kwargs)
    fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=14)





report={
    'early_stopping_best_epoch':[],
    'train_scores': [],
    'val_scores': [],

}

if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])

if 'TEMP_DIRECTORY' in d and not os.path.exists(d['TEMP_DIRECTORY']):
    os.makedirs(d['TEMP_DIRECTORY'])

if not os.path.exists(d['EXECUTION_OUTPUT_DIRECTORY']):
    os.makedirs(d['EXECUTION_OUTPUT_DIRECTORY'])


def copy_to_ramdisk(dashboard_directory_key, ramdisk_directory):
    ramdisk_directory=d['TEMP_DIRECTORY']+ramdisk_directory
    if not os.path.exists(ramdisk_directory):
        print("Copying chunks into ramdisk: ", d[dashboard_directory_key], " --> ", ramdisk_directory)
        shutil.copytree(d[dashboard_directory_key], ramdisk_directory)
    d[dashboard_directory_key]=ramdisk_directory    

if d['USE_RAMDISK']:
    print("INFO: Using ramdisk")
    copy_to_ramdisk('BOWL_INPUT_DIRECTORY', '8B-bowl_chunks/')
    copy_to_ramdisk('LUNA_INPUT_DIRECTORY', '9A-augmented_luna_chunks/')
#    copy_to_ramdisk('LUNA_NON_NODULES_INPUT_DIRECTORY', 'LUNA_NON_NODULES/')
#    copy_to_ramdisk('LUNA_OTHER_TISSUES_INPUT_DIRECTORY', 'LUNA_OTHER_TISSUES/')




img_rows=img_cols=img_depth=chunck_size=d['CHUNK_SIZE'] # ¿X, Y,?, Z





def get_model(channels, img_rows, img_cols, img_depth, summary=False, backend=None):
    """ Return the Keras model of the network
    
    """
    
    if backend is None:
        backend = K.image_dim_ordering()
    
    model = Sequential()
    if backend == 'tf':
        input_shape=(img_rows, img_cols, img_depth, channels) # ???(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(channels, img_rows, img_cols, img_depth) # ??? 3, 16, 112, 112) # c, l, h, w
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a'))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if summary:
        print(model.summary())
        
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def read_luna_annotations_csv(filename):
    annotations__pd = pd.read_csv(filename, sep=",")
    annotations__pd['nodule_id']=["ann"+str(x) for x in annotations__pd.index.values]
#    annotations__pd['nodule_id']=annotations__pd['nodule_id'].index.apply(lambda x: "ann"+str(x))
    return annotations__pd

luna_annotations__pd=read_luna_annotations_csv(d['INPUT_LUNA_NODULES_METADATA'])


luna_chunks_file_list=sorted(glob(d['LUNA_INPUT_DIRECTORY']+"*.pickle"))
luna_chunks_list=[os.path.basename(x)[:-len(".pickle")] for x in luna_chunks_file_list]
luna_chunks_list_without_augmentation=sorted(set([x.split(CHUNK_VARIATION_SEP)[0] for x in luna_chunks_list]))
luna_affected_nodules=luna_annotations__pd.loc[luna_annotations__pd['diameter_mm']>=2*d['RADIUS_THRESHOLD_MM']]['nodule_id'].values
luna_healthy_nodules=[x for x in luna_chunks_list_without_augmentation if x not in luna_affected_nodules]
#affected_file_list=[d['LUNA_INPUT_DIRECTORY']+x+".pickle" for x in luna_chunks_list_without_augmentation if x in affected_nodules]
#healthy_file_list=[d['LUNA_INPUT_DIRECTORY']+x +".pickle"for x in luna_chunks_list_without_augmentation if x not in affected_nodules]



'''
      _       _          _                 _ _             
     | |     | |        | |               | (_)            
   __| | __ _| |_ __ _  | | ___   __ _  __| |_ _ __   __ _ 
  / _` |/ _` | __/ _` | | |/ _ \ / _` |/ _` | | '_ \ / _` |
 | (_| | (_| | || (_| | | | (_) | (_| | (_| | | | | | (_| |
  \__,_|\__,_|\__\__,_| |_|\___/ \__,_|\__,_|_|_| |_|\__, |
                                                      __/ |
                                                     |___/ 

'''

print ("="*15 + "\n" + d['CLASSES_TXT'] + "="*15)

def get_nodules_and_augmentations_dict(directory, nodules_file_list,nodules_dict=None, dataset_ids=None, dataset_X=None,dataset_Y=None,y_value=None, augmentation_sep=CHUNK_VARIATION_SEP, augmentations=True, max_chunks=None):
    num_chuncks=0
    
    for nodule_id in tqdm(nodules_file_list):
        nodule_augmented_chunks_file_list=list(glob(directory+nodule_id+augmentation_sep+"*.pickle")) # ¿debería ser random?
        random.shuffle(nodule_augmented_chunks_file_list)
        for input_filename in nodule_augmented_chunks_file_list:
            with open(input_filename, 'rb') as handle:
                nodule_3d = pickle.load(handle, encoding='latin1')
                num_chuncks+=1
                if(chunck_size<nodule_3d.shape[0]):
                    i=round((nodule_3d.shape[0]-chunck_size)/2)
                    j=i+chunck_size
                    nodule_3d=nodule_3d[i:j,i:j,i:j]
                
                if nodules_dict is not None:
                    if nodule_id not in nodules_dict:
                        nodules_dict[nodule_id]=[nodule_3d]
                    else:
                        nodules_dict[nodule_id]+=[nodule_3d]
                if dataset_X is not None:
                    dataset_X.append(nodule_3d)
                if dataset_Y is not None:
                    dataset_Y.append(y_value)
                if dataset_ids is not None:
                    augmented_nodule_id=input_filename[len(directory):-len(".pickle")]
                    dataset_ids.append(augmented_nodule_id)
            if not augmentations:
                break
            if max_chunks is not None and num_chuncks >= max_chunks:
                break
        if max_chunks is not None and num_chuncks >= max_chunks:
            print("Reached MAX_CHUNKS_PER_CLASS of " , max_chunks , ".  Stopped.")
            break
    return num_chuncks

X_tr=[]
Y_tr=[]
#luna_affected_augmentations={}
#luna_healthy_augmentations={}


if 'CLASS_HIGHER_DIAMETER_NODULES' in d['USE_CLASSES']:
    luna_num_affected_chucks=get_nodules_and_augmentations_dict(d['LUNA_INPUT_DIRECTORY'], luna_affected_nodules,  dataset_X=X_tr, dataset_Y=Y_tr, y_value=d['CLASS_HIGHER_DIAMETER_NODULES'], max_chunks=d['MAX_CHUNKS_PER_CLASS'] )
    print("\nCLASS_HIGHER_DIAMETER_NODULES nodules (includes augmentation): " + str(luna_num_affected_chucks))

if 'CLASS_LOWER_DIAMETER_NODULES' in d['USE_CLASSES']:
    luna_num_healthy_chucks =get_nodules_and_augmentations_dict(d['LUNA_INPUT_DIRECTORY'], luna_healthy_nodules,  dataset_X=X_tr, dataset_Y=Y_tr, y_value=d['CLASS_LOWER_DIAMETER_NODULES'], augmentations=False, max_chunks=d['MAX_CHUNKS_PER_CLASS'] )
    print("CLASS_LOWER_DIAMETER_NODULES nodules: " + str(luna_num_healthy_chucks))


bowl_labels__pd=pd.read_csv(d['BOWL_LABELS'])
bowl_non_affected_lungs=bowl_labels__pd['id'].loc[bowl_labels__pd['cancer']==0].values
#all_patients=next(os.walk(d['BOWL_PATIENTS']))[1]
if 'CLASS_SEGMENTED_FROM_NON_AFFECTED_LUNGS' in d['USE_CLASSES']:
    bowl_num_healthy_chunks =get_nodules_and_augmentations_dict(d['BOWL_INPUT_DIRECTORY'], bowl_non_affected_lungs, augmentation_sep='',  dataset_X=X_tr, dataset_Y=Y_tr, y_value=d['CLASS_SEGMENTED_FROM_NON_AFFECTED_LUNGS'], augmentations=False, max_chunks=d['MAX_CHUNKS_PER_CLASS'] )
    print("CLASS_SEGMENTED_FROM_NON_AFFECTED_LUNGS: ", bowl_num_healthy_chunks)

if 'CLASS_NON_NODULES' in d['USE_CLASSES']:
    num_non_nodules =get_nodules_and_augmentations_dict(d['LUNA_NON_NODULES_INPUT_DIRECTORY'], [''], augmentation_sep='',  dataset_X=X_tr, dataset_Y=Y_tr, y_value=d['CLASS_NON_NODULES'], augmentations=True, max_chunks=d['MAX_CHUNKS_PER_CLASS'] )
    print("CLASS_NON_NODULES nodules: " + str(num_non_nodules))

if 'CLASS_OTHER_TISSUES' in d['USE_CLASSES']:
    num_other_tissues =get_nodules_and_augmentations_dict(d['LUNA_OTHER_TISSUES_INPUT_DIRECTORY'], [''], augmentation_sep='',  dataset_X=X_tr, dataset_Y=Y_tr, y_value=d['CLASS_OTHER_TISSUES'], augmentations=True, max_chunks=d['MAX_CHUNKS_PER_CLASS'] )
    print("CLASS_OTHER_TISSUES nodules: " + str(num_non_nodules))





#num_affected_without_augmentation = len(luna_affected_nodules)
#num_healthy_without_augmentation = len(luna_healthy_nodules)

#num_affected = len(luna_affected_nodules)
#num_healthy = len(luna_healthy_nodules)
#print("\nnum_affected: " + str(num_affected))
#print("num_healthy: " + str(num_healthy))





label=np.array(Y_tr)

X_tr_array = np.array(X_tr) 

num_samples = len(X_tr_array)
#print ("num_samples without augmentation: ", num_samples)


train_data = [X_tr_array,label]

(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)



'''                                                  _             
                                                  (_)            
  _ __  _ __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
 | '_ \| '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
 | |_) | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
 | .__/|_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
 | |            | |                                         __/ |
 |_|            |_|                                        |___/ 

'''


def get_normalized_dataset_for_cnn(dataset, substract=None, divide_by=None, img_rows=img_rows, img_cols=img_cols, img_depth=img_depth):
    num_samples=len(dataset)
    print("DEBUG: num observations ", num_samples)
    result_dataset = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))
    
    for i in range(num_samples):
        z=dataset[i,:,:,:]
        result_dataset[i][0][:][:][:]=z
    
    print(dataset.shape, ' samples') 
    
    dataset_mean = np.mean(result_dataset)
    dataset_max = np.max(np.abs(result_dataset))
    
#     = np.min(result_dataset)
#     = np.max(np.abs(result_dataset))-dataset_min
    
    if substract is None:
        substract=dataset_mean
        
    
    if divide_by is None:
        divide_by=dataset_max
        
    
    print("DEBUG - dataset_mean: ", dataset_mean)
    print("DEBUG - dataset_max: ", dataset_max)
    
    
    
    result_dataset = result_dataset.astype('float32')
    
    result_dataset -= substract
    
    result_dataset /=divide_by
    
    result_dataset = np.clip(result_dataset, -1.0, 1.0)
    
    return result_dataset, dataset_mean, dataset_max 

train_set, train_mean, train_max = get_normalized_dataset_for_cnn((X_tr_array/2).clip(-250,1200))
report['train_mean']=train_mean
report['train_max']=train_max


X_test=[]
ids_test=[]
num_test_set_chunks =get_nodules_and_augmentations_dict(d['BOWL_INPUT_DIRECTORY'], [''], augmentation_sep='',  dataset_X=X_test, dataset_ids=ids_test, max_chunks=d['MAX_CHUNKS_TO_PREDICT'])
X_test_array = np.array(X_test) 
del(X_test)
ids_test__pd=pd.DataFrame(ids_test, columns={'nodule_id'})
test_set, test_mean, test_max = get_normalized_dataset_for_cnn((X_test_array/2).clip(-250,1200), substract=train_mean, divide_by=train_max)
report['test_mean']=test_mean
report['test_max']=test_max




'''
                      _      _   _             _       _             
                     | |    | | | |           (_)     (_)            
  _ __ ___   ___   __| | ___| | | |_ _ __ __ _ _ _ __  _ _ __   __ _ 
 | '_ ` _ \ / _ \ / _` |/ _ \ | | __| '__/ _` | | '_ \| | '_ \ / _` |
 | | | | | | (_) | (_| |  __/ | | |_| | | (_| | | | | | | | | | (_| |
 |_| |_| |_|\___/ \__,_|\___|_|  \__|_|  \__,_|_|_| |_|_|_| |_|\__, |
                                                                __/ |
                                                               |___/ 
'''

patch_size = img_depth    # img_depth or number of frames used for each video

  


batch_size = d['BATCH_SIZE']
nb_classes = d['NUM_CLASSES']
nb_epoch =   d['EPOCHS'] 

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]




def cnn_model_old(input_shape, output_shape):
    inp = Input(shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    #Layer 1
    l1_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(inp)
    l1_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_conv1)
    l1_maxpool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(l1_conv2)
    
    # Flatten
    flat = Flatten()(l1_maxpool1)
    dense1 = Dense(512, init='glorot_uniform', activation='relu')(flat)
    dense2 = Dense(64, init='glorot_uniform', activation='relu', name="features_layer")(dense1)
    out = Dense(output_shape, activation='softmax', name="output_layer")(dense2)
    
    model = Model(input=inp, output=out)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_model(input_shape, output_shape):
    model = Sequential()
    
    cnn_shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    #Layer 1
    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same', init='glorot_uniform',  input_shape=cnn_shape))
    model.add(PReLU())
    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
    model.add(PReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    
    #Layer 2
    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
    model.add(PReLU())
    model.add(Convolution3D(32, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
    model.add(PReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    
    #Layer 3
    model.add(Convolution3D(64, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
    model.add(PReLU())
    model.add(Convolution3D(64, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
    model.add(PReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    
#    #Layer 3
#    model.add(Convolution3D(64, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
#    model.add(PReLU())
#    model.add(Convolution3D(64, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
#    model.add(PReLU())
#    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    
#    #Layer 3
#    model.add(Convolution3D(256, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
#    model.add(PReLU())
#    model.add(Convolution3D(256, 3, 3, 3, border_mode = 'same', init='glorot_uniform'))
#    model.add(PReLU())
#    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    
    # Flatten
#    model.add(Flatten())
#    model.add(Dense(512, init='glorot_uniform'))
#    model.add(PReLU())
    model.add(Flatten())
    model.add(Dense(512, init='glorot_uniform', name="features_layer_1"))
    model.add(PReLU())
    model.add(Dense(64, init='glorot_uniform', name="features_layer_2"))
    model.add(PReLU())
    model.add(Dense(output_shape, activation='softmax', name="output_layer"))
    
    #model = Model(input=inp, output=out)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model








#model = get_model(1,img_rows, img_cols, img_depth, backend=None)
 
# Split the data

#X_train_fold, X_val_fold, y_train_fold,y_val_fold =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)





num_samples=Y_train.shape[0]

oof_preds  = np.zeros(num_samples)
oof_val_index=None
#test_preds = np.zeros(xtest.shape[0])

history = History()

use_k_fold=d['USE_K_FOLD']

if(use_k_fold):
    num_folds=d['NUM_FOLDS'][1]
    stop_after_fold=d['NUM_FOLDS'][0]
    skf = StratifiedKFold(n_splits=num_folds, random_state=RANDOM_SEED, shuffle=True)
    validation_indexes=skf.split(np.zeros(num_samples) , Y_train[:,1])
else:
    num_folds=0
    stop_after_fold=0
    validation_indexes=zip("do_not_K_fold", "do_not_K_fold")
    



if 'RESUME_TRAINING' in d and d['RESUME_TRAINING'] is not None:
    resume_training=True
    previous_training_weights=sorted(glob(d['OUTPUT_DIRECTORY']+d['RESUME_TRAINING']+"/*.hdf5"))
else:
    resume_training=False

num_fold=1
for train_index, val_index in validation_indexes:
    print("Fold: {} / {}".format(num_fold,num_folds))
    print("========")
    
    model = cnn_model((1, img_rows, img_cols, img_depth), d['NUM_CLASSES'])
    
    if resume_training:
        model.load_weights(previous_training_weights[num_fold - 1])
    
    model_output_filename = str(d['DASHBOARD_ID']) + "_fold_" + str(num_fold) +  ".hdf5" #"_batch_" + str(num_bag) +
        
    
    #print("TRAIN:", train_index, "TEST:", val_index)
    if use_k_fold:
        X_train_fold, X_val_fold = train_set[train_index], train_set[val_index]
        y_train_fold,y_val_fold = Y_train[train_index], Y_train[val_index]
        validation_data = (X_val_fold,y_val_fold)
        
        
        if oof_val_index is None:
            oof_val_index = val_index
        else:
            oof_val_index=np.hstack([oof_val_index, val_index])
        
        # CALLBACKS
        filepath = d['TEMP_DIRECTORY'] +model_output_filename
        early_stopping = EarlyStopping(monitor='val_loss', patience=d['EARLY_STOPPING_ROUNDS'], verbose=1, mode='auto')
        
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        callbacks_list= [early_stopping, checkpoint] #, history]
    else:
        X_train_fold = train_set
        y_train_fold = Y_train
        validation_data=None
        
        early_stopping=None
        callbacks_list= []
        
        filepath = d['EXECUTION_OUTPUT_DIRECTORY'] +model_output_filename
        

    

    # Train the model
    print("Training...")
    hist = model.fit(X_train_fold,
            y_train_fold,
            validation_data=validation_data,
            batch_size=batch_size,
            nb_epoch = nb_epoch,
            shuffle=True
            ,callbacks= callbacks_list
    ) #show_accuracy=True,
    print("Done")
    

    
    if early_stopping in callbacks_list:
        print("INFO: loading best model according to early_stopping")
        model.load_weights(filepath) # loading best model from best epoch
        print('Fold: ' + str(num_fold) + " - Best Epoch: ",str(early_stopping.best))
        report['early_stopping_best_epoch']+=[early_stopping.best]
        
        if d['TEMP_DIRECTORY']!=d['EXECUTION_OUTPUT_DIRECTORY']:
            shutil.move(filepath, d['EXECUTION_OUTPUT_DIRECTORY'])
    else:
        print("DEBUG: early stopping disabled")
        print("INFO: saving model to disk")
        save_model(model, filepath)
    

    
    


    #hist = model.fit(train_set, Y_train, batch_size=batch_size,
    #         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
    #           shuffle=True)
    
    
     # Evaluate the model
    #score = model.evaluate(X_val_fold, y_val_fold, batch_size=batch_size) #, show_accuracy=True)
    if use_k_fold:
        preds=model.predict(X_val_fold, batch_size=batch_size)
        results_val=pd.DataFrame(preds[:,1].astype(np.float), columns={'prediction'})
        results_val['affected']=y_val_fold[:,1]
    #    results_val['affected']=pd.to_numeric(results_val['affected'], downcast='float')
    #    results_val['prediction']=pd.to_numeric(results_val['prediction'])
        #results_val.loc[results_val['affected']<0.0001]=0
        score = log_loss(results_val['affected'], results_val['prediction'])
        report['val_scores']+=[score]
        if len(results_val['prediction'].unique())<d['NUM_CLASSES']:
            print("Fold: {} - WARN: predicted just {} different classes of {}".format(num_fold, len(results_val['prediction'].unique()), d['NUM_CLASSES']))
        oof_preds[val_index] = preds[:,1]
        print('Validation Score - LogLoss fold {}: {}'.format(num_fold, score))
    
    
    train_preds=model.predict(X_train_fold, batch_size=batch_size)
    train_results=pd.DataFrame(train_preds[:,1].astype(np.float), columns={'prediction'})
    train_results['affected']=y_train_fold[:,1]
    
    train_score = log_loss(train_results['affected'], train_results['prediction'])    
    report['train_scores']+=[train_score]
    
    print('Train Score      - LogLoss fold {}: {}'.format(num_fold, train_score))
    
    
    print("Test set predictions: in progress")
    predict_start_time=time.time()
    test_preds=model.predict(test_set, batch_size=batch_size)
    test_preds__pd=pd.concat([ids_test__pd,
                            pd.DataFrame(test_preds.astype(np.float))], 
                            axis=1) 
    
    test_output_base_filename= d['EXECUTION_OUTPUT_DIRECTORY']+"test_predictions_"+ model_output_filename[:-len(".hdf5")] 
    with open(test_output_base_filename + ".pickle", 'wb') as handle:
        pickle.dump(test_preds__pd, handle, protocol=PICKLE_PROTOCOL)
    test_preds__pd.to_csv(test_output_base_filename + ".csv", sep=",", header=True, index=False)
    print("Test set predictions: done in ", round(time.time() - predict_start_time), " seconds")
    
    print("sencond last layer output: in progress")
    predict_start_time=time.time()
    intermediate_layer_model = Model(input=model.input, output=model.get_layer("features_layer_1").output)
    intermediate_output = intermediate_layer_model.predict(test_set, batch_size=batch_size)
    intermediate_output__pd=pd.concat([ids_test__pd,
                            pd.DataFrame(intermediate_output)], 
                            axis=1) 
    
    intermediate_output_base_filename= d['EXECUTION_OUTPUT_DIRECTORY']+"intermediate_output_1_"+ model_output_filename[:-len(".hdf5")] 
    with open(intermediate_output_base_filename + ".pickle", 'wb') as handle:
        pickle.dump(intermediate_output__pd, handle, protocol=PICKLE_PROTOCOL)
    intermediate_output__pd.to_csv(intermediate_output_base_filename + ".csv", sep=",", header=True, index=False)
    print("second last layer predictions: done in ", round(time.time() - predict_start_time), " seconds")
    
    print("Last layer output: in progress")
    predict_start_time=time.time()
    intermediate_layer_model = Model(input=model.input, output=model.get_layer("features_layer_2").output)
    intermediate_output = intermediate_layer_model.predict(test_set, batch_size=batch_size)
    intermediate_output__pd=pd.concat([ids_test__pd,
                            pd.DataFrame(intermediate_output)], 
                            axis=1) 
    
    intermediate_output_base_filename= d['EXECUTION_OUTPUT_DIRECTORY']+"intermediate_output_2_"+ model_output_filename[:-len(".hdf5")] 
    with open(intermediate_output_base_filename + ".pickle", 'wb') as handle:
        pickle.dump(intermediate_output__pd, handle, protocol=PICKLE_PROTOCOL)
    intermediate_output__pd.to_csv(intermediate_output_base_filename + ".csv", sep=",", header=True, index=False)
    print("Last layer predictions: done in ", round(time.time() - predict_start_time), " seconds")
    
#    print("Alternative predict: in progress")
#    predict_start_time=time.time()
#    output_layer_model = Model(input=model.get_layer("output_layer").input, output=model.output)
#    alternative_output = output_layer_model.predict(intermediate_output, batch_size=batch_size)
#    alternative_output__pd=pd.concat([ids_test__pd,
#                            pd.DataFrame(alternative_output)], 
#                            axis=1) 
#
#    alternative_output_base_filename= d['EXECUTION_OUTPUT_DIRECTORY']+"intermediate_output_"+ model_output_filename[:-len(".hdf5")] 
#    with open(intermediate_output_base_filename + ".pickle", 'wb') as handle:
#        pickle.dump(intermediate_output__pd, handle, protocol=PICKLE_PROTOCOL)
#    intermediate_output__pd.to_csv(intermediate_output_base_filename + ".csv", sep=",", header=True, index=False)
#    print("Alternative predictions: done in ", round(time.time() - predict_start_time), " seconds")


    '''
    # Plot the results
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(100)
    
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    print (plt.style.available) # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    '''
    
    num_fold+=1
    
    if num_fold>stop_after_fold:
        break
    


if stop_after_fold < num_folds:
    print("INCOMPLETE EXECUTION RESULTS")
    print("Only {} folds of {}".format(stop_after_fold, num_folds))
    
    if use_k_fold:
        oof_incomplete_results=pd.DataFrame(oof_preds[oof_val_index], columns={'prediction'})
        oof_incomplete_results['affected']=Y_train[oof_val_index][:,1]    
        oof_incomplete_score = log_loss(oof_incomplete_results['affected'], oof_incomplete_results['prediction'])
        print('OOF Score - (incomplete, {} observations): {}'.format(oof_val_index.shape[0], oof_incomplete_score))
        results=oof_incomplete_results
else:
    # FAIL!!!:   average needed!
    #train_results=pd.DataFrame(train_preds, columns={'prediction'})
    #train_results['affected']=Y_train[:,1]    
    train_score = log_loss(train_results['affected'], train_results['prediction'])
    print('TRAIN Score: {}'.format(train_score))
    report['train_score']=train_score
    

    if use_k_fold:
        oof_results=pd.DataFrame(oof_preds, columns={'prediction'})
        oof_results['affected']=Y_train[:,1]    
        oof_score = log_loss(oof_results['affected'], oof_results['prediction'])
        print('OOF Score: {}'.format(oof_score))
        report['oof_score']=oof_score
        results=oof_results
    else:
        results=train_results



if len(results['prediction'].unique())<d['NUM_CLASSES']:
        print("ERROR: predicted just {} different classes of {}".format(len(results['prediction'].unique()), d['NUM_CLASSES']))

            

results['prediction'][results['prediction']>=d['CLASS_1_THRESHOLD']]=1
results['prediction'][results['prediction']<d['CLASS_1_THRESHOLD']]=0
df_confusion = pd.crosstab(results['affected'], results['prediction'], margins=True)
print(df_confusion)

# REPORT_SAVING
report['DASHBOARD_ID']=d['DASHBOARD_ID']
report['execution_time']=(time.time() - start_time)

report['early_stopping_best_epoch']=list(report['early_stopping_best_epoch'])
report['train_scores']=list(report['train_scores'])
report['val_scores']=list(report['val_scores'])
report['dashboard']=d
report['history']=hist.history

report['ellapsed_time']=round(time.time() - start_time) #seconds

report['classes']=d['CLASSES_TXT']

try:
    with open(d['EXECUTION_OUTPUT_DIRECTORY'] + "report_" + str(d['DASHBOARD_ID']) + ".json" , 'w') as f:
        json.dump(report, f, indent=4, sort_keys=True)
except:
    print("Can't save to JSON")
with open(d['EXECUTION_OUTPUT_DIRECTORY'] + "report_" + str(d['DASHBOARD_ID']) + ".pickle", 'wb') as handle:
    pickle.dump(report, handle, protocol=PICKLE_PROTOCOL)
with open(d['EXECUTION_OUTPUT_DIRECTORY'] + "report_" + str(d['DASHBOARD_ID']) + ".yaml" , 'w')  as outfile:
    yaml.dump(report, outfile, default_flow_style=False)        
        
print("Ellapsed time: {} seconds".format(report['ellapsed_time']))










'''
class InMemoryBestModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(InMemoryBestModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

'''








'''
# Define model
model = Sequential()
model.add(Convolution3D(nb_filters[0],
                        nb_depth=nb_conv[0], 
                        nb_row=nb_conv[0], 
                        nb_col=nb_conv[0], 
                        input_shape=(1, img_rows, img_cols, patch_size), 
                        activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes,init='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
'''
'''
def cnn_model(input_shape, output_shape):
    inp = Input(shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    #Layer 1
    l1_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(inp)
    l1_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_conv1)
    l1_maxpool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(l1_conv2)
    
    #Layer 2
    l2_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_maxpool1)
    l2_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l2_conv1)
    l2_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l2_conv2)
    
    #Layer 3
    l3_conv1 = Convolution3D(64, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l2_maxpool1)
    l3_conv2 = Convolution3D(64, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l3_conv1)
    l3_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l3_conv2)
    
    #Layer 4
    l4_conv1 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l3_maxpool1)
    l4_conv2 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l4_conv1)
    l4_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l4_conv2)
    
    #Layer 5
    l5_conv1 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l4_maxpool1)
    l5_conv2 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l5_conv1)
    l5_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l5_conv2)
    
    #Layer 6
    l6_conv1 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l5_maxpool1)
    l6_conv2 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l6_conv1)
    l6_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l6_conv2)
    
    # Flatten
    flat = Flatten()(l6_maxpool1)
    dense1 = Dense(512, init='glorot_uniform', activation='relu')(flat)
    dense2 = Dense(64, init='glorot_uniform', activation='relu')(dense1)
    out = Dense(output_shape, activation='softmax')(dense2)
    
    model = Model(input=inp, output=out)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
'''