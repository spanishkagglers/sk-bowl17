# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:08:19 2017

@author: spanish kagglers
"""

import os
#os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO"]="device=gpu,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic;"
import keras.backend as K
K.set_image_dim_ordering('th') #?????????????????????????????????????

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

if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])

if 'TEMP_DIRECTORY' in d and not os.path.exists(d['TEMP_DIRECTORY']):
    os.makedirs(d['TEMP_DIRECTORY'])

if not os.path.exists(d['EXECUTION_OUTPUT_DIRECTORY']):
    os.makedirs(d['EXECUTION_OUTPUT_DIRECTORY'])


img_rows=img_cols=img_depth=chunck_size=d['CHUNK_SIZE'] # ¿X, Y,?, Z



def read_luna_annotations_csv(filename):
    annotations__pd = pd.read_csv(filename, sep=",")
    annotations__pd['nodule_id']=["ann"+str(x) for x in annotations__pd.index.values]
#    annotations__pd['nodule_id']=annotations__pd['nodule_id'].index.apply(lambda x: "ann"+str(x))
    return annotations__pd

luna_annotations__pd=read_luna_annotations_csv(d['INPUT_LUNA_NODULES_METADATA'])


luna_chunks_file_list=sorted(glob(d['LUNA_INPUT_DIRECTORY']+"*.pickle"))
luna_chunks_list=[os.path.basename(x)[:-len(".pickle")] for x in luna_chunks_file_list]
affected_nodules=luna_annotations__pd.loc[luna_annotations__pd['diameter_mm']>=2*d['RADIUS_THRESHOLD_MM']]['nodule_id'].values
affected_file_list=[d['LUNA_INPUT_DIRECTORY']+x+".pickle" for x in luna_chunks_list if x in affected_nodules]
healthy_file_list=[d['LUNA_INPUT_DIRECTORY']+x +".pickle"for x in luna_chunks_list if x not in affected_nodules]


X_tr=[]



for input_filename in tqdm(affected_file_list):
    try:
        with open(input_filename, 'rb') as handle:
            nodule_3d = pickle.load(handle, encoding='latin1')
            if(chunck_size<nodule_3d.shape[0]):
                i=round((nodule_3d.shape[0]-chunck_size)/2)
                j=i+chunck_size
                nodule_3d=nodule_3d[i:j,i:j,i:j]
    except Exception as e:
        print("No se pudo leer ", input_filename)
        print(e)
    
    X_tr.append(nodule_3d)

num_affected = len(affected_file_list)


for input_filename in tqdm(healthy_file_list):
    try:
        with open(input_filename, 'rb') as handle:
            nodule_3d = pickle.load(handle, encoding='latin1')
            if(chunck_size<nodule_3d.shape[0]):
                i=round((nodule_3d.shape[0]-chunck_size)/2)
                j=i+chunck_size
                nodule_3d=nodule_3d[i:j,i:j,i:j]
    except Exception as e:
        print("No se pudo leer ", input_filename)
        print(e)
        
    
    X_tr.append(nodule_3d)

num_healthy = len(healthy_file_list)

print("\nnum_affected: " + str(num_affected))
print("num_healthy: " + str(num_healthy))




label=np.hstack((
    np.ones((num_affected,),dtype = int),
    np.zeros((num_healthy,),dtype = int)
))


X_tr_array = np.array(X_tr) 

num_samples = len(X_tr_array)
print (num_samples)
train_data = [X_tr_array,label]

(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)

train_set = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))

for h in range(num_samples):
    train_set[h][0][:][:][:]=X_train[h,:,:,:]
 

patch_size = img_depth    # img_depth or number of frames used for each video

print(train_set.shape, 'train samples')    


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

# Pre-processing

train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)



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


def cnn_model(input_shape, output_shape):
    inp = Input(shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    #Layer 1
    l1_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(inp)
    l1_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_conv1)
    l1_maxpool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(l1_conv2)
    
    # Flatten
    flat = Flatten()(l1_maxpool1)
    dense1 = Dense(512, init='glorot_uniform', activation='relu')(flat)
    dense2 = Dense(64, init='glorot_uniform', activation='relu')(dense1)
    out = Dense(output_shape, activation='softmax')(dense2)
    
    model = Model(input=inp, output=out)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


model = cnn_model((1, img_rows, img_cols, img_depth), 2)








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

#model = get_model(1,img_rows, img_cols, img_depth, backend=None)
 
# Split the data

#X_train_fold, X_val_fold, y_train_fold,y_val_fold =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

report={
    'early_stopping_best_epoch':[],
    'train_scores': [],
    'val_scores': [],

}



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
    


num_fold=1
for train_index, val_index in validation_indexes:
    print("Fold: {} / {}".format(num_fold,num_folds))
    print("========")
    
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
    
    hist = model.fit(X_train_fold,
            y_train_fold,
            validation_data=validation_data,
            batch_size=batch_size,
            nb_epoch = nb_epoch,
            shuffle=True
            ,callbacks= callbacks_list
    ) #show_accuracy=True,
    

    
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
            print("Fold: {} - WARN: predicted just {} different classes of []".format(num_fold, len(results_val['prediction'].unique()), d['NUM_CLASSES']))
        oof_preds[val_index] = preds[:,1]
        print('Validation Score - LogLoss fold {}: {}'.format(num_fold, score))
    
    
    train_preds=model.predict(X_train_fold, batch_size=batch_size)
    train_results=pd.DataFrame(train_preds[:,1].astype(np.float), columns={'prediction'})
    train_results['affected']=y_train_fold[:,1]
    
    train_score = log_loss(train_results['affected'], train_results['prediction'])    
    report['train_scores']+=[train_score]
    
    print('Train Score      - LogLoss fold {}: {}'.format(num_fold, train_score))


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
        print("ERROR: predicted just {} different classes of []".format(num_fold, len(results['prediction'].unique()), d['NUM_CLASSES']))

            

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

try:
    with open(d['EXECUTION_OUTPUT_DIRECTORY'] + "report_" + str(d['DASHBOARD_ID']) + ".json" , 'w') as f:
        json.dump(report, f, indent=4, sort_keys=True)
except:
    print("Can't save to JSON")
with open(d['EXECUTION_OUTPUT_DIRECTORY'] + "report_" + str(d['DASHBOARD_ID']) + ".pickle", 'wb') as handle:
    pickle.dump(report, handle, protocol=PICKLE_PROTOCOL)
with open(d['EXECUTION_OUTPUT_DIRECTORY'] + "report_" + str(d['DASHBOARD_ID']) + ".yaml" , 'w')  as outfile:
    yaml.dump(report, outfile, default_flow_style=False)        
        
print("Ellapsed time: {} seconds".format((time.time() - start_time)))

