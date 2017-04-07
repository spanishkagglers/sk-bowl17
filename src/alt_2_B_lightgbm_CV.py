# https://www.kaggle.com/vanausloos/data-science-bowl-2017/cntk-and-lightgbm-quick-start-and-resnet-152
# This script has been released under the Apache 2.0 open source license.
#
# This script uses a pretrained ResNet model of 152 layers in CNTK and a boosted tree in LightGBM to 
# classify the data. It takes the next to last layer of ResNet to generate the features. Then
# they are averaged and fed to a tree. 
#
# With this configuration we were able to get a score on the leaderboard of 0.55979 with a execution
# time of 54min. Using ResNet 18, the score was 0.5708 and the execution time was 31min.
#
# This script is based on
# https://www.kaggle.com/drn01z3/data-science-bowl-2017/mxnet-xgboost-baseline-lb-0-57 
#

#Load libraries
from scipy.stats import gmean
import sys,os
import numpy as np
import dicom
import glob
import cv2
import time
import pandas as pd
from sklearn import cross_validation

from sklearn.model_selection import StratifiedKFold
#from cntk import load_model
#from cntk.ops import combine
#from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs

import sys
sys.path.append("../")
from competition_config import *
d=alternative_model_1a

import numpy as np
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)


try:
    from tqdm import tqdm
except:
    print('"pip install tqdm" to get the progress bar working!')
    tqdm = lambda x: x
from lightgbm.sklearn import LGBMRegressor


#Put here the number of your experiment
EXPERIMENT_NUMBER = '0042'

USE_GMEAN=True

USE_CNN_CLASSIFICATION=False

#Put here the path to the downloaded ResNet model
#model available at : https://migonzastorage.blob.core.windows.net/deep-learning/models/cntk/imagenet/ResNet_152.model
#other model: https://www.cntk.ai/Models/ResNet/ResNet_18.model
MODEL_PATH='../resnet-152/ResNet_152.model'

#Maximum batch size for the network to evaluate. 
BATCH_SIZE=60

#Put here the path where you downloaded all kaggle data
DATA_PATH='../'

# Path and variables
STAGE1_LABELS = os.path.join(DATA_PATH, 'stage1_labels.csv')
STAGE1_SAMPLE_SUBMISSION = os.path.join(DATA_PATH, 'stage1_sample_submission.csv')
STAGE1_FOLDER = os.path.join(DATA_PATH, 'stage1/')
EXPERIMENT_FOLDER = DATA_PATH + 'output/alt_2/features%s' % EXPERIMENT_NUMBER
FEATURE_FOLDER = EXPERIMENT_FOLDER #os.path.join(EXPERIMENT_FOLDER, 'features')
SUBMIT_OUTPUT='../output/alt_2/submit%s.csv' % EXPERIMENT_NUMBER
SUBMIT_OOF_OUTPUT='../output/alt_2/oof_%s.csv' % EXPERIMENT_NUMBER

if not os.path.exists(FEATURE_FOLDER):
    os.makedirs(FEATURE_FOLDER)

class Timer(object):
    """Timer class."""
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start = time.clock()

    def stop(self):
        self.end = time.clock()
        self.interval = self.end - self.start




'''
def get_train_test_ids():
    labels__pd=pd.read_csv(STAGE1_LABELS)#d['BOWL_LABELS'])
    all_patients=sorted(list(next(os.walk(STAGE1_FOLDER)[1]))) #d['BOWL_PATIENTS'])
    
    train_set_patients=list(labels__pd['id'].values)
    test_set_patients=[x for x in all_patients if x not in train_set_patients]
    
    return train_set_patients, test_set_patients, all_patients
'''


def train_lightgbm(verbose=True):
    """Train a boosted tree with LightGBM."""
    if verbose: print("Training with LightGBM")
    df = pd.read_csv(STAGE1_LABELS)
    x = np.array([np.mean(np.load(os.path.join(FEATURE_FOLDER,'%s.npy' % str(id))), axis=0).flatten() for id in df['id'].tolist()])
    y = df['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)
                                                                   
    '''
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': 21,
        'learning_rate': 0.001,
        'nthread':24,
        'subsample':0.80,
        'colsample_bytree':0.80,
        'seed':42,
        'verbose': verbose,
    }
    
    train_set_patients, _, _ = get_train_test_ids()
    ids = pd.DataFrame(train_set_patients, columns=['id'])
    '''
    ids=df['id']
    
#    cnn_output=pd.read_csv('../output/14_Nodule_3D_classifier/CNN-5_classes_5_epochs_5_early_stopping_2017-03-28_22.53.01/test_predictions_CNN-5_classes_5_epochs_5_early_stopping_2017-03-28_22.53.01_fold_1.csv')
#    cnn=pd.concat(df['id'],cnn_output,
    
    
    skf = StratifiedKFold(n_splits=5, random_state=2048, shuffle=True)
#    result =[]
    clfs = []
    oof_preds=[]
    fold=0
    for train_index, test_index in skf.split(x, y):
        print("FOLD: " , fold+1)
        print("===========")
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]
        
        val_ids=pd.DataFrame(ids.iloc[test_index].values, columns=['id'])
        
        early_stopping_rounds=300
        
        clf = LGBMRegressor(max_depth=50,
                    num_leaves=21,
                    n_estimators=5000,
                    min_child_weight=1,
                    learning_rate=0.001,
                    nthread=24,
                    subsample=0.80,
                    colsample_bytree=0.80,
                    seed=42)
        
        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=verbose, eval_metric='l2', early_stopping_rounds=early_stopping_rounds)
        
        val_preds=pd.DataFrame(clf.predict(val_x), columns=["cancer"])
        oof_preds.append(pd.concat([val_ids, val_preds], axis=1))
        clfs.append(clf)
        fold+=1
    
    oof_preds=pd.concat(oof_preds,axis=0)
        
    return clfs, oof_preds


def compute_training(verbose=True):
    """Wrapper function to perform the training."""
    if verbose: print("Compute training")
    with Timer() as t:
        clf = train_lightgbm()
    if verbose: print("Training took %.03f sec.\n" % t.interval)
    return clf

'''
def compute_prediction(clf, verbose=True):  
    """Wrapper function to perform the prediction."""
    if verbose: print("Compute prediction")
    df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
    x = np.array([np.mean(np.load(os.path.join(FEATURE_FOLDER, '%s.npy' % str(id))), axis=0).flatten() for id in df['id'].tolist()])
    
    with Timer() as t:
        pred = clf.predict(x)
    if verbose: print("Prediction took %.03f sec.\n" % t.interval)
    df['cancer'] = pred
    return df


def save_results(df, verbose=True):
    """Wrapper function to save the results."""
    if verbose: print("Save results to csv")
    df.to_csv(SUBMIT_OUTPUT, index=False)
    if verbose: 
        print("Results:")
        print(df.head())
    '''
if __name__ == "__main__":
    
#    calc_features(verbose=False)
    clfs, oof_preds = compute_training()
#    df = compute_prediction(clf)
#    save_results(df)
    
    '''    
    df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
    x = np.array([np.mean(np.load(os.path.join(FEATURE_FOLDER, '%s.npy' % str(id))), axis=0).flatten() for id in df['id'].tolist()])
    '''    
    
    ''' 
    results_avg=np.average(result)
    print(np.average(result))
    _, test_set_patients= get_train_test_ids()
    df = pd.DataFrame(test_set_patients, columns=['id'])
#    df.rename(columns={'ct_scan_id':'id'}, inplace=True)
    x = get_test_set()
    '''
    
    df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
    x = np.array([np.mean(np.load(os.path.join(FEATURE_FOLDER, '%s.npy' % str(id))), axis=0).flatten() for id in df['id'].tolist()])
 
    
    preds = []
    
    for clf in clfs:
        if USE_GMEAN:
            print("Using GMEAN")
            preds.append(np.clip(clf.predict(x),0.0001,1))
        else:
            preds.append(clf.predict(x))

    if USE_GMEAN:
        print("Using GMEAN")
        pred = gmean(np.array(preds), axis=0)
    else:
        pred = np.mean(np.array(preds), axis=0)
        
    df['cancer'] = pred
    df.to_csv(SUBMIT_OUTPUT, index=False)
    print(df.head())
    
    
    
    oof_preds.to_csv(SUBMIT_OOF_OUTPUT, index=False)
    
    





