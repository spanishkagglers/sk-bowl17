# This script has been released under the Apache 2.0 open source license.
# Based on:
# https://www.kaggle.com/akilaw/data-science-bowl-2017/resnet50-features-xgboost/comments


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import gmean


import sys
sys.path.append("../")
from competition_config import *
d=alternative_model_1c

seed=d['RANDOM_SEED']
if len(sys.argv)>1:
    import hashlib, ctypes
    seed=str(sys.argv[1])
    seed=hashlib.sha224(seed.encode('utf-8')).hexdigest()
    seed=abs(int(hash(str(seed))))
    seed=ctypes.c_uint32(seed).value
print("seed is: ", seed)

ABSOLUTE_VALUES = "?"
if len(sys.argv)>2:
    ABSOLUTE_VALUES=str(sys.argv[2])
    
WHITEN = '?'
if len(sys.argv)>3:
    WHITEN=str(sys.argv[3])

SVD_SOLVER ='?'
if len(sys.argv)>4:
    SVD_SOLVER=str(sys.argv[4])

if not os.path.exists(d['OUTPUT_DIRECTORY']):
    os.makedirs(d['OUTPUT_DIRECTORY'])



def get_decomposition_features():
    with open(d['INPUT_DIRECTORY'] + 'pca_resnet_features_average_pandas.pickle', 'rb') as handle:
        features = pickle.load(handle)
    
    for feature_id in [x for x in features.columns if x!='ct_scan_id']:
        features[feature_id]=features[feature_id]*1000
        features[feature_id]=features[feature_id].astype(int)
    return features

def get_train_test_ids():
    labels__pd=pd.read_csv(d['BOWL_LABELS'])
    all_patients=next(os.walk(d['BOWL_PATIENTS']))[1]
    
    train_set_patients=list(labels__pd['id'].values)
    test_set_patients=[x for x in all_patients if x not in train_set_patients]
    
    return train_set_patients, test_set_patients

def get_train_set():
    train_set_patients, _ = get_train_test_ids()
    
    features = get_decomposition_features()
    
    train_set= features.loc[features['ct_scan_id'].isin(train_set_patients)]
    del(train_set['ct_scan_id'])
    
    return train_set.as_matrix()

def get_test_set():
    _, test_set_patients= get_train_test_ids()
    
    features = get_decomposition_features()
    
    test_set= features.loc[features['ct_scan_id'].isin(test_set_patients)]
    del(test_set['ct_scan_id'])
    
    return test_set.as_matrix()
    
zz=get_decomposition_features()    
zzz=get_train_set()
def train_xgboost():
    df = pd.read_csv(d['BOWL_LABELS'])
    
    train_set_patients, _ = get_train_test_ids()
    ids = pd.DataFrame(train_set_patients, columns=['id'])

    x = get_train_set()
    y = df['cancer'].as_matrix()

    skf = StratifiedKFold(n_splits=5, random_state=2048, shuffle=True)
    result =[]
    clfs = []
    oof_preds=[]
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]
        
        val_ids=pd.DataFrame(ids.iloc[test_index].values, columns=['id'])

        #clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03756, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        # max_depth=5 Public score = ?
        # max_depth=4 Public score = 0.54721
        # max_depth=3 Public score = 0.55193
        
        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
        result.append(clf.best_score)
        clfs.append(clf)
        
        val_preds=pd.DataFrame(clf.predict(val_x), columns=["cancer"])
        oof_preds.append(pd.concat([val_ids, val_preds], axis=1))
    
    oof_preds=pd.concat(oof_preds,axis=0)
        

    return clfs, result, oof_preds


def make_submite():
    clfs, result, oof_preds = train_xgboost()
    results_avg=np.average(result)
    print(np.average(result))
    _, test_set_patients= get_train_test_ids()
    df = pd.DataFrame(test_set_patients, columns=['id'])
#    df.rename(columns={'ct_scan_id':'id'}, inplace=True)
    x = get_test_set()
    preds = []
    
    for clf in clfs:
        preds.append(np.clip(clf.predict(x),0.0001,1))

    pred = gmean(np.array(preds), axis=0)
    df['cancer'] = pred
    df.to_csv(d['OUTPUT_DIRECTORY'] + "submission_resnet_xgboost.csv", index=False)
    print(df.head())
    
    oof_preds.to_csv(d['OUTPUT_DIRECTORY'] + "oof_resnet_xgboost.csv", index=False)
    
    
    RESULTS_FILE_NAME="seeds.csv"
    print("seed is: ", seed)
    row="{};{};{};{};{}\n".format(seed, results_avg, ABSOLUTE_VALUES,WHITEN,SVD_SOLVER)
    if not os.path.isfile(RESULTS_FILE_NAME):
        row = "seed;score;absolute;whiten;SVD_SOLVER\r\n" + row
    with open(RESULTS_FILE_NAME, 'a+', newline='', encoding='utf-8') as f:
        f.write(row)


if __name__ == '__main__':
    make_submite()