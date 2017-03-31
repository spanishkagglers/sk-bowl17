# This script has been released under the Apache 2.0 open source license.
# Based on:
# https://www.kaggle.com/akilaw/data-science-bowl-2017/resnet50-features-xgboost


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import gmean


import sys
sys.path.append("../")
from alternative_model_1b import *
d=alternative_model_1b




def train_xgboost():
    df = pd.read_csv(d['BOWL_LABELS'])

    x = get_trn()
    y = df['cancer'].as_matrix()

    skf = StratifiedKFold(n_splits=5, random_state=2048, shuffle=True)
    result =[]
    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        #clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03756, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        # max_depth=5 Public score = ?
        # max_depth=4 Public score = 0.54721
        # max_depth=3 Public score = 0.55193
        
        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
        result.append(clf.best_score)
        clfs.append(clf)

    return clfs, result


def make_submite():
    clfs, result = train_xgboost()
    print(np.average(result))
    df = pd.read_csv('../input/stage1_sample_submission.csv')
    x = get_tst()
    preds = []
    
    for clf in clfs:
        preds.append(np.clip(clf.predict(x),0.0001,1))

    pred = gmean(np.array(preds), axis=0)
    df['cancer'] = pred
    df.to_csv('submission_20170309-01.csv', index=False)
    print(df.head())


if __name__ == '__main__':
    make_submite()