import numpy as np
import pandas as pd
import sys

from src.kfold import kFoldSplit
from src.preprocess import preprocessData
from src.data_import import loadData
from src.learning import prepareLearningData, prepareTestData, modelFunc, modelPredict, compute_MSE
import gc

submission = True

if __name__ == '__main__':
    print("Loading Data")
    learningData, testData = loadData()

    #Transform Data
    tData = preprocessData(learningData)
    del learningData; gc.collect()

    learningDataX, learningDataY = prepareLearningData(tData)
    del tData; gc.collect()

    print(learningDataX.shape, learningDataY.shape)

    params = {
        'max_depth': 4,
        'n_estimators': 10000,
        'objective': 'reg:linear',
        'silent': 0,
        'eta': 0.02,
        'eval_metric': 'mae',

        'base_score': 0.5,
        'colsample_bytree': 1,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_delta_step': 0,
        'min_child_weight': 1,
        'missing': None,
        'nthread': -1,
        'seed': 0,
        'subsample': 1}
    cols = learningDataX.columns
    testDataX = prepareTestData(testData, cols)
    clf = modelFunc(learningDataX, learningDataY, params)

    if submission :
        print("Predicting ...")
        predictY = modelPredict(clf, testDataX)

        sub = pd.read_csv('files/sample_submission.csv')
        for c in sub.columns[sub.columns != 'ParcelId']:
            sub[c] = predictY

        print('Writing csv ...')
        sub.to_csv('xgb_dev.csv', index=False, float_format='%.4f')
        sub.to_csv('xgb_dev.csv.gz', index=False, float_format='%.4f', compression='gzip')
    '''
    #EvalMethod ?
    MSE = compute_MSE(predictY, EvalDataY)
    print(MSE)

    print (MSEtab)
    print (np.mean(MSEtab))
    print (np.median(MSEtab))
    print (np.std(MSEtab))
    '''
    print ("OVER")