import numpy as np

from src.kfold import kFoldSplit
from src.preprocess import preprocessData
from src.data_import import loadData
from src.learning import prepareData, modelFunc, modelPredict, compute_MSE

if __name__ == '__main__':
    data = loadData()

    #Transform Data
    tData = preprocessData(data)

    learningDataX, learningDataY = prepareData(tData)

    params = {
        'max_depth': 5,
        'n_estimators': 50,
        'objective': 'reg:linear',

        'base_score': 0.5,
        'colsample_bytree': 1,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_delta_step': 0,
        'min_child_weight': 1,
        'missing': None,
        'nthread': -1,
        'seed': 0,
        'silent': True,
        'subsample': 1}

    # K - FOLD
    k = 5
    MSEtab = []
    for i in range(k):
        print ("K-Fold iteration: #" + str(i))
        kLearningDataX, kLearningDataY, kEvalDataX, kEvalDataY = kFoldSplit(learningDataX, learningDataY)

        model = modelFunc(kLearningDataX, kLearningDataY, params)
        predictY = modelPredict(model, kEvalDataX)

        #EvalMethod ?
        kMSE = compute_MSE(predictY, kEvalDataY)
        MSEtab.append(kMSE)

        print ("K-Fold iteration: #" + str(i) + " OVER.")

    print (MSEtab)
    print (np.mean(MSEtab))
    print (np.median(MSEtab))
    print (np.std(MSEtab))

    print ("OVER")