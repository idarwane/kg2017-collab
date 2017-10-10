import numpy as np
import pandas as pd
import sklearn as sk
#import xgboost

from data_import import loadData
from preprocess import preprocessData
from learning import preprareDta, modelFunc, modelPredict, compute_MSE
from kfold import kFoldSplit

if __name__ == '__main__':
    data = loadData()

    #Transform Data
    tData = preprocessData(data)

    learningDataX, learningDataY = prepareData(tData)

    # K - FOLD
    k = 5
    MSEtab = []
    for i in range(k):
        print ("K-Fold iteration: #" + str(i))
        kLearningDataX, kLearningDataY, kEvalDataX, kEvalDataY = kFoldSplit(learningDataX, learningDataY)

        model = modelFunc(kLearningDataX, kLearningDataY)
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