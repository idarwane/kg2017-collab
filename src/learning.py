import numpy as np
import xgboost as xgb
import gc

def prepareLearningData(df):
    #basic split X/Y with Y = 1 col
    x = df.drop(['parcelid', 'logerror',
                 'transactiondate', 'propertyzoningdesc',
                 'propertycountylandusecode'], axis=1)
    y = df['logerror'].values

    for c in x.dtypes[x.dtypes == object].index.values:
        x[c] = (x[c] == True)

    return x, y

def prepareTestData(df_test, cols):
    x_test = df_test[cols]
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)

    return x_test

def modelPredict(clf, X_toPred):
    d_toPred = xgb.DMatrix(X_toPred)
    del X_toPred; gc.collect()

    return clf.predict(d_toPred)

def modelFunc(X, Y, params):
    #XGBOOST algorithm
    print ("Learning - Start")

    split = 80000
    x_train, y_train, x_valid, y_valid = X[:split], Y[:split], X[split:], Y[split:]

    print('Building DMatrix...')

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    del x_train, x_valid;gc.collect()

    print('Training ...')

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, params['n_estimators'], watchlist, early_stopping_rounds=100, verbose_eval=10)

    del d_train, d_valid; gc.collect()
    return clf

    #data_matrix = xgb.DMatrix(X, Y)
    #model = xgb.train(dtrain=data_matrix, params=params)

    #return model



def compute_MSE(y, yhat):
    return np.mean((y-yhat)**2)