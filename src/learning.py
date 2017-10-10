import numpy as np
from xgboost import XGBRegressor

def prepareData(df):
    #basic split X/Y with Y = 1 col
    x = df[:-1]
    y = df[-1]
    return x, y

def modelFunc(X, Y, params):
    #XGBOOST algorithm
    model = XGBRegressor(**params)
    model.fit(X, Y)

    return model

def modelPredict(model, X_toPred):
    return model.predict(X_toPred)

def compute_MSE(y, yhat):
    return np.mean((y-yhat)**2)