import pandas as pd
import numpy as np

def loadData():
    #2016 only
    train = pd.read_csv('files/train_2016_v2.csv')
    prop = pd.read_csv('files/properties_2016.csv')
    sample = pd.read_csv('files/sample_submission.csv')

    print('Binding to float32')

    for c, dtype in zip(prop.columns, prop.dtypes):
        if dtype == np.float64:
            prop[c] = prop[c].astype(np.float32)

    print('Creating training set ...')
    df_train = train.merge(prop, how='left', on='parcelid')

    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop, on='parcelid', how='left')

    return df_train, df_test