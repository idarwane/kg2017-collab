import pandas as pd

def loadData():

    return pd.read_csv("path.csv", parse_dates=["trdate"])
    #TBD