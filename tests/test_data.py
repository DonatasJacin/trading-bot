import os.path
from os.path import exists
from solution.solution import LoadFNG, PrepareCPI, CombineDatasets, PrepareFed, GetDataset, ReshapeData
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.chdir("/home/donatas/Desktop/github-repos/asset-predictor/data/")

# To run tests: python -m pytest in /trading-bot directory

#------- Test Definitions

def test_LoadFNG():
    assert (LoadFNG() == 1) and exists("FNG_Daily.csv")

def test_PrepareCPI():
    assert (PrepareCPI("constant") == 1) and (PrepareCPI("interpolate") == 1) and exists("CPIU_Daily.csv")

def test_PrepareFed():
    assert (PrepareFed("constant") == 1) and (PrepareFed("interpolate") == 1) and exists("FEDFunds_Daily.csv")

def test_CombineDatasets():
    assert (CombineDatasets() == 1) and exists("Combined.csv")

def test_GetDataset():
    assert (type(GetDataset()) == pd.DataFrame)

def test_ReshapeData():
    Combined_df = GetDataset()
    Combined_df = Combined_df.drop(['date'], axis = 1)
    ratio = 0.95
    split = round(len(Combined_df) * ratio)
    train = Combined_df[:split]
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    scaled_train = scaler.transform(train)
    train_X, train_Y = ReshapeData(1, 30, train)
