import os.path
from os.path import exists
from solution.solution import LoadFNG, PrepareCPI, CombineDatasets

os.chdir("/home/donatas/Desktop/github-repos/asset-predictor/data/")

# To run tests: python -m pytest in /trading-bot directory

#------- Test Definitions

def test_LoadFNG():
    assert (LoadFNG() == 1) and exists("FNG_Daily.csv")

def test_PrepareCPI():
    assert (PrepareCPI() == 1) and exists("CPIU_Daily.csv")

def test_CombineDatasets():
    assert (CombineDatasets() == 1) and exists("Combined.csv")

