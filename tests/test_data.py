import os.path
from os.path import exists
from bot.newbot import LoadFNG, CombineDatasets

os.chdir("/home/donatas/Desktop/github-repos/trading-bot/data/")

# To run tests: python -m pytest in /trading-bot directory

#------- Test Definitions

def test_LoadFNG():
    assert (LoadFNG() == 1) and exists("FNG_Daily.csv")

def test_CombineDatasets():
    assert (CombineDatasets() == 1) and exists("Combined.csv")
