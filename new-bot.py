import requests
import csv
from csv import DictReader
import pandas as pd
import numpy as np

# Loads up-to-date FNG figures into "FNG_Daily.csv"
def LoadFNG():
    #Obtain FNG index data from API
    url = "https://api.alternative.me/fng/?limit=2000&format=csv&date_format=kr"
    response = requests.get(url)
    jsontext = response.text
    index = 0
    dataStartFound = False

    #Turn JSON response into csv
    for char in jsontext:
        if char == ']':
            dataEnd = index
            break
        if char == '2' and dataStartFound == False:
            dataStart = index - 1
            dataStartFound = True
    index += 1

    jsontext = 'date,fng_value,fng_classification' + jsontext[dataStart:dataEnd]

    filename = "FNG_Daily.csv"

    with open(filename, "w") as f:
        f.write(jsontext)

# Combines datasets for price data, FNG, and macroeconomic variables
def CombineDatasets():
    # Import BTC OHLCV dataset, FNG dataset, 52-week new high/low ratio. Convert all date columns into the same datetime datatype
    BTC_df = pd.read_csv("BTC_Daily.csv") #From CryptoDataDownload.com
    BTC_df['date'] = pd.to_datetime(BTC_df['date'])

    FNG_df = pd.read_csv("FNG_Daily.csv") #From alternative.me
    FNG_df['date'] = pd.to_datetime(FNG_df['date'])

    NYHLR_df = pd.read_csv("NYHLR_Daily.csv") #From StockCharts.com
    NYHLR_df['date'] = pd.to_datetime(NYHLR_df['date'])

    # Merge datasets on date column
    BTCFNG_df = pd.merge(BTC_df, FNG_df, on = 'date')
    Combined_df = pd.merge(BTCFNG_df, NYHLR_df, on = 'date')

    # Drop some unused columns and rename for convenience ---- Maybe keep day? Some evidence that fridays have more positive markets than monday --- For later, ask Yu
    Combined_df = Combined_df.drop(['unix', 'symbol', 'volume BTC', 'tradecount', 'fng_classification', 'high1', 'low1', 'close1', 'volume1', 'day'], axis = 1)
    Combined_df = Combined_df.rename(columns = {'volume USDT': 'volume', 'fng_value': 'fng'})

    Combined_df.to_csv('Combined.csv', index=False)


CombineDatasets()
