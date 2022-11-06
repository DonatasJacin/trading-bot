import requests
import csv
from csv import DictReader
import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt

os.chdir("/home/donatas/Desktop/github-repos/asset-predictor/data/")

# Loads up-to-date FNG figures into "FNG_Daily.csv"
def LoadFNG():
    #Obtain FNG index data from API
    url = "https://api.alternative.me/fng/?limit=2000&format=csv&date_format=kr"
    response = requests.get(url)
    json_text = response.text
    index = 0
    dataStartFound = False

    #Turn JSON response into csv
    for char in json_text:
        if char == ']':
            dataEnd = index
            break
        if char == '2' and dataStartFound == False:
            dataStart = index - 1
            dataStartFound = True
        index += 1

    json_text = 'date,fng_value,fng_classification' + json_text[dataStart:dataEnd]

    file_name = "FNG_Daily.csv"

    with open(file_name, "w") as f:
        f.write(json_text)
        return 1

# Upsamples monthly inflation data and interpolates, stores in "CPIU_Daily.csv"
def PrepareCPI():
    # Import monthly inflation data
    CPIU_df = pd.read_csv("CPIU_Monthly.csv", parse_dates=['date_of_publication'])
    CPIU_df = CPIU_df.drop(['month'], axis = 1)
    CPIU_df = CPIU_df.set_index('date_of_publication')
    CPIU_df = CPIU_df.resample('D', convention = 'start').interpolate()

    CPIU_df.to_csv("CPIU_Daily.csv")
    return 1


# Combines datasets for price data, FNG, and macroeconomic variables, stores in "Combined.csv"
def CombineDatasets():
    # Import BTC OHLCV dataset, FNG dataset, 52-week new high/low ratio. Convert all date columns into the same datetime datatype
    BTC_df = pd.read_csv("BTC_Daily.csv") #From CryptoDataDownload.com
    BTC_df['date'] = pd.to_datetime(BTC_df['date'])

    FNG_df = pd.read_csv("FNG_Daily.csv") #From alternative.me
    FNG_df['date'] = pd.to_datetime(FNG_df['date'])

    FED_df = pd.read_csv("FEDFunds_Daily.csv")
    FED_df['date'] = pd.to_datetime(FED_df['date'])

    CPIU_df = pd.read_csv("CPIU_Daily.csv")
    CPIU_df = CPIU_df.rename(columns = {'date_of_publication' : 'date'})
    CPIU_df['date'] = pd.to_datetime(CPIU_df['date'])

    # Merge datasets on date column
    Combined_df = pd.merge(BTC_df, FNG_df, on = 'date')
    Combined_df = pd.merge(Combined_df, FED_df, on = 'date')
    Combined_df = pd.merge(Combined_df, CPIU_df, on = 'date')

    # Drop some unused columns and rename for convenience ---- Maybe keep day? Some evidence that fridays have more positive markets than monday --- For later, ask Yu
    Combined_df = Combined_df.drop(['unix', 'symbol', 'volume BTC', 'tradecount', 'fng_classification'], axis = 1)
    Combined_df = Combined_df.rename(columns = {'volume USDT': 'volume', 'fng_value': 'fng', 'value': 'inflation'})

    Combined_df = Combined_df[::-1]
    Combined_df.to_csv('Combined.csv', index=False)
    return 1

columns = ['date','open','high','low','close','volume','fng','fundrate','inflation']
Combined_df = pd.read_csv("Combined.csv", usecols = columns)

fig, ax = plt.subplots()
ax.plot(Combined_df.close, color = "red")
ax.set_xlabel("Date", fontsize = 14)
ax.set_ylabel("Close Prices", fontsize = 14)

ax2 = ax.twinx()
ax2.plot(Combined_df.inflation, color = "blue")
ax2.set_ylabel("Inflation Rate", fontsize = "14")

plt.show()