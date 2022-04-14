import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import csv
from csv import DictReader

badFormat = ['12-AM', '01-AM', '02-AM', '03-AM', '04-AM', '05-AM', '06-AM', '07-AM', '08-AM', '09-AM', '10-AM', '11-AM', '12-PM', '01-PM', '02-PM', '03-PM', '04-PM', '05-PM', '06-PM', '07-PM', '08-PM', '09-PM', '10-PM', '11-PM',]
goodFormat = ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00', '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00']

#---------- Obtain FNG index data
url = "https://api.alternative.me/fng/?limit=2000&format=csv&date_format=kr"
response = requests.get(url)
jsontext = response.text
index = 0
dataStartFound = False

#Turn JSON response into csv
for char in jsontext:
  if char == "]":
    dataEnd = index
    break
  if char == "2" and dataStartFound == False:
    dataStart = index - 1
    dataStartFound = True
  index += 1

jsontext = "date,fng_value,fng_classification" + jsontext[dataStart:dataEnd]

filename1 = "FNGUnmult.csv"
filename2 = "FNG.csv"

with open(filename1, "w") as f:
  f.write(jsontext)

with open(filename1, "r") as read_obj:
  with open(filename2, "w") as write_obj:
    csv_dict_reader = DictReader(read_obj)
    writer = csv.writer(write_obj)
    writer.writerow(["date", "fng_value"])
    for row in csv_dict_reader:
      for time in goodFormat:
        if row['fng_value'] == None:
          continue
        datetime = row['date'] + " " + time
        writer.writerow([datetime, row['fng_value']])

df_FNG = pd.read_csv(filename2)
df_FNG = df_FNG.iloc[::-1]
df_FNG.dropna(inplace=True)

#---------- Read BTC/USDT historical hourly data, remove some columns and reverse dataframe
filename = "/mnt/d/trading-bot/trading-bot/Binance_BTCUSDT_1h.csv"
newfilename = "/mnt/d/trading-bot/trading-bot/BTCUSDTH.csv"

with open(filename, "r") as read_obj:
  with open(newfilename, "w") as write_obj:
    csv_dict_reader = DictReader(read_obj)
    writer = csv.writer(write_obj)
    rowIndex = 0
    writer.writerow(["date", "open", "high", "low", "close", "Volume USDT"])
    for row in csv_dict_reader:
      if row['date'] != df_FNG.iloc[1,]['date'] + " 12-AM":
        if str(row['date'][11:]) in badFormat:
          dateIndex = badFormat.index(row['date'][11:])
          goodDate = row['date'][:10] + " " + goodFormat[dateIndex]
          writer.writerow([goodDate, row['open'], row['high'], row['low'], row['close'], row['Volume USDT']])
        else:
          writer.writerow([row['date'], row['open'], row['high'], row['low'], row['close'], row['Volume USDT']])
      else:
        break

df_BTC = pd.read_csv(newfilename)
df_BTC = df_BTC.iloc[::-1]
df_BTC.dropna(inplace=True)

#---------- Combine BTCUSDT data and FNG data, then make csv file

df_BTC_FNG = pd.merge(df_BTC, df_FNG, on="date")
df_BTC_FNG = df_BTC_FNG.drop_duplicates()
df_BTC_FNG.dropna(inplace=True)
completeData = "BTC_FNG.csv"
df_BTC_FNG.to_csv(completeData, index=False)

df_BTC_FNG.plot()



