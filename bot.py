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
#export DISPLAY=localhost:0.0
df_BTC_FNG = pd.merge(df_BTC, df_FNG, on="date")
df_BTC_FNG = df_BTC_FNG.drop_duplicates()
df_BTC_FNG.dropna(inplace=True)
completeData = "BTC_FNG.csv"
df_BTC_FNG.to_csv(completeData, index=False)

#Show open and close data, then fng
#df_BTC_FNG[['open','close']].plot()
#plt.show()
#df_BTC_FNG[['fng_value']].plot()
#plt.show()

#---------- Split data into training and testing
#Drop date since scaler.fit cannot handle strings, could do some conversions but date doesn't matter anyways
df_BTC_FNG = df_BTC_FNG.drop(['date'], axis=1)
df_BTC_FNG = df_BTC_FNG[int(len(df_BTC_FNG)*0.75):]
split = int(0.97 * len(df_BTC_FNG))
train = df_BTC_FNG.iloc[:split]
test = df_BTC_FNG.iloc[split:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from keras.preprocessing.sequence import TimeseriesGenerator
#Define timeseries generator
n_size = 32
n_features = 6
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_size, batch_size=1)

#Example test
#X,y = generator[0]
#print(f'Given the array: \n{X.flatten()}')
#print(f'Predict this y: \n {y}')
#print(X.shape)

#Now to define the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_size, n_features)))
model.add(Dense(6))
model.compile(optimizer='adam', loss='mse')

model.fit(generator, epochs=5)

test_predictions = []
first_eval_batch = scaled_train[-n_size:]
current_batch = first_eval_batch.reshape((1, n_size, n_features))

for i in scaled_test:
  current_pred = model.predict(current_batch)[0]
  test_predictions.append(current_pred)
  current_actual = np.array(i)
  current_batch = np.append(current_batch[:,1:,:], [[current_actual]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
count = 0
cum_inaccuracy = 0
overestimations = 0
cum_overestimation_inaccuracy = 0
underestimations = 0
cum_underestimation_inaccuracy = 0
cum_change = 0

for i in test['close']:
  print(str(i) + " v.s. Prediction " + str(true_predictions[count][3]))
  if i > true_predictions[count][3]:
    underestimations += 1
    cum_underestimation_inaccuracy += abs(i - true_predictions[count][3]) / i
  elif i < true_predictions[count][3]:
    overestimations += 1
    cum_overestimation_inaccuracy += abs(i - true_predictions[count][3]) / i
  cum_inaccuracy += abs(i - true_predictions[count][3]) / i
  if count > 0:
    cum_change += abs(test['close'].iloc[count-1] - true_predictions[count][3])
  count += 1

change = cum_change / (count-1)
inaccuracy = cum_inaccuracy / count
overestimation_inaccuracy = cum_overestimation_inaccuracy / overestimations
underestimation_inaccuracy = cum_underestimation_inaccuracy / underestimations

print("Total inaccuracy is " + str(inaccuracy*100) + "%")
print("Number of overesimations is " + str(overestimations))
print("Total inaccuracy of overestimations is " + str(overestimation_inaccuracy*100) + "%")
print("Number of underestimations is " + str(underestimations))
print("Total inaccuracy of underestimations is " + str(underestimation_inaccuracy*100) + "%")
print("Average predicted change is " + str(change))