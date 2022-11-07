import requests
import csv
from csv import DictReader
import pandas as pd
import numpy as np
import os.path
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense, Dropout

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
    CPIU_df = pd.read_csv("CPIU_Monthly.csv", parse_dates = ['date_of_publication'])
    CPIU_df = CPIU_df.drop(['month'], axis = 1)
    CPIU_df = CPIU_df.set_index('date_of_publication')
    CPIU_df = CPIU_df.resample('D').interpolate()

    CPIU_df.to_csv("CPIU_Daily.csv")
    return 1

#Upsamples daily federal fund rate data (not tracked on weekends) and interpolates, stores in "FEDFunds_Daily.csv"
def PrepareFed():
    # Import federal fund rate data
    Fed_df = pd.read_csv("FEDFunds.csv", parse_dates = ['date'])
    Fed_df = Fed_df.reset_index(drop=True)
    Fed_df = Fed_df.set_index('date')
    Fed_df = Fed_df.resample('D').interpolate()

    Fed_df.to_csv("FEDFunds_Daily.csv")
    return 1

# Combines datasets for price data, FNG, and macroeconomic variables, stores in "Combined.csv"
def CombineDatasets():
    # Import BTC OHLCV dataset, FNG dataset, 52-week new high/low ratio. Convert all date columns into the same datetime datatype
    BTC_df = pd.read_csv("BTC_Daily.csv") #From CryptoDataDownload.com
    BTC_df['date'] = pd.to_datetime(BTC_df['date'])

    FNG_df = pd.read_csv("FNG_Daily.csv") #From alternative.me
    FNG_df['date'] = pd.to_datetime(FNG_df['date'])

    Fed_df = pd.read_csv("FEDFunds_Daily.csv")
    Fed_df['date'] = pd.to_datetime(Fed_df['date'])

    CPIU_df = pd.read_csv("CPIU_Daily.csv")
    CPIU_df = CPIU_df.rename(columns = {'date_of_publication' : 'date'})
    CPIU_df['date'] = pd.to_datetime(CPIU_df['date'])

    # Merge datasets on date column
    Combined_df = pd.merge(BTC_df, FNG_df, on = 'date')
    Combined_df = pd.merge(Combined_df, Fed_df, on = 'date')
    Combined_df = pd.merge(Combined_df, CPIU_df, on = 'date')

    # Drop some unused columns and rename for convenience ---- Maybe keep day? Some evidence that fridays have more positive markets than monday --- For later, ask Yu
    Combined_df = Combined_df.drop(['unix', 'symbol', 'volume BTC', 'tradecount', 'fng_classification'], axis = 1)
    Combined_df = Combined_df.rename(columns = {'volume USDT': 'volume', 'fng_value': 'fng', 'value': 'inflation'})

    Combined_df = Combined_df[::-1]
    Combined_df.to_csv('Combined.csv', index=False)
    return 1

def GetDataset():
    # Check if datasets have been prepared, if not, run corresponding functions
    reload = 0
    if not exists("FNG_Daily.csv"):
        LoadFNG()
        reload = 1
    if not exists("CPIU_Daily.csv"):
        PrepareCPI()
        reload = 1
    if not exists("FEDFunds_Daily.csv"):
        PrepareFed()
        reload = 1
    if not exists("Combined.csv") or reload == 1:
        CombineDatasets()

    Combined_df = pd.read_csv("Combined.csv")
    return Combined_df

# Reshapes data into shape (n_samples * timesteps * n_features)
def ReshapeData(n_future, n_lookback, prediction_param_index, df):
    train_X, train_Y = [], []
    for i in range(n_lookback, len(df) - n_future + 1):
        train_X.append(df[i - n_lookback:i, 0:df.shape[1]])
        train_Y.append(df[i + n_future - 1:i + n_future, prediction_param_index])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    return train_X, train_Y
    
Combined_df = GetDataset()
# Dates are extracted for plotting purposes
dates = pd.to_datetime(Combined_df['date'])
Combined_df = Combined_df.drop(['date'], axis = 1)

# Split data into training/validation and testing
ratio = 0.98
split = round(len(Combined_df) * ratio)
train = Combined_df[:split]
test = Combined_df[split:]

# Normalise dataset
scaler = MinMaxScaler()
scaler = scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

days_to_predict = 1
days_to_use = 30
train_X, train_Y = ReshapeData(days_to_predict, days_to_use, 3, scaled_train)

#print("scaled_df shape == {}.".format(scaled_train.shape))
#print("train_X shape == {}.".format(train_X.shape))
#print("train_Y shape == {}.".format(train_Y.shape))

n_size = train_X.shape[1]
n_features = train_X.shape[2]

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(n_size, n_features), return_sequences=False)) # Can only use CuDNNLSTMs if GPU is enabled for tensorflow, default activation is tanh
model.add(Dense(train_Y.shape[1]))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

history = model.fit(train_X, train_Y, epochs=100, batch_size=8, validation_split=0.1, verbose=1)

test_predictions = []
first_eval_batch = scaled_train[-n_size:]
current_batch = first_eval_batch.reshape((1, n_size, n_features))

for i in scaled_test:
  current_pred = model.predict(current_batch)[0]
  test_predictions.append(current_pred)
  current_actual = np.array(i)
  current_batch = np.append(current_batch[:,1:,:], [[current_actual]], axis=1)

#Perform inverse transformation to rescale back to original range
#Since we used 8 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 8 times and discard them after inverse transform
prediction_copies = np.repeat(test_predictions, Combined_df.shape[1], axis=-1)
true_predictions = scaler.inverse_transform(prediction_copies)[:,0]

prediction_dates = dates[split:]

columns = ['date','open','high','low','close','volume','fng','fundrate','inflation']
Combined_df = pd.read_csv("Combined.csv", usecols = columns)
open = (Combined_df.close)[-480:]
close_dates = dates[-480:]

rmse = []
for mse in history.history['mse']:
    rmse.append(np.sqrt(mse))

plt.plot(rmse)
plt.show()

plt.plot(close_dates, open, color = "red")
plt.plot(prediction_dates, true_predictions, color = "blue")

plt.show()
