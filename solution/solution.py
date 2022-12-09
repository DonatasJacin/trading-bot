import requests
import csv
from csv import DictReader
import pandas as pd
import numpy as np
import os.path
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Model
from keras.utils.vis_utils import plot_model
from tabulate import tabulate


os.chdir("/home/donatas/Desktop/github-repos/asset-predictor/data/")

# Loads up-to-date FNG figures into "FNG_Daily.csv"
def LoadFNG(mode):
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
        
    FNG_df = pd.read_csv(file_name, parse_dates = ['date'])
    FNG_df = FNG_df.set_index('date')

    if mode == "interpolate":
        FNG_df = FNG_df.resample('D').interpolate()
    elif mode == "constant":
        FNG_df = FNG_df.resample('D').ffill()
    else:
        return ValueError("invalid data mode")

    FNG_df.to_csv("FNG_Daily.csv")
    return 1

# Upsamples monthly inflation data and interpolates, stores in "CPIU_Daily.csv"
def PrepareCPI(mode):
    # Import monthly inflation data
    CPIU_df = pd.read_csv("CPIU_Monthly.csv", parse_dates = ['date_of_publication'])
    CPIU_df = CPIU_df.drop(['month'], axis = 1)
    CPIU_df = CPIU_df.set_index('date_of_publication')
    if mode == "interpolate":
        CPIU_df = CPIU_df.resample('D').interpolate()
    elif mode == "constant":
        CPIU_df = CPIU_df.resample('D').ffill()
    else:
        return ValueError("invalid data mode")

    CPIU_df.to_csv("CPIU_Daily.csv")
    return 1

#Upsamples daily federal fund rate data (not tracked on weekends) and interpolates, stores in "FEDFunds_Daily.csv"
def PrepareFed(mode):
    # Import federal fund rate data
    Fed_df = pd.read_csv("FEDFunds.csv", parse_dates = ['date'])
    Fed_df = Fed_df.reset_index(drop=True)
    Fed_df = Fed_df.set_index('date')
    if mode == "interpolate":
        Fed_df = Fed_df.resample('D').interpolate()
    elif mode == "constant":
        Fed_df = Fed_df.resample('D').ffill()
    else:
        return ValueError("invalid data mode")

    Fed_df.to_csv("FEDFunds_Daily.csv")
    return 1

def CreateLabels():
    BTC_df = pd.read_csv("BTC_Daily.csv") #From CryptoDataDownload.com
    BTC_df['date'] = pd.to_datetime(BTC_df['date'])

    labels = []
    for index, row in BTC_df.iterrows():
        if index + 1 == BTC_df.shape[0]:
            continue
        label = 0
        if (row['close'] > BTC_df.iloc[index + 1]['close']):
            label = 1
        labels.append(label)

    # For the very first datapoint
    labels.append(0)

    return BTC_df, labels

# Combines datasets for price data, FNG, and macroeconomic variables, stores in "Combined.csv"
def CombineDatasets():
    # Import BTC OHLCV dataset, FNG dataset, 52-week new high/low ratio. Convert all date columns into the same datetime datatype
    BTC_df, labels = CreateLabels()

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
    Combined_df['label'] = labels

    Combined_df = Combined_df[::-1]
    Combined_df.to_csv('Combined.csv', index=False)
    return 1

def GetDataset(reload):
    # Check if datasets have been prepared, if not, run corresponding functions
    mode = "interpolate"
    if not exists("FNG_Daily.csv") and reload == 1:
        LoadFNG(mode)
        reload = 1
    if not exists("CPIU_Daily.csv") and reload == 1:
        PrepareCPI(mode)
        reload = 1
    if not exists("FEDFunds_Daily.csv") and reload == 1:
        PrepareFed(mode)
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

def NormaliseData(df, df_test):
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)
    scaled_df = scaler.transform(df)
    scaled_df_test = scaler.transform(df_test)
    return scaled_df, scaled_df_test, scaler

def CreateLSTMModel(df, days_to_use, days_to_predict, ratio, prediction_param_index):
    # Split data into training/validation and testing
    split = round(len(df) * ratio)
    train = df[:split]
    test = df[split:]

    scaled_df, scaled_test_df, scaler = NormaliseData(train, test)

    train_x, train_y = ReshapeData(days_to_predict, days_to_use, prediction_param_index, scaled_df)
    train_x = tf.convert_to_tensor(train_x)
    train_y = tf.convert_to_tensor(train_y)

    n_size = train_x.shape[1]
    n_features = train_x.shape[2]

    input = keras.Input(shape = (n_size, n_features))
    lstm = layers.CuDNNLSTM(128)(input)
    model = Model(input, lstm)
    return model, scaled_df, scaled_test_df, scaler, train_x, train_y


reload = 1
Combined_df = GetDataset(reload)
# Dates are extracted for plotting purposes
dates = pd.to_datetime(Combined_df['date'])
Combined_df = Combined_df.drop(['date'], axis = 1)

# Split data into training/validation and testing
ratio = 0.95
split = round(len(Combined_df) * ratio)
train = Combined_df[:split]
test = Combined_df[split:]
test_labels = test['label']

# Normalise dataset
scaler = MinMaxScaler()
scaler = scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


days_to_predict = 1
days_to_use = 45

train_labels = []

for i in range(days_to_use, len(train) - days_to_predict + 1):
    train_labels.append(train.iloc[i + days_to_predict - 1:i + days_to_predict, 8])

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(train_labels)
print(test_labels)

train = train.drop(['label'], axis = 1)
test = test.drop(['label'], axis = 1)
train_X, train_Y = ReshapeData(days_to_predict, days_to_use, 3, scaled_train)
print("scaled_df shape == {}.".format(scaled_train.shape))
print("train_X shape == {}.".format(train_X.shape))
print("train_Y shape == {}.".format(train_Y.shape))
print("train_labels shape == {}.".format(train_labels.shape))

n_size = train_X.shape[1]
n_features = train_X.shape[2]

iterations = 5
auc_sum = 0
accuracies = []
val_accuracies = []
best_auc = 0
losses = []
val_losses = []

for i in range(iterations):
    model = Sequential()
    model.add(CuDNNLSTM(64, input_shape=(n_size, n_features), return_sequences=True)) # Can only use CuDNNLSTMs if GPU is enabled for tensorflow, default activation is tanh
    #model.add(Dense(train_Y.shape[1]))
    model.add(Dropout(0.4))
    model.add(CuDNNLSTM(64, input_shape=(n_size, n_features), return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation = 'softmax'))
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    #model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    epochs = 250
    history = model.fit(train_X, train_labels, epochs=epochs, batch_size=8, validation_split=0.1, verbose=1)

    test_predictions = []
    first_eval_batch = scaled_train[-n_size:]
    current_batch = first_eval_batch.reshape((1, n_size, n_features))

    for i in scaled_test:
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_actual = np.array(i)
        current_batch = np.append(current_batch[:,1:,:], [[current_actual]], axis=1)

    # only contains probabilities for positive class
    test_predictions_positive = []
    test_predictions_negative = []
    for prediction in test_predictions:
        test_predictions_negative.append(prediction[0])
        test_predictions_positive.append(prediction[1])


    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions_positive)
    roc_auc = auc(fpr, tpr)
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_test_predictions_positive = test_predictions_positive
        best_test_predictions_negative = test_predictions_negative
        #best_accuracy = history.history['accuracy']
        #best_accuracy_validation = history.history['val_accuracy']
        #best_loss = history.history['loss']
        #best_loss_validation = history.history['val_loss']
    auc_sum += roc_auc
    accuracies.append(history.history['accuracy'])
    val_accuracies.append(history.history['val_accuracy'])
    losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

    plt.plot(fpr, tpr, label = 'Model (area = {:.3f})'.format(roc_auc))

print("Best AUC: {:.3f}".format(best_auc))

plt.plot([0,1], [0,1], 'k--', label = "No predictive power (area = 0.5)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve     average AUC = {:.3f}'.format(auc_sum/iterations))
plt.legend(loc = 'best')
plt.show()

epoch_list = np.arange(0, epochs, 1, dtype=int)

count = 0
best_accuracy = []
for accuracy in accuracies:
    if accuracy == best_accuracy:
        plt.plot(epoch_list, accuracy, color = "black", label = "Highest AUC Accuracy")
        plt.plot(epoch_list, val_accuracies[count], color = "green", label = "Highest AUC Validation Accuracy")
    else:
        plt.plot(epoch_list, accuracy, color = "blue")
        plt.plot(epoch_list, val_accuracies[count], color = "red")
    count += 1

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training (blue) and Validation (red) accuracy")
plt.show()

count = 0
for loss in losses:
    plt.plot(epoch_list, loss, color = "blue")
    plt.plot(epoch_list, val_losses[count], color = "red")
    count += 1

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training (blue) and Validation (red) loss")
plt.show()

#test_predictions = []
#first_eval_batch = scaled_train[-n_size:]
#current_batch = first_eval_batch.reshape((1, n_size, n_features))

#for i in scaled_test:
#  current_pred = model.predict(current_batch)[0]
#  test_predictions.append(current_pred)
#  current_actual = np.array(i)
#  current_batch = np.append(current_batch[:,1:,:], [[current_actual]], axis=1)

#Perform inverse transformation to rescale back to original range
#Since we used 8 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 8 times and discard them after inverse transform

# prediction_copies = np.repeat(test_predictions, Combined_df.shape[1], axis=-1)
# true_predictions = scaler.inverse_transform(prediction_copies)[:,0]

# prediction_dates = dates[split:]

# columns = ['date','open','high','low','close','volume','fng','fundrate','inflation']
# Combined_df = pd.read_csv("Combined.csv", usecols = columns)
# open = (Combined_df.close)[-480:]
# close_dates = dates[-480:]

#rmse = []
#for mse in history.history['mse']:
#    rmse.append(np.sqrt(mse))

#plt.plot(epoch_list, history.history['mse'], label = "training mse", color = "red")
#plt.plot(epoch_list, history.history['val_mse'], label = "validation mse", color = "blue")
#plt.xlabel("Epoch")
#plt.ylabel("Mean Squared Error")
#plt.grid()
#plt.legend(loc="upper center")
#plt.show()

#plt.plot(close_dates, open, color = "red")
#plt.plot(prediction_dates, true_predictions, color = "blue")

#plt.show()


# Assume positive to be buy, negative to be sell:
#
# True positive would be predicted buy, actual buy
# False positive would be predicted buy, actual sell
# True negative would be predicted sell, actual sell
# False negative would be predicted sell, actual buy


# true_positives = 0
# true_negatives = 0
# false_positives = 0
# false_negatives = 0
# count = 0

# # Calculate true/false positive/negative metrics
# for close in test['close']:
#     if count == 0:
#         count += 1
#         continue
#     # Positive/Negative outcome, 1/0
#     outcome = 0
#     if (close > test['close'].iloc[count - 1]):
#         outcome = 1
#     elif (close < test['close'].iloc[count - 1]):
#         outcome = 0
#     else:
#         # Very rare
#         outcome = 1
#     if outcome == 1 and true_predictions[count] > test['close'].iloc[count - 1]:
#         # True positive
#         true_positives += 1
#     elif outcome == 1 and true_predictions[count] < test['close'].iloc[count - 1]:
#         # False negative
#         false_negatives += 1
#     elif outcome == 0 and true_predictions[count] > test['close'].iloc[count - 1]:
#         # False positive
#         false_positives += 1
#     elif outcome == 0 and true_predictions[count] < test['close'].iloc[count - 1]:
#         # True negative
#         true_negatives += 1
        
#     count += 1

# # TruePositiveRate = TruePositives / (TruePositives + False Negatives)
# true_positive_rate = true_positives / (true_positives + false_negatives)

# # FalsePositiveRate = FalsePositives / (FalsePositives + TrueNegatives)
# false_positive_rate = false_positives / (false_positives + true_negatives)

# true_negative_rate = true_negatives / (true_negatives + false_positives)

# false_negative_rate = false_negatives / (false_negatives + true_positives)
# print("True positive rate: " + str(true_positive_rate))
# print("False positive rate: " + str(false_positive_rate))
# print("True negative rate: " + str(true_negative_rate))
# print("False negative rate: " + str(false_negative_rate))

count = 0
cum_inaccuracy = 0
overestimations = 0
cum_overestimation_inaccuracy = 0
underestimations = 0
cum_underestimation_inaccuracy = 0
cum_prediction_change = 0
leverage = 1

money_wfees_naive = 1000
money_wfees_naive_array = []
money_naive = 1000
money_naive_array = []
money_wfees_smart = 1000
money_wfees_smart_array = []
money_smart = 1000
money_smart_array = []
correct = 0
incorrect = 0

cum_price_change = 0
trades_executed_naive = 0
difference_threshold = 0.005
taker_fee = 0.0005
maker_fee = 0.0005

#Convert strings from test into np.float64
test = test.astype(np.float64)

#Do simulation of trading
for i in test['close']:
  if count > 0:
    trades_executed_naive += 1
    # If last close is less than the prediction
    if best_test_predictions_positive[count] > 0.5:
      # Would have longed
      print("Longing at index " + str(count))
      percent_change = i/test['close'].iloc[count-1]
      if percent_change >= 1:
        correct += 1
      else:
        incorrect += 1
      amplified_change = 1 + ((percent_change - 1) * leverage)
      print("Last close: {:.3f}     Current Close: {:.3f}       % Change: {:.3f}".format(test['close'].iloc[count-1], i, amplified_change))
      # No mathematical need to seperate multiplication with brackets, but makes it more readable
      money_wfees_naive = money_wfees_naive - (money_wfees_naive * (taker_fee * leverage))
      money_wfees_naive = money_wfees_naive - (money_wfees_naive * (maker_fee * leverage))
      money_wfees_naive = money_wfees_naive * amplified_change
      money_naive = money_naive * amplified_change
      # Non-naive strategy - if the change is equal or greater than difference_threshold, long
    #   if test['close'].iloc[count-1] <= true_predictions[count]*(1-difference_threshold):
    #     money_wfees_smart = money_wfees_smart - (money_wfees_smart * (taker_fee * leverage))
    #     money_wfees_smart = money_wfees_smart - (money_wfees_smart * (maker_fee * leverage))
    #     money_wfees_smart = money_wfees_smart * amplified_change
    #     money_smart = money_smart * amplified_change
    # Else if last close is more than the prediction
    elif best_test_predictions_negative[count] > 0.5:
      # Would have shorted
      print("Shorting at index " + str(count))
      percent_change = test['close'].iloc[count-1]/i
      if percent_change >= 1:
        correct += 1
      else:
        incorrect += 1
      amplified_change = 1 + ((percent_change - 1) * leverage)
      print("Last close: {:.3f}     Current Close: {:.3f}       % Change: {:.3f}".format(test['close'].iloc[count-1], i, amplified_change))
      # No mathematical need to seperate multiplication with brackets, but makes it more readable
      money_wfees_naive = money_wfees_naive - (money_wfees_naive * (maker_fee * leverage))
      money_wfees_naive = money_wfees_naive - (money_wfees_naive * (taker_fee * leverage))
      money_wfees_naive = money_wfees_naive * amplified_change
      money_naive = money_naive * amplified_change
      # Non-naive strategy - if the change is equal or greater than difference_threshold, short
    #   if test['close'].iloc[count-1]*(1-difference_threshold) > true_predictions[count]: 
    #     money_wfees_smart = money_wfees_smart - (money_wfees_smart * (maker_fee * leverage))
    #     money_wfees_smart = money_wfees_smart - (money_wfees_smart * (taker_fee * leverage))
    #     money_wfees_smart = money_wfees_smart * amplified_change
    #     money_smart = money_smart * amplified_change
    money_wfees_naive_array.append(money_wfees_naive)
    money_naive_array.append(money_naive)
    # money_wfees_smart_array.append(money_wfees_smart)
    # money_smart_array.append(money_smart)
  count += 1

# prediction_change = cum_prediction_change / (count-1)
# price_change = cum_price_change / (count-1)
# inaccuracy = cum_inaccuracy / count
# if overestimations > 0:
#   overestimation_inaccuracy = cum_overestimation_inaccuracy / overestimations
# else:
#   overestimation_inaccuracy = 0
# if underestimations > 0:
#   underestimation_inaccuracy = cum_underestimation_inaccuracy / underestimations
# else:
#   underestimation_inaccuracy = 0

# print("Total inaccuracy: " + str(inaccuracy*100) + "%")
# print("Number of overesimations: " + str(overestimations))
# print("Total inaccuracy of overestimations: " + str(overestimation_inaccuracy*100) + "%")
# print("Number of underestimations: " + str(underestimations))
# print("Total inaccuracy of underestimations: " + str(underestimation_inaccuracy*100) + "%")
# print("Average predicted change: " + str(prediction_change))
# print("Average actual change: " + str(price_change))
# print("Executed " + str(trades_executed_naive) + " trades with maker fee (sell) of " + str(maker_fee) + " and taker fee (buy) of " + str(taker_fee))
# print("Money after trading including fees using naive strategy: " + str(money_wfees_naive))
# print("Money after trading not including fees using naive strategy: " + str(money_naive))
# print("Money after trading including fees using smart strategy: " + str(money_wfees_smart))
# print("Money after trading not including fees using smart strategy: " + str(money_smart))
# print("Elapsed days: " + str(count))

plt.plot(money_naive_array, label = "Without Fees")
plt.plot(money_wfees_naive_array, label = "With Fees")
plt.xlabel("Day")
plt.ylabel("Equity")
# plt.plot(money_smart_array, label = "smart no fees")
# plt.plot(money_wfees_smart_array, label = "smart fees")
plt.legend()
plt.show()


test_accuracy = correct / (correct + incorrect)
print("Accuracy on unseen testing data: {:.3f}".format(test_accuracy))

#days_to_predict = 1
#days_to_use = 30
#ratio = 0.98
# --------------- BTC

# Dates are extracted for plotting purposes
#BTC_df = pd.read_csv("BTC_Daily.csv") #From CryptoDataDownload.com
#BTC_df = BTC_df.iloc[::-1]
#dates = pd.to_datetime(BTC_df['date'])
#BTC_df = BTC_df.drop(['date', 'symbol', 'unix', 'volume BTC', 'tradecount'], axis = 1)
#split = round(len(BTC_df) * ratio)
#model_BTC, scaled_df_BTC, scaled_BTC_test_df, scaler_BTC, train_x_BTC, train_y_BTC = CreateLSTMModel(BTC_df, days_to_use, days_to_predict, ratio, 3)

# --------------- FNG

#FNG_df = pd.read_csv("FNG_Daily.csv")
#FNG_df = FNG_df.iloc[::-1]
#FNG_df = FNG_df.drop(['date', 'fng_classification'], axis = 1)

#model_FNG, scaled_df_FNG, scaled_df_test_FNG, scaler_FNG, train_x_FNG, train_y_FNG = CreateLSTMModel(FNG_df, days_to_use, days_to_predict, ratio, 0)

# --------------- CPI

#CPI_df = pd.read_csv("CPIU_Daily.csv")
#CPI_df = CPI_df.drop(['date_of_publication'], axis = 1)

#model_CPI, scaled_df_CPI, scaled_df_test_CPI, scaler_CPI, train_x_CPI, train_y_CPI = CreateLSTMModel(CPI_df, days_to_use, days_to_predict, ratio, 0)

# --------------- FED

#FED_df = pd.read_csv("FEDFunds_Daily.csv")
#FED_df = FED_df.drop(['date'], axis = 1)

#model_FED, scaled_df_FED, scaled_df_test_FED, scaler_FED, train_x_FED, train_Y_FED = CreateLSTMModel(FED_df, days_to_use, days_to_predict, ratio, 0)

# --------------- Combine univariate LSTMs

#combined = layers.concatenate([model_BTC.output, model_FNG.output, model_CPI.output, model_FED.output])
#model_combined = Dense(64, activation = "relu")(combined)

#model = Model(inputs = [model_BTC.input, model_FNG.input, model_CPI.input, model_FED.input], outputs = model_combined)

#keras.utils.plot_model(model, "model.png", show_shapes = True)

#model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#model.summary()

#history = model.fit([train_x_BTC, train_x_FNG, train_x_CPI, train_x_FED], train_y_BTC, epochs=1, batch_size=16, validation_split=0.1, verbose=1)

#test_predictions = []
#n_size = train_x_BTC.shape[1]
#n_features = train_x_BTC.shape[2]
#first_eval_batch_BTC = scaled_df_BTC[-n_size:]
#current_batch_BTC = first_eval_batch_BTC.reshape((1, n_size, n_features))

#n_size = train_x_FNG.shape[1]
#n_features = train_x_FNG.shape[2]
#first_eval_batch_FNG = scaled_df_FNG[-n_size:]
#current_batch_FNG = first_eval_batch_FNG.reshape((1, n_size, n_features))

#n_size = train_x_CPI.shape[1]
#n_features = train_x_CPI.shape[2]
#first_eval_batch_CPI = scaled_df_CPI[-n_size:]
#current_batch_CPI = first_eval_batch_CPI.reshape((1, n_size, n_features))

#n_size = train_x_FED.shape[1]
#n_features = train_x_FED.shape[2]
#first_eval_batch_FED = scaled_df_FED[-n_size:]
#current_batch_FED = first_eval_batch_FED.reshape((1, n_size, n_features))

#index = 1
#for i in scaled_BTC_test_df:
#  current_pred = model.predict([current_batch_BTC, current_batch_FNG, current_batch_CPI, current_batch_FED])[0]
#  print(current_pred)
#  test_predictions.append(current_pred)
#  current_actual_BTC = np.array(i)
#  current_actual_FNG = np.array(scaled_df_test_FNG[-n_size+index:])
#  current_actual_CPI = scaled_df_test_CPI[-n_size+index:]
#  current_actual_FED = scaled_df_test_FED[-n_size+index:]
#  current_batch_BTC = np.append(current_batch_BTC[:,1:,:], [[current_actual_BTC]], axis=1)
#  current_batch_FNG = np.append(current_batch_FNG[:,1:,:], [current_actual_FNG], axis=1)
#  current_batch_CPI = np.append(current_batch_CPI[:,1:,:], [current_actual_CPI], axis=1)
#  current_batch_FED = np.append(current_batch_FED[:,1:,:], [current_actual_FED], axis=1)
#  index += 1


#Perform inverse transformation to rescale back to original range
#Since we used 8 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 8 times and discard them after inverse transform
#prediction_copies = np.repeat(test_predictions, BTC_df.shape[1], axis=-1)
#true_predictions = scaler_BTC.inverse_transform(prediction_copies)[:,0]

#prediction_dates = dates[split:]
#print(prediction_dates.head())

#columns = ['date','open','high','low','close','volume USDT']
#BTC_df = pd.read_csv("BTC_Daily.csv", usecols = columns)
#BTC_df = BTC_df.iloc[::-1]
#open = (BTC_df.close).iloc[-480:]
#close_dates = dates.iloc[-480:]

#rmse = []
#for mse in history.history['mse']:
#    rmse.append(np.sqrt(mse))

#plt.plot(rmse)
#plt.show()

#plt.plot(close_dates, open, color = "red")
#plt.plot(prediction_dates, true_predictions, color = "blue")

#plt.show()


