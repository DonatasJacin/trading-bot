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
from keras.models import Sequential, Model
from keras.layers import CuDNNLSTM
from keras.layers import Dense, Dropout, Input, LSTM, Dense, concatenate, CuDNNLSTM
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Model
from keras.utils.vis_utils import plot_model
from keras import backend as K
from tabulate import tabulate
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import talib

os.chdir("/home/donatas/Desktop/github-repos/asset-predictor/data/")

def GetTechnicalIndicatorDataset():

    BTC_df, labels = CreateLabels()
    BTC_df = BTC_df.drop(['unix', 'symbol', 'volume BTC', 'tradecount'], axis = 1)
    BTC_df = BTC_df.rename(columns = {'volume USDT': 'volume', 'fng_value': 'fng', 'value': 'inflation'})
    BTC_df = BTC_df[::-1]
    labels = labels[::-1]

    rsi = talib.RSI(BTC_df['close'])
    macd, signal, _ = talib.MACD(BTC_df['close'])
    ma = talib.MA(BTC_df['close'])

    BTC_df['rsi'] = rsi
    BTC_df['macd'] = macd
    BTC_df['ma'] = ma
    BTC_df['label'] = labels

    cleaned_BTC_df = BTC_df.dropna()
    cleaned_BTC_df.set_index('date', inplace=True)

    cleaned_BTC_df.to_csv("TechnicalIndicatorDataset.csv")

    return cleaned_BTC_df

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
        raise ValueError("Invalid data mode '{}'".format(mode))

    FNG_df.to_csv("FNG_Daily.csv")
    return 1

# Upsamples monthly inflation data and interpolates, stores in "CPIU_Daily.csv"
def PrepareCPI(mode):
    # Import monthly inflation data
    CPIU_df = pd.read_csv("CPIU_Monthly.csv", parse_dates = ['date'])
    CPIU_df = CPIU_df.drop(['month'], axis = 1)
    CPIU_df = CPIU_df.set_index('date')
    if mode == "interpolate":
        CPIU_df = CPIU_df.resample('D').interpolate()
    elif mode == "constant":
        CPIU_df = CPIU_df.resample('D').ffill()
    else:
        raise ValueError("Invalid data mode '{}'".format(mode))

    CPIU_df.to_csv("CPIU_Daily.csv")
    return 1

#Upsamples daily federal fund rate data (not tracked on weekends) and interpolates, stores in "FEDFunds_Daily.csv"
def PrepareFed(mode):
    # Import federal fund rate data
    Fed_df = pd.read_csv("FEDFunds.csv", parse_dates = ['date'])
    Fed_df = Fed_df.drop(["rt","1","25","75","99","volume","tr","trt","idh","idl","std","sofr30","sofr90","sofr180","sofr","ri","fid"], axis = 1)
    Fed_df = Fed_df.reset_index(drop=True)
    Fed_df = Fed_df.set_index('date')
    if mode == "interpolate":
        Fed_df = Fed_df.resample('D').interpolate()
    elif mode == "constant":
        Fed_df = Fed_df.resample('D').ffill()
    else:
        return ValueError("Invalid data mode '{}'".format(mode))

    Fed_df.to_csv("FEDFunds_Daily.csv")
    return 1

def CreateLabels():
    BTC_df = pd.read_csv("BTC_Daily.csv") #From CryptoDataDownload.com
    BTC_df['date'] = pd.to_datetime(BTC_df['date'])

    labels = []
    for index, row in BTC_df.iterrows():
        label = 0
        if (row['close'] > row['open']):
            label = 1
        labels.append(label)

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
    CPIU_df['date'] = pd.to_datetime(CPIU_df['date'])

    # Merge datasets on date column
    print(BTC_df)
    Combined_df = pd.merge(BTC_df, FNG_df, on = 'date')
    Combined_df = pd.merge(Combined_df, Fed_df, on = 'date')
    Combined_df = pd.merge(Combined_df, CPIU_df, on = 'date')
    print(Combined_df)

    # Drop some unused columns and rename for convenience ---- Maybe keep day? Some evidence that fridays have more positive markets than monday --- For later, ask Yu
    Combined_df = Combined_df.drop(['unix', 'symbol', 'volume BTC', 'tradecount', 'fng_classification'], axis = 1)
    Combined_df = Combined_df.rename(columns = {'volume USDT': 'volume', 'fng_value': 'fng', 'value': 'inflation'})
    Combined_df['label'] = labels

    Combined_df = Combined_df[::-1]
    Combined_df.to_csv('Combined.csv', index=False)
    return 1

def GetDataset(reload, mode): #mode = constant, interpolate, or model fill
    # Check if datasets have been prepared, if not, run corresponding functions
    if mode == "rf" or mode == "lr":
        modelfill_mode = mode
        mode = "modelfill"
    if not exists("FNG_Daily.csv") or reload == 1:
        print("Loading FNG...")
        if mode == "modelfill":
            LoadFNG("interpolate")
        else:
            LoadFNG(mode)
        print("FNG loaded!")
    if not exists("CPIU_Daily.csv") or reload == 1:
        print("Loading CPI-U...")
        if mode == "modelfill":
            PrepareCPI("interpolate")
        else:
            PrepareCPI(mode)
        print("CPI-U loaded!")
    if not exists("FEDFunds_Daily.csv") or reload == 1:
        print("Loading FED...")
        if mode == "modelfill":
            PrepareFed("constant")
        else:
            PrepareFed(mode)
        print("FED loaded!")

    if mode == "modelfill":
        ModelFillCPI([modelfill_mode], False)
        Combined_df = pd.read_csv("ModelFillCPI.csv", index_col='date', parse_dates=True)
    else:
        print("Combining datasets...")
        CombineDatasets()
        print("Datasets combined!")
        Combined_df = pd.read_csv("Combined.csv", index_col='date', parse_dates=True)

    return Combined_df

from tabulate import tabulate
# Reshapes data into shape (n_samples * timesteps * n_features)
def ReshapeData(n_future, n_lookback, prediction_param_index, df):
    train_X, train_Y = [], []
    for i in range(n_lookback, len(df) - n_future + 1):
        train_X.append(df[i - n_lookback:i, 0:df.shape[1]])
        train_Y.append(df[i + n_future - 1:i + n_future, prediction_param_index])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    #table = tabulate(train_X[0], headers=['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ma', 'label'], tablefmt='fancy_grid')
    #print(table)

    return train_X, train_Y

def NormaliseData(df, df_test):
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)
    scaled_df = scaler.transform(df)
    scaled_df_test = scaler.transform(df_test)
    return scaled_df, scaled_df_test, scaler

def CreateLSTMModel(df, days_to_use, days_to_predict, ratio, prediction_param_index): #This function is actually unused in the final iteration of the project. This was here for a past experiment.
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

def TradingSimulation(test, prediction_array, optimal_threshold, use_threshold):

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
            #print(prediction_array[1][count])
            #print(prediction_array[0][count])
            #print(optimal_threshold)
            if (prediction_array[1][count] >= optimal_threshold and use_threshold == True) or (prediction_array[1][count] >= 0.5 and use_threshold == False):
                # Would have longed
                #print("Longing at index " + str(count))
                percent_change = i/test['close'].iloc[count-1]
                if percent_change >= 1:
                    correct += 1
                else:
                    incorrect += 1
                amplified_change = 1 + ((percent_change - 1) * leverage)
                #print("Last close: {:.3f}     Current Close: {:.3f}       % Change: {:.3f}".format(test['close'].iloc[count-1], i, amplified_change))
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
            #elif prediction_array[0][count] > optimal_threshold:
            elif (prediction_array[1][count] < optimal_threshold and use_threshold == True) or (prediction_array[1][count] < 0.5 and use_threshold == False):
                # Would have shorted
                #print("Shorting at index " + str(count))
                percent_change = test['close'].iloc[count-1]/i
                if percent_change >= 1:
                    correct += 1
                else:
                    incorrect += 1
                amplified_change = 1 + ((percent_change - 1) * leverage)
                #print("Last close: {:.3f}     Current Close: {:.3f}       % Change: {:.3f}".format(test['close'].iloc[count-1], i, amplified_change))
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
    return money_wfees_naive_array, correct, incorrect

def PrepareDataForModel(Combined_df, days_to_use, label_index, remove_labels, ratio_train_test):
    # Split data into training/validation and testing
    split_train_test = round(len(Combined_df) * ratio_train_test)
    train = Combined_df[:split_train_test]
    test = Combined_df[split_train_test:]
    test_labels = test['label']

    # Normalise dataset
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    days_to_predict = 1

    train_labels = []
    test_labels = []

    for i in range(days_to_use, len(train) - days_to_predict + 1):
        train_labels.append(train.iloc[i + days_to_predict - 1:i + days_to_predict, label_index])

    for i in range(0, len(test) - days_to_predict + 1):
        test_labels.append(test.iloc[i + days_to_predict - 1:i + days_to_predict, label_index])

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train_X, train_Y = ReshapeData(days_to_predict, days_to_use, label_index, scaled_train)
    #print("scaled_df shape == {}.".format(scaled_train.shape))
    #print("train_X shape == {}.".format(train_X.shape))
    #print("train_Y shape == {}.".format(train_Y.shape))
    #print("train_labels shape == {}.".format(train_labels.shape))
    #print("test_X shape == {}".format(scaled_test.shape))
    #print("test_labels shape == {}".format(test_labels.shape))

    if remove_labels == True:
        # Now remove labels from each day in each training sample
        train_X = train_X[:, :, :-1]

        scaled_train = scaled_train[:, :-1]
        scaled_test = scaled_test[:, :-1]

    return train_X, train_labels, test_labels, scaled_train, scaled_test, test

def TrainModel(train_X, train_labels, test_labels, scaled_train, scaled_test, hp_choices, iterations):

    # Unpack hyperparameters
    batch_size = hp_choices['batch_size']
    epochs = hp_choices['epochs']
    dropout_rate = hp_choices['dropout']
    LSTM_cells = hp_choices['lstm_cells']

    n_size = train_X.shape[1]
    n_features = train_X.shape[2]

    auc_sum = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    accuracies = []
    val_accuracies = []
    best_auc = 0
    losses = []
    val_losses = []
    combined_prediction_array = []
    best_val_index = 0
    optimal_thresholds = []
    optimal_thresholds_test = []
    threshold_magnitude_diff_sum = 0
    best_auc_predictions_positive = []
    best_auc_optimal_threshold = 0

    for i in range(iterations):
        validation_split = 0.1
        model = Sequential()
        model.add(CuDNNLSTM(LSTM_cells, input_shape=(n_size, n_features), return_sequences=True)) # Can only use CuDNNLSTMs if GPU is enabled for tensorflow, default activation is tanh
        model.add(Dropout(dropout_rate))
        model.add(CuDNNLSTM(LSTM_cells, input_shape=(n_size, n_features), return_sequences=False, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(2, activation = 'softmax'))
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(train_X, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

        validation_ratio = round(len(train_X) * (1-validation_split))
        validation_X = train_X[validation_ratio:]
        validation_labels = train_labels[validation_ratio:]


        validation_predictions = model.predict(validation_X)

        # only contains probabilities for positive class
        validation_predictions_positive = []
        validation_predictions_negative = []
        for prediction in validation_predictions:
            validation_predictions_negative.append(prediction[0])
            validation_predictions_positive.append(prediction[1])

        fpr, tpr, thresholds = roc_curve(validation_labels, validation_predictions_positive)
        roc_auc_validation = auc(fpr, tpr)

        # Calculate optimal threshold based on the ROC curve for validation set

        optimal_i = np.argmax(tpr - fpr)
        optimal_th = thresholds[optimal_i]
        optimal_thresholds.append(optimal_th)

        # Obtain predictions for test set

        test_predictions = []
        first_eval_batch = scaled_train[-n_size:]
        current_batch = first_eval_batch.reshape((1, n_size, n_features))
        label_counter = 0
        for x in scaled_test:
            current_pred = model.predict(current_batch)[0]
            test_predictions.append(current_pred)
            current_actual = np.array(x)
            current_batch = np.append(current_batch[:,1:,:], [[current_actual]], axis=1)

        # only contains probabilities for positive class
        
        test_predictions_positive = []
        test_predictions_negative = []
        for prediction in test_predictions:
            test_predictions_negative.append(prediction[0])
            test_predictions_positive.append(prediction[1])
        combined_prediction_array.append([test_predictions_negative, test_predictions_positive])

        if roc_auc_validation > best_auc:
            best_auc = roc_auc_validation
            best_val_index = i
            best_auc_predictions_positive = test_predictions_positive
            best_auc_optimal_threshold = optimal_th

        fpr_final, tpr_final, thresholds_test = roc_curve(test_labels, test_predictions_positive)
        roc_auc = auc(fpr_final, tpr_final)

        # Calculate optimal threshold for test set to compare with optimal threshold from validation set
        # If they are close, then this technique is sound

        optimal_i_test = np.argmax(tpr_final - fpr_final)
        optimal_th_test = thresholds_test[optimal_i_test]
        optimal_thresholds_test.append(optimal_th_test)

        # Compare thresholds

        threshold_magnitude_diff_sum += abs(optimal_th_test - optimal_th)

        auc_sum += roc_auc
        accuracies.append(history.history['accuracy'])
        val_accuracies.append(history.history['val_accuracy'])
        losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

        # Apply optimal threshold from validation set onto the test set

        binary_test_predictions_positive = []

        for prediction in test_predictions_positive:
            if prediction > optimal_th:
                binary_test_predictions_positive.append(1)
            else:
                binary_test_predictions_positive.append(0)

        # Calculate precision, recall, and F1 score
        precision_sum += precision_score(test_labels, binary_test_predictions_positive)
        recall_sum += recall_score(test_labels, binary_test_predictions_positive)
        f1_sum += f1_score(test_labels, binary_test_predictions_positive)

        #plt.plot(fpr, tpr, ':', label = 'Validation Model (area = {:.3f}), Optimal Threshold {:.3f}'.format(roc_auc_validation, optimal_th))
        plt.plot(fpr_final, tpr_final, label = "Test Model (area = {:.3f}), Optimal Threshold {:.3f}".format(roc_auc, optimal_th_test))

    del model

    K.clear_session()

    average_auc = auc_sum / iterations
    precision = precision_sum / iterations
    recall = recall_sum / iterations
    f1 = f1_sum / iterations
    threshold_magnitude_diff_avg = threshold_magnitude_diff_sum / iterations

    print("The average magnitude of the difference between validation and test thresholds: {:.3f}".format(threshold_magnitude_diff_avg))

    # Write the positive softmax predictions from the model with the highest validation accuracy to csv file
    best_auc_predictions_positive.append(best_auc_optimal_threshold)
    prediction_df = pd.DataFrame(best_auc_predictions_positive, columns = ['prediction'])
    prediction_df.to_csv('BestAUCPredictions.csv', index=False, header=True)

    return average_auc, best_auc, combined_prediction_array, accuracies, val_accuracies, losses, val_losses, optimal_thresholds, precision, recall, f1, best_val_index


def PlotModelIterations(average_auc, best_auc, accuracies, val_accuracies, losses, val_losses, epochs):
    print("Best AUC: {:.3f}".format(best_auc))

    plt.plot([0,1], [0,1], 'k--', label = "No predictive power (area = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve     average AUC = {:.3f}'.format(average_auc))
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

    return 1

def RunSimulations(combined_prediction_array, optimal_thresholds, iterations, test, best_val_index, display_choice):
    total_return = 0
    lost_count = 0
    running_accuracy = 0
    running_accuracy_noth = 0
    total_return_noth = 0
    best_return = 0
    best_accuracy = 0
    worst_return = 100000000000000
    worst_accuracy = 100

    for i in range(iterations):
        money_wfees_naive_array, correct, incorrect = TradingSimulation(test, combined_prediction_array[i], optimal_thresholds[i], True)
        test_accuracy = correct / (correct + incorrect)
        running_accuracy += test_accuracy

        if test_accuracy < worst_accuracy:
            worst_accuracy = test_accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        print("Accuracy on unseen testing data: {:.3f}%".format(test_accuracy*100))
        current_return = money_wfees_naive_array[-1]
        total_return += current_return

        if current_return < 1000:
            lost_count += 1
        if current_return < worst_return:
            worst_return = current_return
        if current_return > best_return:
            best_return = current_return

        if display_choice == "y":
            if best_val_index == i:
                plt.plot(money_wfees_naive_array, label = "Best validation run /w fees, accuracy {:.3f} using optimal thresholds from validation set".format(test_accuracy))
            else:
                plt.plot(money_wfees_naive_array)

        # Now simulate without using non-optimal threshold
        money_wfees_naive_array_noth, correct_noth, incorrect_noth = TradingSimulation(test, combined_prediction_array[i], optimal_thresholds[i], False)

        test_accuracy_noth = correct_noth / (correct_noth + incorrect_noth)
        running_accuracy_noth += test_accuracy_noth
        current_return_noth = money_wfees_naive_array_noth[-1]
        total_return_noth += current_return_noth

        if display_choice == "y":
            if best_val_index == i:
                plt.plot(money_wfees_naive_array_noth, ':', label = "Best validation run /w fees, accuracy {:.3f}".format(test_accuracy_noth))
            else:
                plt.plot(money_wfees_naive_array_noth, ':')

            plt.xlabel("Day")
            plt.ylabel("Equity")

    average_return = total_return / iterations 
    average_return_noth = total_return_noth / iterations
    average_accuracy = running_accuracy / iterations
    average_accuracy_noth = running_accuracy_noth / iterations

    if display_choice == "y":
        plt.legend()
        plt.show()

    # This is purely done for readability - no need to return 9 values. We can unpack later when needed
    outcome = {"average_return": average_return,
               "average_return_noth": average_return_noth,
               "average_accuracy": average_accuracy,
               "average_accuracy_noth": average_accuracy_noth,
               "lost_count": lost_count,
               "best_return": best_return,
               "worst_return": worst_return,
               "best_accuracy": best_accuracy,
               "worst_accuracy": worst_accuracy}

    return outcome

def TrainLSTM(removed_features, async_mode, hp_choices, iterations, display_choice, trading_choice):

    # Unpack hyperparameters
    batch_size = hp_choices['batch_size']
    epochs = hp_choices['epochs']
    dropout = hp_choices['dropout']
    lstm_cells = hp_choices['lstm_cells']
    days_to_use = hp_choices['days_to_use']

    reload = 1

    Combined_df = GetDataset(reload, async_mode)

    # Drop specified features if needed (used for testing)
    Combined_df = Combined_df.drop(removed_features, axis = 1)

    train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df=Combined_df, days_to_use=days_to_use, label_index=8 - len(removed_features), remove_labels=False, ratio_train_test=0.9)

    average_auc, best_auc, combined_prediction_array, accuracies, val_accuracies, losses, val_losses, optimal_thresholds, precision, recall, f1, best_val_index = TrainModel(train_X=train_X,
                                                                                                                        train_labels=train_labels,
                                                                                                                        test_labels=test_labels,
                                                                                                                        scaled_train=scaled_train,
                                                                                                                        scaled_test=scaled_test,
                                                                                                                        hp_choices=hp_choices,
                                                                                                                        iterations=iterations)
                                                                                                                        
    if display_choice == "y":
        PlotModelIterations(average_auc=average_auc, best_auc=best_auc, accuracies=accuracies, val_accuracies=val_accuracies, losses=losses, val_losses=val_losses, epochs=epochs)

    outcome = {"average_return": "N/A",
               "average_return_noth": "N/A",
               "average_accuracy": "N/A",
               "average_accuracy_noth": "N/A",
               "lost_count": "N/A",
               "best_return": "N/A",
               "worst_return": "N/A",
               "best_accuracy": "N/A",
               "worst_accuracy": "N/A"}

    if trading_choice == "y":
        outcome = RunSimulations(combined_prediction_array=combined_prediction_array,
                                optimal_thresholds=optimal_thresholds,
                                iterations=iterations,
                                best_val_index=best_val_index,
                                display_choice=display_choice,
                                test=test)

        print("Average return {:.3f}".format(outcome['average_return']))
        print("Average accuracy on unseen testing data: {:.3f}%".format(outcome['average_accuracy']*100))
        print("Runs that made negative ROI {}".format(outcome['lost_count']))
        print("Average precision/recall/f1 score {}/{}/{}".format(precision, recall, f1))
        print("Average accuracy using 0.5 threshold {}".format(outcome['average_accuracy_noth']))
        print("Average return using 0.5 threshold {:.3f}".format(outcome['average_return_noth']))

    combination_metrics = pd.DataFrame({"Average AUC": average_auc, "Best AUC": best_auc, "Average Return": outcome['average_return'], "Average Accuracy": outcome['average_accuracy'], "Lost Money Count": outcome['lost_count'], "Best Return": outcome['best_return'], "Worst Return": outcome['worst_return'], "Best Accuracy": outcome['best_accuracy'], "Worst Accuracy": outcome['worst_accuracy'], "Average Return NOTH": outcome['average_return_noth'], "Average Accuracy NOTH": outcome['average_accuracy_noth'], "Average Precision": precision, "Average Recall": recall, "Average F1": f1}, index=[0])
    file_name = "{},{},{},{},{}.csv".format(str(batch_size),
                                        str(epochs),
                                        str(dropout),
                                        str(lstm_cells),
                                        str(days_to_use))
    combination_metrics.to_csv(file_name)

    return 1

def HyperparameterGridSearch(r_features, hp_ranges, display_choice, trading_choice, iterations, mode):

    # Get our combined features dataset, specify whether to interpolate, forward-fill (constant), or model fill
    reload = 1
    Combined_df = GetDataset(reload, mode)

    # Drop specified features if needed (used for testing)
    Combined_df = Combined_df.drop(r_features, axis = 1)

    days_to_use_array = hp_ranges['days_to_use']
    LSTM_cells_array = hp_ranges['lstm_cells']
    dropout_array = hp_ranges['dropout']
    epochs_array = hp_ranges['epochs']
    batch_size_array = hp_ranges['batch_size']

    hp_combinations = list(itertools.product(batch_size_array, epochs_array, dropout_array, LSTM_cells_array, days_to_use_array))

    hp_combinations_df = pd.DataFrame(hp_combinations, columns = ["Batch Size", "Epochs", "Dropout Rate", "LSTM Cells", "Days to Use"])
    hp_combinations_df.to_csv("hp_combinations.csv")

    # Print the number of hyperparameter combinations
    print(f"Total number of hyperparameter combinations: {len(hp_combinations)}")

    # Print the first 10 hyperparameter combinations
    print("First 10 hyperparameter combinations:")
    for combination in hp_combinations[:10]:
        print(combination)

    combination_metrics = pd.DataFrame(columns=["Average AUC", "Best AUC", "Average Return", "Average Accuracy", "Lost Money Count", "Best Return", "Worst Return", "Best Accuracy", "Worst Accuracy", "Return Range", "Accuracy Range"])

    combination_counter = 0

    for combination in hp_combinations:

        # TrainModel takes a dictionary of hyperparameters, so we create this now
        combination_dict = {"batch_size": combination[0],
                            "epochs": combination[1],
                            "dropout": combination[2],
                            "lstm_cells": combination[3]}

        print("--------------------------")
        print("{} COMBINATION OUT OF {}".format(combination_counter, len(hp_combinations)))
        print("--------------------------")
        print("COMBINATION DETAILS:")
        print("--------------------------")
        print(combination)
        print("--------------------------")

        train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df=Combined_df, days_to_use=combination[4], label_index=8 - len(r_features), remove_labels=False, ratio_train_test=0.9)

        average_auc, best_auc, combined_prediction_array, accuracies, val_accuracies, losses, val_losses, optimal_thresholds, precision, recall, f1, best_val_index = TrainModel(train_X=train_X,
                                                                                                                            train_labels=train_labels,
                                                                                                                            test_labels=test_labels,
                                                                                                                            scaled_train=scaled_train,
                                                                                                                            scaled_test=scaled_test,
                                                                                                                            hp_choices=combination_dict,
                                                                                                                            iterations=iterations)
                                                                                                                            
        if display_choice == "y":
            PlotModelIterations(average_auc=average_auc, best_auc=best_auc, accuracies=accuracies, val_accuracies=val_accuracies, losses=losses, val_losses=val_losses, epochs=combination[1])

        outcome = {"average_return": "N/A",
                "average_return_noth": "N/A",
                "average_accuracy": "N/A",
                "average_accuracy_noth": "N/A",
                "lost_count": "N/A",
                "best_return": "N/A",
                "worst_return": "N/A",
                "best_accuracy": "N/A",
                "worst_accuracy": "N/A"}

        if trading_choice == "y":
            outcome = RunSimulations(combined_prediction_array=combined_prediction_array,
                                    optimal_thresholds=optimal_thresholds,
                                    iterations=iterations,
                                    best_val_index=best_val_index,
                                    display_choice=display_choice,
                                    test=test)

            print("Average return {:.3f}".format(outcome['average_return']))
            print("Average accuracy on unseen testing data: {:.3f}%".format(outcome['average_accuracy']*100))
            print("Runs that made negative ROI {}".format(outcome['lost_count']))
            print("Average precision/recall/f1 score {}/{}/{}".format(precision, recall, f1))
            print("Average accuracy using 0.5 threshold {}".format(outcome['average_accuracy_noth']))
            print("Average return using 0.5 threshold {:.3f}".format(outcome['average_return_noth']))

        combination_metrics = combination_metrics.append({"Average AUC": average_auc, "Best AUC": best_auc, "Average Return": outcome['average_return'], "Average Accuracy": outcome['average_accuracy'], "Lost Money Count": outcome['lost_count'], "Best Return": outcome['best_return'], "Worst Return": outcome['worst_return'], "Best Accuracy": outcome['best_accuracy'], "Worst Accuracy": outcome['worst_accuracy'], "Average Return NOTH": outcome['average_return_noth'], "Average Accuracy NOTH": outcome['average_accuracy_noth'], "Average Precision": precision, "Average Recall": recall, "Average F1": f1}, ignore_index=True)
        file_name = "Hyperparameter Grid Search Metrics.csv"
        combination_metrics.to_csv(file_name)
        combination_counter += 1
    
    return 1, combination_counter

#------------------- Non-interpolation/forwardfill

def LoadDailyData():
    # Import BTC OHLCV dataset, FNG dataset, 52-week new high/low ratio. Convert all date columns into the same datetime datatype
    BTC_df, labels = CreateLabels()

    FNG_df = pd.read_csv("FNG_Daily.csv") #From alternative.me
    FNG_df['date'] = pd.to_datetime(FNG_df['date'])

    # Merge datasets on date column
    Combined_df = pd.merge(BTC_df, FNG_df, on = 'date')

    # Drop some unused columns and rename for convenience ---- Maybe keep day? Some evidence that fridays have more positive markets than monday --- For later, ask Yu
    Combined_df = Combined_df.drop(['unix', 'symbol', 'volume BTC', 'tradecount', 'fng_classification'], axis = 1)
    Combined_df = Combined_df.rename(columns = {'volume USDT': 'volume'})
    Combined_df['label'] = labels

    Combined_df = Combined_df[::-1]
    Combined_df.to_csv('Combined_Daily_Only.csv', index=False)


def ModelFill(df, model_choice, display_choice):

    if model_choice == 'lr':
        model = LinearRegression()
    elif model_choice == 'rf':
        model = RandomForestRegressor()

    # Create a mask for NaN values in the 'CPI' column of the original dataset
    missing_values_mask = df['CPI'].isna()

    # Split the dataset into train and test sets based on the missing values mask
    train_set = df[~missing_values_mask]

    test_set = df[missing_values_mask]

    # Define the features and target columns
    features = ['open', 'high', 'low', 'close', 'volume', 'fng', 'label', 'fundrate']
    target = 'CPI'

    # Train the selected model
    print(df)
    model.fit(train_set[features], train_set[target])

    # Predict the CPI values for the test set
    predicted_CPI = model.predict(test_set[features])

    # Fill the missing values in the original dataset with the predicted CPI values
    df_filled = df
    df_filled.loc[missing_values_mask, 'CPI'] = predicted_CPI
    
    if display_choice:
        date = df['date']
        plt.plot(date, df_filled['CPI'], label = "Filled CPI with {}".format(model_choice))

        observed_indices = missing_values_mask[~missing_values_mask].index.intersection(df.index)
        plt.scatter(date[observed_indices], df.loc[observed_indices, 'CPI'], color='red', label='Observed CPI')

    columns = df_filled.columns.tolist()

    # Move the 'label' column to the last index
    columns.remove('label')
    columns.append('label')

    # Swap the positions of 'CPI' and 'label' columns
    columns[columns.index('CPI')], columns[columns.index('label')] = columns[columns.index('label')], columns[columns.index('CPI')]

    # Reorder the columns in the DataFrame
    df_filled = df_filled[columns]

    return df_filled

def ModelFillCPI(model_choices, display_choice):

    LoadDailyData()

    if display_choice:
        plt.figure(figsize=(10,6))

    for model in model_choices:
        # Load the data
        df = pd.read_csv('Combined_Daily_Only.csv')
        df = df.rename(columns={'fng_value': 'fng'})
        df_cpi = pd.read_csv('CPIU_Monthly.csv')
        df_fed = pd.read_csv('FEDFunds.csv')
        df_fed = df_fed.drop(["rt","1","25","75","99","volume","tr","trt","idh","idl","std","sofr30","sofr90","sofr180","sofr","ri","fid"], axis = 1)

        # Convert the date columns to datetime format
        df['date'] = pd.to_datetime(df['date'])
        df_cpi['date'] = pd.to_datetime(df_cpi['date'])
        df_fed['date'] = pd.to_datetime(df_fed['date'])

        # Merge the CPI and FEDFunds data with the daily Bitcoin data
        df = pd.merge(df, df_cpi, on='date', how='left')
        df = pd.merge(df, df_fed, on='date', how='left')
        df['fundrate'] = df['fundrate'].interpolate()
        df = df.drop('month', axis = 1)
        if model == "lr":
            df_lr = ModelFill(df, "lr", display_choice)
            df_lr.set_index('date', inplace=True)
        elif model == "rf":
            df_rf = ModelFill(df, "rf", display_choice)
            df_rf.set_index('date', inplace=True)

    if display_choice:
        plt.xlabel("Date")
        plt.ylabel("CPI")
        plt.title("CPI Values (Before and After Filling Missing Values)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # By default uses random forest

    if model_choices == ['lr']:
        df_lr.to_csv("ModelFillCPI.csv")
        return df_lr
    
    df_rf.to_csv("ModelFillCPI.csv")
    return df_rf

#------------------- Random Forest Classifier

def TrainRandomForest(removed_features, async_mode, trading_choice, display_choice, dataset_choice, hp_choices, hp_testing):
    reload = 1

    tree_count = hp_choices[0]
    tree_depth = hp_choices[1]
    feature_count = hp_choices[2]

    if dataset_choice == "technical":
        Combined_df = GetTechnicalIndicatorDataset()
    else:
        Combined_df = GetDataset(reload, async_mode)

    print(Combined_df)

    # Drop specified features if needed (used for testing)
    Combined_df = Combined_df.drop(removed_features, axis = 1)
    days_to_use = 1 # Random Forest uses just one past day for predicting
    train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df=Combined_df, days_to_use=days_to_use, label_index=8 - len(removed_features), remove_labels=False, ratio_train_test=0.9)

    # PrepareDataForModel is designed to work with the LSTM model
    # as a result, we need to tweak the datasets returned from PrepareDataForModel to make them suitable
    # for RandomForest.

    # Shift labels
    new_labels = []
    for i in range(0, len(test_labels) - 1):
        new_labels.append(test_labels[i + 1])
    test_labels = new_labels

    # Remove final training and testing sample
    scaled_train = scaled_train[:-1]
    scaled_test = scaled_test[:-1]


    train_labels = train_labels.flatten()

    # Initialise and train Random Forest
    model = RandomForestClassifier(n_estimators=tree_count, max_depth=tree_depth, max_features=feature_count, random_state=100)

    model.fit(scaled_train, train_labels)

    # Make predictions on test set - 0th element is negative class, 1st element is positive class
    test_predictions = model.predict_proba(scaled_test)

    test_predictions_positive = []
    test_predictions_negative = []

    for prediction in test_predictions:
        test_predictions_positive.append(prediction[1])
        test_predictions_negative.append(prediction[0])
    test_predictions = [test_predictions_negative, test_predictions_positive]

    
    test = test[1:]
    if trading_choice == "y":
        outcome = RunSimulations(combined_prediction_array=[test_predictions],
                                optimal_thresholds=[[0.5]],
                                iterations=1,
                                best_val_index=0,
                                display_choice=display_choice,
                                test=test)

    if display_choice == "y":
        # Get the feature importances
        importances = model.feature_importances_

        # Get the names of the features
        feature_names = Combined_df.columns

        # Sort the feature importances in descending order
        sorted_indices = importances.argsort()[::-1]
        sorted_importances = importances[sorted_indices]
        sorted_feature_names = feature_names[sorted_indices]

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feature_names)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Random Forest Feature Importance')
        plt.show()

    # flatten test_labels column
    test_labels = np.array(test_labels).flatten()

    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions_positive)
    roc_auc = auc(fpr, tpr)

    if display_choice == "y":
        plt.plot([0,1], [0,1], 'k--', label = "No predictive power (area = 0.5)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label = "Test Model (area = {:.3f})".format(roc_auc))
        plt.legend()
        plt.show()

    if hp_testing == "y":
        return outcome, fpr, tpr, roc_auc

    return 1

#------------------- Logistic Regression Classifier

def TrainLogisticRegression(removed_features, days_to_use, async_mode, trading_choice, display_choice, dataset_choice, hp_choices, hp_testing):
    reload = 1

    penalty = hp_choices[0]
    c = hp_choices[1]

    if dataset_choice == "technical":
        Combined_df = GetTechnicalIndicatorDataset()
    else:
        Combined_df = GetDataset(reload, async_mode)

    # Drop specified features if needed (used for testing)
    Combined_df = Combined_df.drop(removed_features, axis = 1)
    train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df=Combined_df, days_to_use=days_to_use, label_index=8 - len(removed_features), remove_labels=False, ratio_train_test=0.9)

    model = LogisticRegression(penalty=penalty, C=c, solver='liblinear')

    train_X_2D = train_X.reshape(len(train_X), days_to_use * (9 - len(removed_features)))
    model.fit(train_X_2D, train_labels)

    test_predictions = []
    first_eval_batch = scaled_train[-days_to_use:]
    current_batch = first_eval_batch.reshape((1, days_to_use * (9 - len(removed_features))))
    label_index = 0
    for x in scaled_test:
        # Make predictions on test set - 0th element is negative class, 1st element is positive class
        current_pred = model.predict_proba(current_batch)[0]
        label_index += 1
        test_predictions.append(current_pred)
        current_actual = np.array(x)
        current_batch = np.append(current_batch[:, 1*(9-len(removed_features)):], np.array([current_actual]), axis=1)

    test_predictions_positive = []
    test_predictions_negative = []

    for prediction in test_predictions:
        test_predictions_positive.append(prediction[1])
        test_predictions_negative.append(prediction[0])

    test_predictions = [test_predictions_negative, test_predictions_positive]

    if trading_choice == "y":
        outcome = RunSimulations(combined_prediction_array=[test_predictions],
                                optimal_thresholds=[[0.5]],
                                iterations=1,
                                best_val_index=0,
                                display_choice=display_choice,
                                test=test)

    # flatten test_labels column
    test_labels = np.array(test_labels).flatten()

    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions_positive)
    roc_auc = auc(fpr, tpr)

    if display_choice == "y":
        plt.plot([0,1], [0,1], 'k--', label = "No predictive power (area = 0.5)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label = "Test Model (area = {:.3f})".format(roc_auc))
        plt.legend()
        plt.show()

    if hp_testing == "y":
        return outcome, fpr, tpr, roc_auc

    return 1

#------------------- Hyperparameter tuning for LR and RF

def HyperparameterGridSearchRF():
    tree_count = [100, 200, 300, 400]
    tree_depth = [10, 20, 30, 40, 50, 60]
    feature_count = [2, 3, 4, 5, 6, 7, 8]
    dataset = ''
    removed_features = []

    hp_combinations = list(itertools.product(tree_count, tree_depth, feature_count))

    highest_accuracy = 0

    print("Total length of hp combinations {}".format(len(hp_combinations)))

    for combination in hp_combinations:
        outcome, tpr, fpr, roc_auc = TrainRandomForest(removed_features, 'rf', 'y', 'n', dataset, combination, 'y')
        if outcome['average_accuracy'] > highest_accuracy:
            highest_accuracy = outcome['average_accuracy']
            highest_combination = combination
            highest_roc = [tpr, fpr, roc_auc]

    print("The highest accuracy is {:.3f}%".format(highest_accuracy*100))
    print("The combination for this model is:")
    print(highest_combination)
    
    plt.plot([0,1], [0,1], 'k--', label = "No predictive power (area = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(highest_roc[0], highest_roc[1], label = "Test Model")
    plt.title('ROC Curve     AUC = {:.3f}'.format(highest_roc[2]))
    plt.legend()
    plt.show()

    return 1

def HyperparameterGridSearchLR():
    penalty = ['l1', 'l2']
    c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    days_to_use = [1, 7, 28, 45]
    removed_features = []
    dataset = 'technical'

    hp_combinations = list(itertools.product(penalty, c, days_to_use))

    print("Total length of hp combinations {}".format(len(hp_combinations)))

    highest_accuracy = 0

    for combination in hp_combinations:
        outcome, tpr, fpr, roc_auc = TrainLogisticRegression(removed_features, combination[2], 'constant', 'y', 'n', dataset, combination, 'y')
        if outcome['average_accuracy'] > highest_accuracy:
            highest_accuracy = outcome['average_accuracy']
            highest_combination = combination
            highest_roc = [tpr, fpr, roc_auc]

    print("The highest accuracy is {:.3f}%".format(highest_accuracy*100))
    print("The combination for this model is:")
    print(highest_combination)
    
    plt.plot([0,1], [0,1], 'k--', label = "No predictive power (area = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(highest_roc[0], highest_roc[1], label = "Test Model")
    plt.title('ROC Curve     AUC = {:.3f}'.format(highest_roc[2]))
    plt.legend()
    plt.show()

    return 1

if __name__ == "__main__":
    print("To create new LSTM models or test for hyperparameters, use 'python3 gui.py'")
    #TrainLogisticRegression([], 14, 'constant', 'y', 'y', 'technical', ['l1', 1], 'n')
    #TrainRandomForest([], 'rf', 'y', 'y', '', [100, 30, 4], 'n')
    #ModelFillCPI(['lr', 'rf'], True)
    #GetTechnicalIndicatorDataset()
    HyperparameterGridSearchRF()
    #HyperparameterGridSearchLR()

