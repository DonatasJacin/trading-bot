import os.path
from os.path import exists
from solution.solution import LoadFNG, PrepareCPI, PrepareFed, CombineDatasets, GetDataset, ReshapeData, TrainLSTM, HyperparameterGridSearch, TrainRandomForest, TrainLogisticRegression, PrepareDataForModel, RunSimulations, PlotModelIterations, TrainModel, TradingSimulation, GetTechnicalIndicatorDataset, HyperparameterGridSearchLR, HyperparameterGridSearchRF
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np

os.chdir("/home/donatas/Desktop/github-repos/asset-predictor/data/")

# To run tests: python -m pytest in /solution directory

#------- Test Definitions

def test_LoadFNG():
    assert (LoadFNG("constant") == 1) and (LoadFNG("interpolate") == 1) and exists("FNG_Daily.csv")

def test_PrepareCPI():
    assert (PrepareCPI("constant") == 1) and (PrepareCPI("interpolate") == 1) and exists("CPIU_Daily.csv")

def test_PrepareFed():
    assert (PrepareFed("constant") == 1) and (PrepareFed("interpolate") == 1) and exists("FEDFunds_Daily.csv")

def test_CombineDatasets():
    assert (CombineDatasets() == 1) and exists("Combined.csv")

def test_GetDataset():
    reload = 1
    mode = "constant"
    assert (type(GetDataset(reload, mode)) == pd.DataFrame)

def test_ReshapeData():
    reload = 1
    mode = "constant"
    Combined_df = GetDataset(reload, mode)
    ratio = 0.95
    split = round(len(Combined_df) * ratio)
    train = Combined_df[:split]
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    scaled_train = scaler.transform(train)
    days_to_use = 30
    label_index = 8
    days_to_predict = 1
    train_X, train_Y = ReshapeData(days_to_predict, days_to_use, label_index, scaled_train)

    assert train_X.shape == (len(train)-days_to_use, days_to_use, label_index+1) and train_Y.shape == (len(train)-days_to_use, days_to_predict)

def test_TradingSimulation():

    reload = 1
    async_mode = "constant"
    days_to_use = 14
    removed_features = []
    Combined_df = GetDataset(reload, async_mode)

    train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df=Combined_df, days_to_use=days_to_use, label_index=8 - len(removed_features), remove_labels=False, ratio_train_test=0.95)

    positive_predictions = negative_predictions = []
    for i in range(len(test)):
        num = random.random()
        positive_predictions.append(num)
        negative_predictions.append(1-num)
    prediction_array = [negative_predictions, positive_predictions]

    optimal_threshold = 0.5
    use_threshold = True

    money_wfees_naive_array, correct, incorrect = TradingSimulation(test, prediction_array, optimal_threshold, use_threshold)

    # Perform assertions on the returned values
    assert isinstance(money_wfees_naive_array, list)
    assert isinstance(correct, int)
    assert isinstance(incorrect, int)

    # Check the length of the returned array
    assert len(money_wfees_naive_array) == len(test) - 1


def test_PrepareDataForModel():
    reload = 1
    async_mode = "constant"
    removed_features = []
    days_to_use = 14

    Combined_df = GetDataset(reload, async_mode)

    # Drop specified features if needed (used for testing)
    Combined_df = Combined_df.drop(removed_features, axis = 1)

    ratio = 0.9
    split = round(len(Combined_df)*ratio)
    label_index = 8
    remove_labels = True

    train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df, days_to_use, label_index, remove_labels, ratio_train_test=0.9)

    # Perform assertions on the returned values
    assert isinstance(train_X, np.ndarray)
    assert isinstance(train_labels, np.ndarray)
    assert isinstance(test_labels, np.ndarray)
    assert isinstance(scaled_train, np.ndarray)
    assert isinstance(scaled_test, np.ndarray)
    assert isinstance(test, pd.DataFrame)

    train_df = Combined_df[:split]
    test_df = Combined_df[split:]

    # Check the shapes of the returned arrays
    assert train_X.shape == (len(train_df)-days_to_use, days_to_use, label_index)
    assert train_labels.shape == (len(train_df)-days_to_use, 1)
    assert test_labels.shape == (len(test_df), 1)
    assert scaled_train.shape == (len(train_df), label_index)
    assert scaled_test.shape == (len(test_df), label_index)
    assert test.shape == (len(test_df), label_index+1)

def test_TrainModel():
    reload = 1
    async_mode = "constant"
    removed_features = []
    days_to_use = 14

    Combined_df = GetDataset(reload, async_mode)

    # Drop specified features if needed (used for testing)
    Combined_df = Combined_df.drop(removed_features, axis = 1)

    train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df=Combined_df, days_to_use=days_to_use, label_index=8 - len(removed_features), remove_labels=False, ratio_train_test=0.9)

    hp_choices = {
        'batch_size': 32,
        'epochs': 10,
        'dropout': 0.2,
        'lstm_cells': 64,
        'days_to_use': 14
    }
    iterations = 3

    # Run the function
    result = TrainModel(train_X, train_labels, test_labels, scaled_train, scaled_test, hp_choices, iterations)

    # Perform assertions or checks on the returned result
    average_auc, best_auc, combined_prediction_array, accuracies, val_accuracies, losses, val_losses, optimal_thresholds, precision, recall, f1, best_val_index = result

    assert isinstance(average_auc, float)
    assert isinstance(best_auc, float)
    assert isinstance(combined_prediction_array, list)
    assert isinstance(accuracies, list)
    assert isinstance(val_accuracies, list)
    assert isinstance(losses, list)
    assert isinstance(val_losses, list)
    assert isinstance(optimal_thresholds, list)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert isinstance(best_val_index, int)

def test_PlotModelIterations():
    average_auc = 0.58
    best_auc = 0.6
    accuracies = [[0.75, 0.8, 0.85], [0.8, 0.85, 0.9]]
    val_accuracies = [[0.7, 0.75, 0.8], [0.65, 0.7, 0.82]]
    losses = [[0.5, 0.4, 0.3], [0.5, 0.35, 0.25]]
    val_losses = [[0.6, 0.5, 0.4], [0.55, 0.5, 0.37]]
    epochs = 3

    result = PlotModelIterations(average_auc, best_auc, accuracies, val_accuracies, losses, val_losses, epochs)

    assert result == 1

def test_RunSimulations():
    positive_predictions = negative_predictions = []
    for i in range(50):
        num = random.random()
        positive_predictions.append(num)
        negative_predictions.append(1-num)
    combined_predictions = [negative_predictions, positive_predictions]
    optimal_thresholds = [random.random()]
    iterations = 1
    reload = 1
    mode = "constant"
    Combined_df = GetDataset(reload, mode)
    removed_features = []
    train_X, train_labels, test_labels, scaled_train, scaled_test, test = PrepareDataForModel(Combined_df=Combined_df, days_to_use=7, label_index=8 - len(removed_features), remove_labels=False, ratio_train_test=0.9)
    best_val_index = 1
    display_choice = 'n'

    outcome = RunSimulations([combined_predictions], optimal_thresholds, iterations, test, best_val_index, display_choice)
    
    assert isinstance(outcome['average_return'], (int, float))

def test_TrainLSTM():
    removed_features = ['fundrate']
    async_mode = 'constant'
    hp_choices = {
        'batch_size': 8,
        'epochs': 50,
        'dropout': 0.2,
        'lstm_cells': 64,
        'days_to_use': 7
    }

    iterations = 2
    display_choice = 'n'
    trading_choice = 'y'

    # Call the TrainLSTM function
    result = TrainLSTM(removed_features, async_mode, hp_choices, iterations, display_choice, trading_choice)
   
    assert result == 1

def test_HyperparameterGridSearch():
    removed_features = ['high']
    trading_choice = 'y'
    display_choice = 'n'
    iterations = 2
    mode = 'constant'

    hp_ranges = {
        'days_to_use': [7],
        'lstm_cells': [64, 128],
        'dropout': [0.2],
        'epochs': [50],
        'batch_size': [8]
    }

    result, combination_counter = HyperparameterGridSearch(removed_features, hp_ranges, display_choice, trading_choice, iterations, mode)
    expected_counter = len(hp_ranges['days_to_use']) * len(hp_ranges['lstm_cells']) * len(hp_ranges['dropout']) * len(hp_ranges['epochs']) * len(hp_ranges['batch_size'])

    assert (result == 1) and (combination_counter == expected_counter)

def test_TrainRandomForest():
    removed_features = []
    async_mode = 'constant'
    trading_choice = 'y'
    display_choice = 'n'
    dataset_choice = ''
    hp_choices = [100, 20, 3]
    hp_testing = 'n'

    result = TrainRandomForest(removed_features, async_mode, trading_choice, display_choice, dataset_choice, hp_choices, hp_testing)

    assert result == 1

    dataset_choice = 'technical'

    result = TrainRandomForest(removed_features, async_mode, trading_choice, display_choice, dataset_choice, hp_choices, hp_testing)
   
    assert result == 1


def test_TrainLogisticRegression():
    removed_features = []
    days_to_use = 7
    async_mode = 'constant'
    trading_choice = 'y'
    display_choice = 'n'
    dataset_choice = ''
    hp_choices = ['l1', 1]
    days_to_use = 14
    hp_testing = 'n'

    result = TrainLogisticRegression(removed_features, days_to_use, async_mode, trading_choice, display_choice, dataset_choice, hp_choices, hp_testing)

    assert result == 1

    dataset_choice = 'technical'

    result = TrainLogisticRegression(removed_features, days_to_use, async_mode, trading_choice, display_choice, dataset_choice, hp_choices, hp_testing)

    assert result == 1

def test_GetTechnicalIndicatorDataset():
    dataset = GetTechnicalIndicatorDataset()

    # Check dataset is a DataFrame and has rows
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0

    # Check dataset has the correct number of columns
    expected_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ma', 'label']
    assert list(dataset.columns) == expected_columns

    # Check dataset doesn't contain any NaN values
    assert not dataset.isnull().values.any()

    # Check dataset index is a DatetimeIndex
    assert isinstance(dataset.index, pd.DatetimeIndex)

    # Check dataset is saved to CSV
    dataset.to_csv("TechnicalIndicatorDataset.csv")
    csv_data = pd.read_csv("TechnicalIndicatorDataset.csv")
    assert isinstance(csv_data, pd.DataFrame)

def test_HyperparameterGridSearchRF():
    expected_result = 1  # Expected return value

    # Run the function
    result = HyperparameterGridSearchRF()

    # Check the return value
    assert result == expected_result

def test_HyperparameterGridSearchLR():
    expected_result = 1  # Expected return value

    # Run the function
    result = HyperparameterGridSearchLR()

    # Check the return value
    assert result == expected_result