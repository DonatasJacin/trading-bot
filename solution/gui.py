import tkinter as tk
import pandas as pd
import mplfinance as mpf
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
import solution

os.chdir("/home/donatas/Desktop/github-repos/asset-predictor/data/")

def FeaturesToRemove():
    removed_features = []
    feature_map = {"1": "open",
                    "2": "high",
                    "3": "low",
                    "4": "close",
                    "5": "volume",
                    "6": "fng",
                    "7": "CPI",
                    "8": "fundrate"}
    removed_feature = ""

    while removed_feature != "9":
        print("Select feature to remove")
        if "1" not in removed_features:
            print("1. Open")
        if "2" not in removed_features:    
            print("2. High")
        if "3" not in removed_features:
            print("3. Low")
        if "4" not in removed_features:
            print("4. Close")
        if "5" not in removed_features:
            print("5. Volume")
        if "6" not in removed_features:
            print("6. FNG (Fear and Greed Index)")
        if "7" not in removed_features:
            print("7. CPI-U (Consumer Inflation)")
        if "8" not in removed_features:
            print("8. FFR (Federal Funds Rate)")
        print("9. Keep all remaining features")
        removed_feature = input("Feature to remove (specify number): ")
        if int(removed_feature) < 1 and int(removed_feature) > 9:
            print("Invalid number selected") 
        if removed_feature != "9":
            removed_features.append(removed_feature)

    temp = []
    for feature in removed_features:
        temp.append(feature_map[feature])

    print(temp)

    return temp

def ChooseHyperparameters():
    hp_choices = {"batch_size" : 0,
                  "epochs" : 0,
                  "dropout" : 0,
                  "lstm_cells" : 0,
                  "days_to_use" : 0}
    batch_size = epochs = dropout = lstm_cells = days_to_use = -1
    while batch_size < 1:
        batch_size = int(input("Specify batch size: "))
    while epochs < 1:
        epochs = int(input("Specify epochs: "))
    while dropout < 0 or dropout > 1:
        dropout = float(input("Specify dropout rate: "))
    while lstm_cells < 1:
        lstm_cells = int(input("Specify LSTM cells: "))
    while days_to_use < 1:
        days_to_use = int(input("Specify number of past days to use in each training sample: "))

    hp_choices['batch_size'] = batch_size
    hp_choices['epochs'] = epochs
    hp_choices['dropout'] = dropout
    hp_choices['lstm_cells'] = lstm_cells
    hp_choices['days_to_use'] = days_to_use

    return hp_choices

def TrainNewModel():
    # Remove any features from the dataset?
    remove_features = input("Remove any features in the dataset? (y/n)")
    removed_features = []
    if remove_features == "y":
        removed_features = FeaturesToRemove()

    async_choice = iterations = display_choice = trading_choice = -1

    # Forward-fill, interpolation, or model fill to asynchronous data
    while int(async_choice) not in range(1, 4):
        async_choice = input("Perform interpolation (1), forward-fill (2), or model-fill (3) to asynchronous features?")

    if async_choice == "3":
        async_choice = input("Model fill with linear regression (4) or with random forest (5)")

    if async_choice == "1":
        async_choice = "interpolate"
    elif async_choice == "2":
        async_choice = "constant"
    elif async_choice == "4":
        async_choice = "lr"
    else:
        async_choice = "rf"

    # Specify the value of hyper-parameters
    hp_choices = ChooseHyperparameters()

    # Specify how many models we wish to train
    while iterations < 1:
        iterations = int(input("How many iterations should we train this model (how many models to train)?"))

    # Specify if we want to display graphs such as ROC, accuracy v.s. epochs, loss v.s epochs
    while display_choice != "y" and display_choice != "n":
        display_choice = input("Display training graphs at end of training? (y/n)")

    # Specify if we should perform a trading simulation to evalute the performance of the model in the real world
    while trading_choice != "y" and trading_choice != "n":
        trading_choice = input("Perform trading simulation at end of training? (y/n)")
    
    solution.TrainLSTM(removed_features, async_choice, hp_choices, iterations, display_choice, trading_choice)

def HyperparameterSearch():
    hp_ranges = {"batch_size": [],
                  "epochs": [],
                  "dropout": [],
                  "lstm_cells": [],
                  "days_to_use": []}
    
    default_choices = {"batch_size": [8],
                        "epochs": [50],
                        "dropout": [0.2],
                        "lstm_cells": [128],
                        "days_to_use": [14]}

    # Remove any features from the dataset?
    remove_features = input("Remove any features in the dataset? (y/n)")
    removed_features = []
    if remove_features == "y":
        removed_features = FeaturesToRemove()

    async_choice = iterations = display_choice = trading_choice = -1
    
    # Forward-fill, interpolation, or model fill to asynchronous data
    while int(async_choice) not in range(1, 4):
        async_choice = input("Perform interpolation (1), forward-fill (2), or model-fill (3) to asynchronous features?")

    if async_choice == "3":
        async_choice = input("Model fill with linear regression (4) or with random forest (5)")

    if async_choice == "1":
        async_choice = "interpolate"
    elif async_choice == "2":
        async_choice = "constant"
    elif async_choice == "4":
        async_choice = "lr"
    else:
        async_choice = "rf"

    # Specify how many models we wish to train
    while iterations < 1:
        iterations = int(input("How many iterations should we train each model (how many models to train for each combination)?"))

    # Specify if we want to display graphs such as ROC, accuracy v.s. epochs, loss v.s epochs
    while display_choice != "y" and display_choice != "n":
        display_choice = input("Display training graphs at end of training for each combination? (y/n)")

    # Specify if we should perform a trading simulation to evalute the performance of the model in the real world
    while trading_choice != "y" and trading_choice != "n":
        trading_choice = input("Perform trading simulation at end of training for each combination? (y/n)")

    chosen = []
    choice = ""
    while choice != "6":
        print("Which hyperparameters to search?")
        if "1" not in chosen:
            print("1. Batch size")
        if "2" not in chosen:
            print("2. Number of epochs")
        if "3" not in chosen:
            print("3. Dropout rate")
        if "4" not in chosen:
            print("4. Number of cells in CuDNNLSTM layers")
        if "5" not in chosen:
            print("5. Number of past days used in each training sample")
        print("6. Done")
        choice = input()
        chosen.append(choice)
        if choice == "1":
            batch_choice = 0
            while True:
                batch_choice = int(input("Enter a batch size to test (-1 to finish)"))
                if batch_choice == -1:
                    break
                if batch_choice > 0:
                    hp_ranges['batch_size'].append(batch_choice)
                else:
                    print("Batch size must be an integer larger than 0")
        elif choice == "2":
            epoch_choice = 0
            while True:
                epoch_choice = int(input("Enter an epoch count to test (-1 to finish)"))
                if epoch_choice == -1:
                    break
                if epoch_choice > 0:
                    hp_ranges['epochs'].append(epoch_choice)
                else:
                    print("Epoch count must be an integer larger than 0")
        elif choice == "3":
            dropout_choice = 0
            while True:
                dropout_choice = float(input("Enter a dropout rate to test (-1 to finish)"))
                if dropout_choice == -1.0:
                    break
                if dropout_choice >= 0 and dropout_choice <= 1:
                    hp_ranges['dropout'].append(dropout_choice)
                else:
                    print("Dropout rate must be larger or equal to 0 and less than or equal to 1")
        elif choice == "4":
            cell_choice = 0
            while True:
                cell_choice = int(input("Enter a number of CuDNNLSTM cells to test (-1 to finish)"))
                if cell_choice == -1:
                    break
                if cell_choice > 0:
                    hp_ranges['lstm_cells'].append(cell_choice)
                else:
                    print("Number of CuDNNLSTM cells must be an integer larger than 0")
        elif choice == "5":
            day_choice = 0
            while True:
                day_choice = int(input("Enter a number of past days used in each training sample to test (-1 to finish)"))
                if day_choice == -1:
                    break
                if day_choice > 0:
                    hp_ranges['days_to_use'].append(day_choice)
                else:
                    print("Number of days used in each training sample must be an integer larger than 0")

    for hp in hp_ranges:
        if len(hp_ranges[hp]) == 0:
            print("No choice given for hyperparameter: {}".format(hp))
            print("Using default value {}".format(str(default_choices[hp][0])))
            hp_ranges[hp] = default_choices[hp]

    display_choices = input("Display final hyperparameter values to be tested? (y/n)")
    if display_choices == "y":
        print(hp_ranges)

    solution.HyperparameterGridSearch(removed_features, hp_ranges, display_choice, trading_choice, iterations, async_choice)

choice = ""

while choice != "1":
    print("Select Option:")
    print("1. Launch BTC Dashboard GUI")
    print("2. Train a new model")
    print("3. Hyper-parameter grid search")
    print("4. Exit")
    choice = input("Input: ")
    if choice == "2":
        TrainNewModel()
        continue
    elif choice == "3":
        HyperparameterSearch()
        continue
    elif choice == "4":
        exit()

# GUI Display

# Load combined data from CSV file
mode = 'constant'
reload = 1
Combined_df = solution.GetDataset(reload, mode)

def ShowFigure(plot_config, Combined_df):
    global days_start
    global days_end
    global days_range

    print("Showing new figure")

    if days_end > len(Combined_df):
        days_end = len(Combined_df)
        days_start = days_end - days_range

    Combined_df_new = Combined_df[days_start:days_end]

    BTC_df = Combined_df_new[['open', 'high', 'low', 'close', 'volume']]

    apd = []
    panel_count = 1
    for indicator in plot_config:
        if indicator == "Volume":
            apd.append(mpf.make_addplot(Combined_df_new['volume'], panel = panel_count, ylabel = 'Volume'))
            panel_count += 1
        elif indicator == "CPI":
            apd.append(mpf.make_addplot(Combined_df_new['CPI'], panel = panel_count, ylabel = 'CPI-U'))
            panel_count += 1
        elif indicator == "FNG":
            apd.append(mpf.make_addplot(Combined_df_new['fng'], panel = panel_count, ylabel = 'FNG Index'))
            panel_count += 1
        elif indicator == "FED":
            apd.append(mpf.make_addplot(Combined_df_new['fundrate'], panel = panel_count, ylabel = 'Fund Rate'))
            panel_count += 1
        elif indicator == "Custom":
            Custom_df = pd.read_csv("BestAUCPredictions.csv")
            optimal_threshold = Custom_df.iloc[-1, 0]
            Custom_df = Custom_df.iloc[:-1, :]
            padding_length = len(Combined_df) - len(Custom_df)
            padding_df = pd.DataFrame({'prediction': [optimal_threshold]*padding_length})
            Custom_df = pd.concat([padding_df, Custom_df], ignore_index=True)
            Custom_df = Custom_df[days_start:days_end]
            Custom_df = Custom_df.sub(optimal_threshold)
            apd.append(mpf.make_addplot(Custom_df, panel = panel_count, ylabel = "LSTM Indicator"))
            panel_count += 1

    figure_BTC, axlist_BTC = mpf.plot(BTC_df, addplot=apd, type='candle', figratio=(1,1), figsize=(100,100), style='binance', returnfig=True, warn_too_much_data=5000000)
    canvas = FigureCanvasTkAgg(figure_BTC, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=1)


customtkinter.set_appearance_mode("dark")
window = customtkinter.CTk()

my_font = ('courier 10 pitch', 24)

global days_range
global days_start
global days_end

plot_config = []
days_range = 730
days_start = len(Combined_df) - days_range
days_end = days_start + days_range

Combined_df_first = Combined_df[days_start:days_end]
BTC_df = Combined_df_first[['open', 'high', 'low', 'close', 'volume']]
figure_BTC, axlist_BTC = mpf.plot(BTC_df, type='candle', figratio=(1,1), figsize=(100,100), style='binance', returnfig=True, warn_too_much_data=5000000)


def ForwardDays(plot_config, df, n_days):
    global days_start
    global days_end

    days_start -= n_days
    days_end -= n_days

    ShowFigure(plot_config, df)

def BackwardsDays(plot_config, df, n_days):
    global days_start
    global days_end

    days_start += n_days
    days_end += n_days

    ShowFigure(plot_config, df)

def RangeDays(plot_config, df):
    global days_start
    global days_end
    global days_range

    days_range = int(text_range.get("1.0", "end-1c"))
    
    if days_start - days_range < 0:
        days_end = days_start + days_range
    else:
        days_start = days_end - days_range

    ShowFigure(plot_config, df)

def ToggleIndicator(plot_config, df, indicator):
    if indicator in plot_config:
        plot_config.remove(indicator)
    else:
        plot_config.append(indicator)

    ShowFigure(plot_config, df)

window.title("Bitcoin Dashboard")

window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

frm_buttons = customtkinter.CTkFrame(window)

label_range = customtkinter.CTkLabel(frm_buttons, text="Days Range", font=my_font)
text_range = customtkinter.CTkTextbox(frm_buttons, height = 10)
btn_range = customtkinter.CTkButton(frm_buttons, text="Apply Range", command=lambda: RangeDays(plot_config, Combined_df), font=my_font)
btn_forw30 = customtkinter.CTkButton(frm_buttons, text="Back 30 Days", command=lambda: ForwardDays(plot_config, Combined_df, 30), font=my_font)
btn_back30 = customtkinter.CTkButton(frm_buttons, text="Forward 30 Days", command=lambda: BackwardsDays(plot_config, Combined_df, 30), font=my_font)
btn_volume = customtkinter.CTkButton(frm_buttons, text="Volume", command=lambda: ToggleIndicator(plot_config, Combined_df, "Volume"), font=my_font)
btn_fng = customtkinter.CTkButton(frm_buttons, text="Fear and Greed Index (FNG)", command=lambda: ToggleIndicator(plot_config, Combined_df, "FNG"), font=my_font)
btn_cpi = customtkinter.CTkButton(frm_buttons, text="CPI-U", command=lambda: ToggleIndicator(plot_config, Combined_df, "CPI"), font=my_font)
btn_fed = customtkinter.CTkButton(frm_buttons, text="Federal Funds Rate", command=lambda: ToggleIndicator(plot_config, Combined_df, "FED"), font=my_font)
btn_custom = customtkinter.CTkButton(frm_buttons, text="LSTM Indicator", command=lambda: ToggleIndicator(plot_config, Combined_df, "Custom"), font=my_font)

label_range.grid(row=0, column=0, sticky="ew", padx=30, pady=5)
text_range.grid(row=1, column=0, sticky="ew", padx=30, pady=5)
btn_range.grid(row=2, column=0, sticky="ew", padx=30, pady=15)
btn_forw30.grid(row=3, column=0, sticky="ew", padx=30, pady=15)
btn_back30.grid(row=4, column=0, sticky="ew", padx=30, pady=15)
btn_volume.grid(row=5, column=0, sticky="ew", padx=30, pady=15)
btn_fng.grid(row=6, column=0, sticky="ew", padx=30, pady=15)
btn_cpi.grid(row=7, column=0, sticky="ew", padx=30, pady=15)
btn_fed.grid(row=8, column=0, sticky="ew", padx=30, pady=15)
btn_custom.grid(row=9, column=0, sticky="ew", padx=30, pady=15)
frm_buttons.grid(row=0, column=0)


canvas = FigureCanvasTkAgg(figure_BTC, master=window)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=1)



window.mainloop()