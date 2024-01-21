# Bitcoin price prediction with machine learning techniques and asynchronous data

The aim of the project is to assess to what extent the stock market and asset prices are predictable with an ML approach.

Predictions are done on Bitcoin, because it is easy to find hourly Bitcoin price data dating as far back as 2018, and investor sentiment figures.

The hourly data is taken from Binance, and the FNG market sentiment figure is taken from alternative.me.


How to use:

For most purposes, simply running gui.py will be sufficient. This allows the launching of the Bitcoin Dashboard GUI, training an LSTM model given some hyperparameter combination, displaying the trading simulation of said LSTM, and performing a hyperparameter grid search for LSTM given ranges of hyperparameters.

If you want to train a Random Forest model, or a Logistic Regression model, you will need to uncomment the function you would like to run in the if __name__ == "__main__" section at the solution.py file. Then run the solution.py file.

Please note that to run the LSTM code, you must have GPU acceleration enabled with tensorflow using the cuda toolkit. Without GPU acceleration and the usage of CuDNN LSTMs, training the model takes a long time.

Instructions on how to enable GPU acceleration:
https://www.nvidia.com/en-sg/data-center/gpu-accelerated-applications/tensorflow/
https://www.tensorflow.org/guide/gpu


To run the pytest unit tests, navigate to asset-predictor/ and run "python -m pytest"
