# trading-bot

This is a work in progress. The aim of the project is to assess to what extent the stock market and asset prices are predictable with an ML approach.
Specifically, an RNN is used for time-series analysis.

Currently the predictions are done on Bitcoin, because it is easy to find hourly Bitcoin price data dating as far back as 2018.

The hourly data is taken from Binance, and the FNG market sentiment figure is taken from alternative.me. Macroeconomic variables will be added to the time-series data to provide the model with a more general view of the stock market.
Technical indicators derived from OHLCV data is not used as this might lead to multi-collinearity - simply adding more noise to the data.

