# trading-bot

This is a work in progress. The aim of the project is to assess to what extent the stock market and asset prices are predictable with a ML approach.
Specifically, an RNN is used for time-series analysis.

Currently the predictions are done on Bitcoin, because it is easy to find hourly Bitcoin price data dating as far back as 2018.

The hourly data is taken from Binance, and the FNG market sentiment figure is taken from alternative.me.

The aim is to use as few external tools as possible, this means I am currently working on my own market sentiment analysis tool.
Needs to be switched from Bitcoin to the S&P500 (or some other well known stock/index), since the S&P500 has inherently much
less volatility and is more predictable.
Another model will be trained on daily price data (as opposed to hourly) to see the difference between the accuracy of predictions - does a larger time interval result in a more accurate or less accurate model?
