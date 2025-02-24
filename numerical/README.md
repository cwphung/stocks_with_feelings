# Numerical

This folder focuses on developing machine learning models for predicting stock prices.

## Content

- Stock datasets
- Data processing functions
- Notebooks for testing and development

## Description

This workspace is currently using the Kaggle datasets `stock_tweets.csv` and `stock_yfinance_data.csv`, which contain a year of data for multiple stocks in the S&P500. The file `stock_data_processing.py` contains functions for cleaning stock price data and calculating technical features, such as moving averages, RSI, MACD, and Bollinger bands. A notebook `test_preprocessing.ipynb` was created to test these functions and generate example figures. Future work will include testing RNN and attention models to predict future stock prices.

## Notes

Use a virtual environment with `requirements.txt`.