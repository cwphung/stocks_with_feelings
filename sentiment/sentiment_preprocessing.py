import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim


def download_dataset_to_df(url: str, file_path: str) -> pd.DataFrame:
    """
    Downloads dataset from kaggle directly into a dataframe.
    """
    try:
        return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, url, file_path)
    except ValueError:
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, url, file_path,
            pandas_kwargs={'encoding': "ISO-8859-1"}
        )


def clean_text(text: str) -> str:
    # tweet specific cleaning
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # remove urls
    text = re.sub(r"@\w+", "", text)  # remove mentions

    # general cleaning
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra white space

    # remove stop words
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


def get_future_return(tweet_timestamp, stock, stocks_df, future_days=1):
    # market close = 4PM ET = 9PM UTC
    # reset time for that date and add 21 hours to get 9PM UTC
    market_close_utc = tweet_timestamp.normalize() + pd.Timedelta(hours = 21)

    # if tweet timestamp is before market close, use same day
    # if after, use next day
    if tweet_timestamp <= market_close_utc:
        tweet_date = tweet_timestamp.date()
    else:
        tweet_date = (tweet_timestamp + pd.Timedelta(days=1)).date()

    # filter stock data by stock then sort by date
    stock_data = stocks_df[stocks_df['Stock Name'] == stock]
    stock_data = stock_data.sort_values(by = 'Date')

    # get closing prices for tweet date or closest day after
    stock_day_data = stock_data.loc[stock_data['Date'] >= tweet_date]
    # if we don't have data for that stock and date, return none
    if stock_day_data.empty:
        return None
    stock_day_close = stock_day_data.iloc[0]['Adj Close']
    stock_day = stock_day_data.iloc[0]['Date']

    # set next day to day after stock date
    next_day = stock_day + pd.Timedelta(days=future_days)

    # get closing prices for next day or closest day after
    next_day_data = stock_data.loc[stock_data['Date'] >= next_day]
    # if we don't have data for that stock and date, return none
    if next_day_data.empty:
        return None
    next_day_close = next_day_data.iloc[0]['Adj Close']
    next_day = next_day_data.iloc[0]['Date']

    # compute return
    return (next_day_close - stock_day_close) / stock_day_close


def assign_labels(future_return, threshold=0.01):
    """
    Assign a label from [-1, 0, 1] based on the next day return. Returns are normalized based on the standard deviation
    of returns for that specific ticker, such that a return that is within (threshold) standard deviations of 0 will be
    labeled 0.
    """
    if future_return >= threshold:
        return 2
    if future_return <= -threshold:
        return 0
    return 1


def get_sd_of_returns(ticker: str, stocks_df: pd.DataFrame) -> float:
    df = stocks_df.loc[stocks_df['Stock Name'] == ticker]
    df['daily_return'] = df['Adj Close'].pct_change() # SettingWithCopyWarning irrelevant because we just discard df after
    return df['daily_return'].std()

def mean_predicted_label_by_day(predictions_df):
    # group predictions by date and compute the mean of predicted labels
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'], utc=True).dt.date
    mean_df = predictions_df.groupby('Date', as_index=False)['predicted_label'].mean()
    mean_df = mean_df[['Date', 'predicted_label']]

    return mean_df
