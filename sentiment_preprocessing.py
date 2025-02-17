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


def clean_text(text):
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


def get_next_day_return(tweet_timestamp, stock, stocks_df):
    # market close = 4PM ET = 9PM UTC
    # reset time for that date and add 21 hours to get 9PM UTC
    market_close_utc = tweet_timestamp.normalize() + pd.Timedelta(hours = 21)

    # if tweet timestamp is before market close, use same day
    # if after, use next day
    if tweet_timestamp <= market_close_utc:
        tweet_date = tweet_timestamp.date()
    else:
        tweet_date = (tweet_timestamp + pd.Timedelta(days=1)).date()

    # filter stock data by stock so date query is more efficient
    stock_data = stocks_df[stocks_df['Stock Name'] == stock]

    # get closing prices for tweet date and day after tweet date
    tweet_day_close = stock_data.loc[stock_data['Date'] == tweet_date, 'Close']
    next_day_close = stock_data.loc[stock_data['Date'] == tweet_date + pd.Timedelta(days=1), 'Close']

    # if we don't have data for that stock and date, return none
    if tweet_day_close.empty or next_day_close.empty:
        return None

    # get value
    tweet_day_close = tweet_day_close.values[0]
    next_day_close = next_day_close.values[0]

    # compute return
    return (next_day_close - tweet_day_close) / tweet_day_close
