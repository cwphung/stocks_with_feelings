'''
COM SCI 247 (Winter 25)

Stock Data Processing Functions
'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as stm
from tqdm import tqdm


def clean_stock_data(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    '''
    ### Description
    Cleans and filters stock data for a given ticker symbol.
    
    ### Parameters
    - df (pd.DataFrame):
        The original stock dataset containing multiple stock entries.
    - ticker (str):
        Ticker symbol for a specific stock, if the dataset contains multiple stocks. 
        If None, no filtering is applied.

    ### Returns:
    - (pd.DataFrame):
        A cleaned DataFrame containing only the specified stock's data, 
        with formatted columns, sorted dates, and missing values handled.
    '''
    if ticker is not None:
        df = df[df['Stock Name'] == ticker]
        df = df.drop(columns=['Stock Name'])
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    pd.to_datetime(df['date'])
    df.sort_values(by=['date'], ignore_index=True, inplace=True)
    df.ffill(inplace=True)
    return df


def percent_price_change(df: pd.DataFrame) -> tuple:
    '''
    ### Description
    Calculates the percent change in price from the previous day closing price.
    
    ### Parameters
    - df (pd.DataFrame)

    ### Returns:
    - (pd.Series): Percent Open
    - (pd.Series): Percent High
    - (pd.Series): Percent Low
    - (pd.Series): Percent Close
    '''
    perc_open = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    perc_high = (df["high"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    perc_low = (df["low"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    perc_close = (df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    return perc_open, perc_high, perc_low, perc_close

def simple_moving_average(df: pd.DataFrame, days=20) -> pd.Series:
    '''
    ### Description
    Calculates the simple moving average of a stock based on the close price.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return df['close'].rolling(window=days, min_periods=1).mean()


def exponential_moving_average(df: pd.DataFrame, days=20) -> pd.Series:
    '''
    ### Description
    Calculates the exponential moving average of a stock based on the close price.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return df['close'].ewm(span=days, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, days=14) -> pd.Series:
    '''
    ### Description
    Calculates the relative strength index (RSI) of a stock based on the close price.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    
    delta = df['close'].diff(1)
    delta.loc[0] = 0
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=days, min_periods=1).mean()
    avg_loss = loss.rolling(window=days, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)

    return rsi


def calculate_macd(df: pd.DataFrame, short_window=12, long_window=26, signal_window=9) -> tuple:
    '''
    ### Description
    Calculates the moving average convergence divergence (MACD) of a stock based on the close price.
    
    ### Parameters
    - df (pd.DataFrame)
    - short_window (int)
    - long_window (int)
    - signal_window (int)

    ### Returns:
    - (pd.Series): MACD line
    - (pd.Series): MACD signal
    - (pd.Series): MACD histogram
    '''
    macd_line = df['close'].ewm(span=short_window, adjust=False).mean() - df['close'].ewm(span=long_window, adjust=False).mean()
    macd_signal = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram


def bollinger_bands(df: pd.DataFrame, days=20) -> tuple:
    '''
    ### Description
    Calculates the Bollinger Bands (BB) of a stock based on the close price.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series): Middle Bollinger Band
    - (pd.Series): Upper Bollinger Band
    - (pd.Series): Lower Bollinger Band
    '''
    bb_middle = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=days).std()
    bb_upper = bb_middle + (2 * std)
    bb_lower = bb_middle - (2 * std)
    return bb_middle, bb_upper, bb_lower


def relative_daily_range(df: pd.DataFrame, days=20):
    '''
    ### Description
    Calculates the relative daily range of a stock from the last number of days.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series): Relative daily rnage
    '''
    rolling_max_high = df['high'].rolling(window=days).max()
    rolling_min_low = df['low'].rolling(window=days).min()
    relative_range = (df['high'] - df['low']) / (rolling_max_high - rolling_min_low)
    return relative_range


def calculate_closing_diff(df: pd.DataFrame) -> tuple:
    '''
    ### Description
    DEPRICATED.
    Calculates the change in closing price between days.
    
    ### Parameters
    - df (pd.DataFrame)

    ### Returns:
    - (pd.Series)
    '''
    delta = df['close'].diff(1)
    delta.loc[0] = 0

    return pd.Series(delta)


def calculate_future_close(df: pd.DataFrame) -> tuple:
    '''
    ### Description
    Calculates the next day percent change in closing price.
    
    ### Parameters
    - df (pd.DataFrame)

    ### Returns:
    - (pd.Series)
    '''
    future_close = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    return future_close


def calculate_moving_normalized(df: pd.DataFrame, feat: str, days=30) -> pd.Series:
    '''
    ### Description
    DEPRICATED.
    Calculates normalized values of a feature based on a moving window.
    
    ### Parameters
    - df (pd.DataFrame)
    - feat (str)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    rolling_min = df[feat].rolling(window=days, min_periods=1).min()
    rolling_max = df[feat].rolling(window=days, min_periods=1).max()
    moving_normalized_feature = (df[feat] - rolling_min) / (rolling_max - rolling_min)
    return moving_normalized_feature


def generate_model_data(data: np.array, sequence_size=20, target_idx=-1, pred_size=1):
    '''
    ### Description
    Formats data to train and test a stock prediction model.
    
    ### Parameters
    - data (np.array): Convert Pandas DataFrame to Numpy array.
    - sequence_size (int): Number of previous day model can see.
    - target_idx (int): Index of target column.
    - pred_size (int): Size of output.

    ### Returns:
    - list: X data (features)
    - list: Y data (prediction)
    '''
    data_x = []
    data_y = []
    for i in range(sequence_size, data.shape[0]-pred_size):
        data_x.append(data[i-sequence_size:i, :data.shape[1]].tolist())
        data_y.append(data[i:i+pred_size, target_idx].tolist())
    return np.array(data_x).astype(np.float32), np.array(data_y).astype(np.float32)

def generate_feature_data(data: np.array, sequence_size=20, pred_size=1):
    '''
    ### Description
    Formats feature data to train and test a stock prediction model.
    
    ### Parameters
    - data (np.array): Convert Pandas DataFrame to Numpy array.
    - sequence_size (int): Number of previous day model can see.
    - pred_size (int): Size of output.

    ### Returns
    - list: X data (features)
    - list: Y data (prediction)
    '''
    data_x = []
    for i in range(sequence_size, data.shape[0]-pred_size):
        data_x.append(data[i-sequence_size:i, :data.shape[1]].tolist())
    return np.array(data_x).astype(np.float32)

def calculate_mean_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    '''
    ### Description
    Computes the mean sentiment score for each stock per date.

    ### Parameters
    - df (pd.DataFrame): A DataFrame containing 'Date', 'Stock Name', and 'Tweet' columns.

    ### Returns:
    - pd.DataFrame: A DataFrame with 'Date', 'Stock Name', and 'Sentiment' columns.
    '''
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    sentiment_scores = []
    for tweet in tqdm(df['Tweet'], desc="Calculating Sentiment Scores"):
        sentiment_scores.append(stm().polarity_scores(tweet)['compound'])
    df['Sentiment'] = sentiment_scores
    sentiment_df = df.groupby(['Date', 'Stock Name'])['Sentiment'].mean().reset_index()
    return sentiment_df