'''
COM SCI 247 (Winter 25)

Stock Data Processing Functions
'''


import pandas as pd
import numpy as np


def clean_stock_data(df: pd.DataFrame, window=30) -> pd.DataFrame:
    '''
    ### Description
    Cleans and filters stock data for a given ticker symbol. Generates moving averages.
    
    ### Parameters
    - df (pd.DataFrame):
        The original stock dataset containing multiple stock entries.

    ### Returns:
    - (pd.DataFrame):
        A cleaned DataFrame containing only the specified stock's data, 
        with formatted columns, sorted dates, and missing values handled.
    '''
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df.ffill(inplace=True)
    moving_df = df.copy()
    moving_df['moving_max'] = moving_max(df, days=window)
    moving_df['moving_min'] = moving_min(df, days=window)
    moving_df['moving_avg_volume'] = moving_avg_volume(df, days=window)
    moving_df['moving_avg_HL'] = moving_avg_HL(df, days=window)
    return moving_df.copy()


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


def calculate_closing_diff(df: pd.DataFrame) -> tuple:
    '''
    ### Description
    Calculates the percent change in closing price between days.
    
    ### Parameters
    - df (pd.DataFrame)

    ### Returns:
    - (pd.Series)
    '''
    delta = df['close'].diff(1)
    delta = 100*delta/df['close'].shift(periods=1, fill_value=0)
    delta.loc[0] = 0
    return pd.Series(delta)

def moving_max(df: pd.DataFrame, days=30) -> pd.Series:
    '''
    ### Description
    Calculates the moving max of a stock.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return df['close'].rolling(window=days, min_periods=1).max()

def moving_min(df: pd.DataFrame, days=30) -> pd.Series:
    '''
    ### Description
    Calculates the moving min of a stock.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return df['close'].rolling(window=days, min_periods=1).min()

def moving_avg_volume(df: pd.DataFrame, days=30) -> pd.Series:
    '''
    ### Description
    Calculates the moving avg trade volume of a stock.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return df['volume'].rolling(window=days, min_periods=1).mean()

def moving_avg_HL(df: pd.DataFrame, days=30) -> pd.Series:
    '''
    ### Description
    Calculates the moving avg trade volume of a stock.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return (df['high']-df['low']).rolling(window=days, min_periods=1).mean()

def normalize_close(df: pd.DataFrame) -> pd.Series:
    '''
    ### Description
    Calculates the closing price of a stock as accumulated percent deltas.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    accum = df['target'].cumsum()
    accum = accum/accum.mean()
    return accum

def normalize_volume(df: pd.DataFrame) -> pd.Series:
    '''
    ### Description
    Calculates the trade volume of a stock normalized to the window's high/low trade volume.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return (df['volume']-df['moving_avg_volume'])/df['moving_avg_volume']

def normalize_HL(df: pd.DataFrame) -> pd.Series:
    '''
    ### Description
    Calculates the daily variation of a stock price normalized to the window's high/low daily variation.
    
    ### Parameters
    - df (pd.DataFrame)
    - days (int)

    ### Returns:
    - (pd.Series)
    '''
    return (df['high']-df['low'])/df['moving_avg_HL']

def generate_LSTM_data(data, sequence_size, target_idx, pred_size=1):
    '''
    ### Description
    Generates LSTM data.
    
    ### Parameters
    - np.array: Raw data

    ### Returns:
    - np.array: X data (features)
    - np.array: Y data (prediction)
    '''
    data_X = []
    data_y = []
    for i in range(sequence_size, data.shape[0]-pred_size):
        data_X.append(data[i-sequence_size:i, :data.shape[1]])
        data_y.append(data[i:i+pred_size, target_idx])

    return np.array(data_X).astype(np.float32), np.array(data_y).astype(np.float32)

def gen_features(df: pd.DataFrame):
    '''
    ### Description
    Generates stock features using above functions and returns new pd dataframe containing these features.
    
    ### Parameters
    - df (pd.DataFrame)

    ### Returns:
    - df (pd.DataFrame)
    '''
    features = pd.DataFrame()
    #normalize close, volume, high-low with respect to moving min/max
    features['target'] = calculate_closing_diff(df)
    features['close'] = normalize_close(features)
    features['volume'] = normalize_volume(df)
    features['day_HL'] = normalize_HL(df)
    #generate subsequent features using normalized close/volume/hl
    features['sma'] = simple_moving_average(features)
    features['ema'] = exponential_moving_average(features)
    features['rsi'] = calculate_rsi(features)/50
    features['macd_line'] , features['macd_signal'], features['macd_histogram'] = calculate_macd(features)
    features['bb_middle'], features['bb_upper'], features['bb_lower'] = bollinger_bands(features)
    return features.copy()