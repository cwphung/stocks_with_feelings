'''
COM SCI 247 (Winter 25)

Stock Data Processing Functions
'''


import pandas as pd


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    ### Description
    Cleans and filters stock data for a given ticker symbol.
    
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
    return df


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
    return df['close'].rolling(window=days).mean()


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
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=days).mean()
    avg_loss = loss.rolling(window=days).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

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