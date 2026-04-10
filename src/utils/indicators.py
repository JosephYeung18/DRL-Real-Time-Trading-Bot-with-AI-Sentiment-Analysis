"""
Technical Indicators and Data Preparation Module
"""

import numpy as np
import pandas as pd
import talib
from finrl.config import INDICATORS
from .config import TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE

def add_custom_indicators(df):
    """Add custom technical indicators using TA-Lib matching FinRL conventions"""
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

    # Bollinger Bands (FinRL naming)
    df['boll_ub'], df['boll_lb'], df['boll_middle'] = talib.BBANDS(close, timeperiod=20)

    # RSI (FinRL naming - RSI with 30 period)
    df['rsi_30'] = talib.RSI(close, timeperiod=14)

    # CCI (FinRL naming)
    df['cci_30'] = talib.CCI(high, low, close, timeperiod=14)

    # DX (FinRL naming - ADX component)
    df['dx_30'] = talib.DX(high, low, close, timeperiod=14)

    # Moving Averages (FinRL naming)
    df['close_30_sma'] = talib.SMA(close, timeperiod=30)
    df['close_60_sma'] = talib.SMA(close, timeperiod=60)

    # Additional indicators
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)
    df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)

    # Sentiment score (momentum-based)
    df['sentiment_score'] = (df['close'] - df['open']) / df['open']

    return df

def add_turbulence_index(df):
    """Add turbulence index for risk assessment"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['turbulence'] = (df['returns'] - df['returns'].rolling(20).mean()) / df['returns'].rolling(20).std()
    df['turbulence'] = df['turbulence'].fillna(0)
    return df

def add_user_defined_features(df):
    """Add user-defined features"""
    df = df.copy()

    # Price momentum
    df['momentum_5'] = df['close'].pct_change(5)

    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_5'] = df['returns'].rolling(5).std()

    # Price relative to moving averages
    df['price_to_sma5'] = df['close'] / df['close_30_sma']
    df['price_to_sma20'] = df['close'] / df['close_60_sma']

    return df

def prepare_data(data_csv='data/TSLA.csv'):
    """Prepare data with technical indicators"""
    print("=" * 60)
    print("Preparing data...")
    print("=" * 60)

    # Load data
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} records from {data_csv}")

    # Standardize column names
    column_mapping = {
        'TradingDate': 'date',
        'ClosePrice': 'close',
        'OpenPrice': 'open',
        'HighPrice': 'high',
        'LowPrice': 'low',
        'Volume': 'volume',
    }
    df = df.rename(columns=column_mapping)

    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.drop('datetime', axis=1)

    df = df.sort_values('date').reset_index(drop=True)

    # Add 'tic' column for FinRL compatibility
    df['tic'] = 'TSLA'

    # Add indicators
    df = add_custom_indicators(df)
    print("Added technical indicators")

    df = add_turbulence_index(df)
    print("Added turbulence index")

    df = add_user_defined_features(df)
    print("Added user defined features")

    # Split data
    from finrl.meta.preprocessor.preprocessors import data_split
    train_data = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
    trade_data = data_split(df, TRADE_START_DATE, TRADE_END_DATE)

    # Indicators list - match original FinRL configuration
    indicators = INDICATORS + ['cci', 'willr', 'adx', 'atr', 'sentiment_score']

    print(f"\nTraining data: {len(train_data)} records")
    print(f"Trading data: {len(trade_data)} records")
    print(f"Indicators ({len(indicators)}): {indicators}")

    return train_data, trade_data, indicators
