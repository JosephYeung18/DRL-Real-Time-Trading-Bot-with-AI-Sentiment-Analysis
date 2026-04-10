"""
Benchmark Strategy Implementations
"""

import numpy as np
import pandas as pd
import talib

SMA_SHORT = 5
SMA_LONG = 20
INITIAL_CAPITAL = 100000

def run_sma_strategy(trade_data):
    """
    Run SMA crossover strategy

    Args:
        trade_data: DataFrame with price data

    Returns:
        sma_returns: Array of returns
        sma_dates: Array of dates
    """
    df = trade_data.copy()

    # Calculate SMAs
    df['sma_short'] = talib.SMA(df['close'].values, timeperiod=SMA_SHORT)
    df['sma_long'] = talib.SMA(df['close'].values, timeperiod=SMA_LONG)

    # Generate signals
    df['signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
    df.loc[df['sma_short'] <= df['sma_long'], 'signal'] = 0

    # Calculate returns
    df['price_return'] = df['close'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['price_return']

    sma_returns = df['strategy_return'].fillna(0).values
    sma_dates = df['date'].values if 'date' in df.columns else df.index.values

    return sma_returns, sma_dates

def run_benchmark(trade_data):
    """
    Run Buy & Hold benchmark strategy

    Args:
        trade_data: DataFrame with price data

    Returns:
        bench_returns: Array of returns
        bench_dates: Array of dates
    """
    df = trade_data.copy()

    # Calculate buy & hold returns
    df['price_return'] = df['close'].pct_change()
    bench_returns = df['price_return'].fillna(0).values
    bench_dates = df['date'].values if 'date' in df.columns else df.index.values

    return bench_returns, bench_dates
