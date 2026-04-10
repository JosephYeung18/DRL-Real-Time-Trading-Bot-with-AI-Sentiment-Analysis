"""
Configuration module for EURUSD DRL Trading System
"""

# Trading Parameters
TICKER = 'EURUSD'
MODEL_TYPE = 'a2c'  # Options: 'a2c', 'ppo'
INITIAL_CAPITAL = 1000000  # $1M for forex (standard lot size)
TRANSACTION_FEE = 0.0001  # 0.01% spread cost (forex is tighter)

# Action smoothing parameter
ACTION_SMOOTHING = 0.8  # α = 0.8, final_action = α * raw + (1-α) * last

# Date ranges
TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2023-12-31'
TRADE_START_DATE = '2024-01-01'
TRADE_END_DATE = '2024-12-31'

# Training parameters
TOTAL_TIMESTEPS = 300000

# Position limits (forex uses lots)
MAX_POSITION = 1.0  # Maximum position (100%)
MIN_POSITION = 0.0  # Minimum position (0%)

# Forex lot size
LOT_SIZE = 100000  # Standard lot = 100,000 units

# Reward configuration
REWARD_CONFIG = {
    'position_penalty_coef': 0.0,
    'transaction_penalty': 0.0,
    'drawdown_threshold': 0.30,
    'drawdown_penalty_coef': 0.05,
    'reward_scaling': 1.0,
    'momentum_boost': 0.5,
    'hold_bonus': 0.01,
}

# Directory paths
OUTPUT_DIR = 'eurusd_backtest_result'
OUTPUT_DIR_SENTIMENT = 'eurusd_backtest_result_sentiment'
TRAINED_MODEL_DIR = 'trained_models/eurusd'
RESULTS_DIR = 'results/eurusd'

# Data
DATA_PATH = 'data/EURUSD.csv'
