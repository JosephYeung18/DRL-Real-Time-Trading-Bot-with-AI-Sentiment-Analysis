"""
Configuration module for TSLA DRL Trading System
"""

# Trading Parameters
TICKER = 'TSLA'
MODEL_TYPE = 'a2c'  # Options: 'a2c', 'ppo'
INITIAL_CAPITAL = 100000
TRANSACTION_FEE = 0.001  # 0.1% transaction fee

# Action smoothing parameter
ACTION_SMOOTHING = 0.8  # α = 0.8, final_action = α * raw + (1-α) * last

# Date ranges
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2023-12-31'
TRADE_START_DATE = '2024-01-01'
TRADE_END_DATE = '2024-12-31'

# Training parameters
TOTAL_TIMESTEPS = 300000

# Position limits
MAX_POSITION = 1.0  # Maximum position (100%)
MIN_POSITION = 0.0  # Minimum position (0%)

# SMA parameters for benchmark
SMA_SHORT = 5
SMA_LONG = 20

# Reward configuration
REWARD_CONFIG = {
    'position_penalty_coef': 0.0,     # No position penalty
    'transaction_penalty': 0.0,        # No transaction penalty
    'drawdown_threshold': 0.30,       # Drawdown threshold (30%)
    'drawdown_penalty_coef': 0.05,   # Drawdown penalty coefficient
    'reward_scaling': 1.0,           # Reward scaling
    'momentum_boost': 0.5,          # Momentum boost for winning positions
    'hold_bonus': 0.01,              # Holding bonus for trend following
}

# Directory paths
OUTPUT_DIR = 'tesla_backtest_result'
TRAINED_MODEL_DIR = 'trained_models'
RESULTS_DIR = 'results'
