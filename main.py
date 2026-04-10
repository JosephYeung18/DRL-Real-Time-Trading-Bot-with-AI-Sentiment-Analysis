"""
TSLA DRL Trading System - Main Entry Point
====================================================================
A2C/PPO Model Training with FinRL, Backtesting with Lumibot.

Usage:
    python main.py --mode train     # Train from scratch
    python main.py --mode backtest  # Load existing model and backtest
    python main.py --mode all       # Train and backtest (default)
"""

import argparse
import numpy as np
import pandas as pd
import warnings
import os
import sys
from datetime import datetime
import pytz

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environments.trading_env import StockTradingEnvOptimized
from src.strategies.rl_strategy import RLStrategyProportional
from src.utils.model_loader import load_trained_model, save_trained_model
from src.utils.indicators import add_custom_indicators, prepare_data
from src.utils.metrics import (
    calculate_metrics, calculate_metrics_numeric, max_drawdown,
    calculate_holding_days, save_metrics_comparison_csv, generate_text_report,
    save_trading_log
)
from src.utils.plotting import plot_full_evaluation
from src.utils.config import (
    TICKER, MODEL_TYPE, INITIAL_CAPITAL, TRANSACTION_FEE,
    ACTION_SMOOTHING, TRAIN_START_DATE, TRAIN_END_DATE,
    TRADE_START_DATE, TRADE_END_DATE, TOTAL_TIMESTEPS,
    OUTPUT_DIR, SMA_SHORT, SMA_LONG, REWARD_CONFIG
)
from src.utils.benchmarks import run_sma_strategy, run_benchmark

# Stable Baselines
from stable_baselines3 import A2C, PPO

# Lumibot
from lumibot.backtesting.backtesting_broker import BacktestingBroker
from lumibot.data_sources.pandas_data import PandasData
from lumibot.entities import Asset, Data
from lumibot.traders import Trader

# FinRL
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.agents.stablebaselines3.models import DRLAgent

def parse_args():
    parser = argparse.ArgumentParser(description='TSLA DRL Trading System')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'backtest', 'all'],
                        help='Mode: train (only training), backtest (load & backtest), all (train & backtest)')
    parser.add_argument('--model', type=str, default='a2c',
                        choices=['a2c', 'ppo'],
                        help='Model type')
    parser.add_argument('--steps', type=int, default=TOTAL_TIMESTEPS,
                        help='Training steps')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to load existing model (for backtest mode)')
    return parser.parse_args()

def train_model(train_data, trade_data, indicators, model_type='a2c'):
    """Train DRL model with early stopping"""
    print("=" * 60)
    print(f"Training {model_type.upper()} model with early stopping...")
    print("=" * 60)

    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension

    print(f"Stock dimension: {stock_dimension}")
    print(f"State space: {state_space}")
    print(f"Action space: continuous [0, 1] (target position ratio)")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": INITIAL_CAPITAL,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [TRANSACTION_FEE],
        "sell_cost_pct": [TRANSACTION_FEE],
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": indicators,
        "action_space": stock_dimension,
        "reward_scaling": REWARD_CONFIG['reward_scaling']
    }

    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    train_env = StockTradingEnvOptimized(df=train_data, **env_kwargs)
    env_train, _ = train_env.get_sb_env()

    # Validation set (fully independent from training data)
    val_start = '2023-06-01'
    val_end = '2023-12-31'
    val_data = data_split(train_data, val_start, val_end)
    val_env = StockTradingEnvOptimized(df=val_data, **env_kwargs)
    env_val, _ = val_env.get_sb_env()

    agent = DRLAgent(env=env_train)

    model_params = {
        'a2c': {
            "n_steps": 2048,
            "ent_coef": 0.001,
            "learning_rate": 0.0007,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        },
        'ppo': {
            "n_steps": 2048,
            "ent_coef": 0.001,
            "learning_rate": 0.0007,
            "batch_size": 128,
            "clip_range": 0.2,
        }
    }

    tensorboard_log = f"results/{model_type}"
    model = agent.get_model(model_type, model_kwargs=model_params.get(model_type, {}), tensorboard_log=tensorboard_log)

    # Early stopping configuration
    best_val_reward = float('-inf')
    patience_counter = 0
    patience_limit = 10
    eval_interval = 10000

    n_envs = 1
    current_step = 0
    best_model_path = f"trained_models/best/agent_{model_type}_best"

    while current_step < TOTAL_TIMESTEPS:
        model = agent.train_model(
            model=model,
            tb_log_name=model_type,
            total_timesteps=min(eval_interval, TOTAL_TIMESTEPS - current_step)
        )
        current_step += eval_interval

        # Evaluate on validation set
        obs_val = env_val.reset()
        val_rewards = []
        done = False
        while not done:
            action_val, _ = model.predict(obs_val, deterministic=True)
            obs_val, reward, terminated, truncated, info = val_env.step(action_val)
            done = terminated or truncated
            val_rewards.append(reward)

        mean_val_reward = np.mean(val_rewards) if val_rewards else 0
        print(f"  Step {current_step}: Validation reward = {mean_val_reward:.4f}")

        if mean_val_reward > best_val_reward:
            best_val_reward = mean_val_reward
            patience_counter = 0
            model.save(best_model_path)
            print(f"    -> New best model! Saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"    -> No improvement ({patience_counter}/{patience_limit})")
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping triggered!")
                break

    # Load best model
    model.load(best_model_path)
    print(f"\nLoaded best model from {best_model_path}")
    return model, trade_data, indicators

def run_lumibot_backtest(model, trade_data, indicators, model_type):
    """Run backtest using Lumibot"""
    print("\n" + "=" * 60)
    print("Running Lumibot Backtest...")
    print("=" * 60)

    trade_df = trade_data.copy()
    trade_df['datetime'] = pd.to_datetime(trade_df['date'])
    trade_df = trade_df.set_index('datetime')
    trade_df.index.name = "datetime"

    trade_df = trade_df[(trade_df.index >= TRADE_START_DATE) & (trade_df.index <= TRADE_END_DATE)]

    print(f"Backtest period: {trade_df.index.min()} to {trade_df.index.max()}")
    print(f"Trading days: {len(trade_df)}")

    eastern = pytz.timezone('America/New_York')
    backtest_start_tz = eastern.localize(datetime(2024, 1, 2))
    backtest_end_tz = eastern.localize(datetime(2024, 12, 30))

    trade_df_tz = trade_df.copy()
    if trade_df_tz.index.tz is None:
        trade_df_tz.index = trade_df_tz.index.tz_localize(eastern)

    tsla_asset = Asset(symbol=TICKER, asset_type="stock")
    data = Data(
        asset=tsla_asset,
        df=trade_df_tz,
        date_start=backtest_start_tz,
        date_end=backtest_end_tz,
        timestep='day'
    )

    pandas_data = PandasData(
        pandas_data=[data],
        datetime_start=backtest_start_tz,
        datetime_end=backtest_end_tz,
        auto_adjust=False,
        allow_option_quote_fallback=False
    )

    broker = BacktestingBroker(pandas_data)

    strategy_params = {
        'model': model,
        'trade_data': trade_data,
        'indicators': indicators,
        'model_type': model_type,
        'symbol': TICKER,
        'action_smoothing': ACTION_SMOOTHING,
        'initial_capital': INITIAL_CAPITAL
    }

    strategy = RLStrategyProportional(broker=broker, parameters=strategy_params)

    trader = Trader(backtest=True)
    trader.add_strategy(strategy)

    results = trader.run_all(
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        show_indicators=False
    )

    # Extract trades - try multiple methods
    trades = []
    if results and isinstance(results, dict):
        for strategy_name, strategy_result in results.items():
            if isinstance(strategy_result, dict):
                strategy_trades = strategy_result.get('trades', [])
                if strategy_trades:
                    trades.extend(strategy_trades)
            elif isinstance(strategy_result, list):
                trades.extend(strategy_result)

    # If no trades found, try to read from Lumibot's internal trade file
    if len(trades) == 0:
        trade_file = 'logs/RLStrategyProportional_trades.csv'
        if os.path.exists(trade_file):
            try:
                trade_df = pd.read_csv(trade_file)
                # Filter only filled orders
                fills = trade_df[trade_df['status'] == 'fill'].copy()
                if len(fills) > 0:
                    # Convert to a list of trade-like objects
                    for _, row in fills.iterrows():
                        trade = type('Trade', (), {
                            'symbol': row['symbol'],
                            'side': row['side'],
                            'quantity': row['filled_quantity'],
                            'price': row['price'],
                            'time': row['time'],
                        })()
                        trades.append(trade)
                    print(f"  -> Loaded {len(trades)} trades from Lumibot log file")
            except Exception as e:
                print(f"  -> Could not parse trade file: {e}")

    # Get account value history and action history
    account_values = None
    raw_actions = None
    smoothed_actions = None
    shares_history = None
    if hasattr(trader, '_strategies') and len(trader._strategies) > 0:
        strategy_obj = trader._strategies[0]
        if hasattr(strategy_obj, 'account_value_history') and len(strategy_obj.account_value_history) > 0:
            account_values = strategy_obj.account_value_history
        if hasattr(strategy_obj, 'raw_action_history') and len(strategy_obj.raw_action_history) > 0:
            raw_actions = strategy_obj.raw_action_history
        if hasattr(strategy_obj, 'smoothed_action_history') and len(strategy_obj.smoothed_action_history) > 0:
            smoothed_actions = strategy_obj.smoothed_action_history
        if hasattr(strategy_obj, 'shares_history') and len(strategy_obj.shares_history) > 0:
            shares_history = strategy_obj.shares_history

    print(f"\nBacktest completed! Trades: {len(trades)}")
    return broker, trades, account_values, raw_actions, smoothed_actions, shares_history

def main():
    args = parse_args()

    print("=" * 60)
    print("TSLA DRL Trading System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model.upper()}")
    print(f"Training period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"Trading period: {TRADE_START_DATE} to {TRADE_END_DATE}")
    print("=" * 60)

    # Prepare data
    print("\n1. Preparing data...")
    train_data, trade_data, indicators = prepare_data('data/TSLA.csv')
    print(f"   Training data: {len(train_data)} records")
    print(f"   Trading data: {len(trade_data)} records")

    model = None
    model_path = args.model_path or f"trained_models/best/agent_{args.model}_best.zip"

    # Train or load model
    if args.mode in ['train', 'all']:
        model, trade_data, indicators = train_model(train_data, trade_data, indicators, args.model)
        print("\nModel training completed!")
    else:
        # Load existing model
        print(f"\nLoading model from: {model_path}")
        try:
            if args.model == 'a2c':
                model = A2C.load(model_path)
            else:
                model = PPO.load(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to training...")
            model, trade_data, indicators = train_model(train_data, trade_data, indicators, args.model)

    # Run backtest
    if args.mode in ['backtest', 'all']:
        try:
            broker, trades, account_values, raw_actions, smoothed_actions, shares_history = run_lumibot_backtest(model, trade_data, indicators, args.model)

            if account_values and len(account_values) > 0:
                # Run benchmark strategies
                sma_returns, sma_dates = run_sma_strategy(trade_data)
                bench_returns, _ = run_benchmark(trade_data)

                # Prepare data
                trade_dates = pd.to_datetime(trade_data['date'].values)
                n = min(len(account_values), len(sma_returns), len(bench_returns))
                model_equity = np.array(account_values)[:n]
                sma_equity = INITIAL_CAPITAL * np.cumprod(1 + sma_returns[:n])
                bench_equity = INITIAL_CAPITAL * np.cumprod(1 + bench_returns[:n])
                prices_plot = trade_data['close'].values[:n]
                dates_plot = trade_dates[:n]
                model_returns = np.diff(model_equity) / model_equity[:-1]

                # Prepare actual model actions for plotting
                model_actions_for_plot = np.array(raw_actions)[:n] if raw_actions is not None else np.zeros(len(dates_plot))
                smoothed_actions_for_plot = np.array(smoothed_actions)[:n] if smoothed_actions is not None else np.ones(len(dates_plot)) * 0.5

                # Generate evaluation
                print("\n2. Generating evaluation reports...")
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                # Plot
                import matplotlib
                matplotlib.use('Agg')
                plot_full_evaluation(
                    dates_plot, model_equity, sma_equity, bench_equity,
                    prices_plot, model_actions_for_plot, smoothed_actions_for_plot,
                    trades=trades, model_returns=model_returns,
                    sma_returns=sma_returns[:n], bench_returns=bench_returns[:n],
                    output_dir=OUTPUT_DIR
                )

                # Save metrics
                metrics_list = [
                    calculate_metrics(model_returns, 'DRL Model'),
                    calculate_metrics(sma_returns[:n], f'SMA ({SMA_SHORT},{SMA_LONG})'),
                    calculate_metrics(bench_returns[:n], 'Benchmark')
                ]
                save_metrics_comparison_csv(metrics_list, f'{OUTPUT_DIR}/metrics_comparison.csv')

                # Generate report (no sentiment mode)
                generate_text_report(
                    dates_plot, model_equity, sma_equity, bench_equity,
                    prices_plot, model_actions_for_plot, smoothed_actions_for_plot,
                    trades, model_returns, sma_returns[:n], bench_returns[:n],
                    f'{OUTPUT_DIR}/report.txt',
                    sentiment_mode=0
                )

                # Save detailed trading logs to logs/
                print("\n3. Saving detailed trading logs...")
                save_trading_log(
                    trades=trades,
                    account_values=account_values,
                    trade_data=trade_data,
                    model_actions=raw_actions if raw_actions is not None else np.zeros(len(dates_plot)),
                    output_dir='logs',
                    sentiment_mode=0
                )

                print(f"\nResults saved to: {OUTPUT_DIR}/")
                print(f"  - full_evaluation_plot.png")
                print(f"  - metrics_comparison.csv")
                print(f"  - report.txt")

        except Exception as e:
            print(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
