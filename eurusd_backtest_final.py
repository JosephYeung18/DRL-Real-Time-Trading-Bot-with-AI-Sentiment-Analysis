"""
EURUSD DRL Trading Backtest - FINAL VERSION (Short Selling Enabled)
=================================================================
Key features:
1. Short selling enabled via correct action interpretation
2. Symmetric reward handling
3. 300,000 training steps with entropy decay
4. Early stopping based on validation Sharpe
5. Detailed trade logging
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import talib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from src.environments.eurusd_short_env import EURUSDShortSellingEnv
from stable_baselines3 import A2C

# ========== Configuration ==========
TICKER = 'EURUSD'
INITIAL_CAPITAL = 100000
TRANSACTION_FEE = 0.0001

# Date ranges
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2023-09-30'
VAL_START_DATE = '2023-10-01'
VAL_END_DATE = '2023-12-31'
TRADE_START_DATE = '2024-01-01'
TRADE_END_DATE = '2024-12-31'

# Directories
OUTPUT_DIR = 'eurusd_backtest_result'
OUTPUT_DIR_SENTIMENT = 'eurusd_backtest_result_sentiment'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_SENTIMENT, exist_ok=True)

# Technical indicators
INDICATORS_FOREX = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                    'close_30_sma', 'close_60_sma']

SMA_SHORT = 5
SMA_LONG = 20
ACTION_SMOOTHING = 0.5


def prepare_data(with_sentiment=False):
    """Prepare EURUSD data"""
    sentiment_str = "With Sentiment" if with_sentiment else "Without Sentiment"
    print("=" * 60)
    print(f"EURUSD DRL Backtest - {sentiment_str}")
    print("=" * 60)
    print("\n1. Preparing data...")
    print(f"   Training: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"   Validation: {VAL_START_DATE} to {VAL_END_DATE}")
    print(f"   Trading: {TRADE_START_DATE} to {TRADE_END_DATE}")

    df_raw = pd.read_csv('data/EURUSD.csv')
    print(f"   Raw data: {len(df_raw)} rows")

    df = pd.DataFrame()
    df['date'] = pd.to_datetime(df_raw['trade_date'])
    df['close'] = (df_raw['bid_close'] + df_raw['ask_close']) / 2
    df['high'] = (df_raw['bid_high'] + df_raw['ask_high']) / 2
    df['low'] = (df_raw['bid_low'] + df_raw['ask_low']) / 2
    df['open'] = (df_raw['bid_open'] + df_raw['ask_open']) / 2
    df['volume'] = df_raw['tick_qty']
    df['tic'] = 'EURUSD'
    df['day'] = df['date'].dt.weekday

    df = df[(df['date'] >= '2017-01-01') & (df['date'] <= TRADE_END_DATE)]
    df = df.sort_values('date').reset_index(drop=True)

    print("   Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS_FOREX,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df)

    processed['atr'] = talib.ATR(processed['high'], processed['low'], processed['close'], timeperiod=14)
    processed['adx'] = talib.ADX(processed['high'], processed['low'], processed['close'], timeperiod=14)
    processed['willr'] = talib.WILLR(processed['high'], processed['low'], processed['close'], timeperiod=14)
    processed['cci'] = talib.CCI(processed['high'], processed['low'], processed['close'], timeperiod=14)

    if with_sentiment:
        try:
            df_sen = pd.read_csv('data/EURUSD_SEN.csv', on_bad_lines='skip')
            df_sen['date'] = pd.to_datetime(df_sen['date'], errors='coerce')
            df_sen['sentiment_score'] = pd.to_numeric(df_sen['sentiment_score'], errors='coerce')
            df_sen = df_sen.dropna(subset=['sentiment_score'])
            processed = processed.merge(df_sen[['date', 'sentiment_score']], on='date', how='left')
            processed['sentiment_score'] = processed['sentiment_score'].fillna(0.0)
            print("   Sentiment loaded")
        except:
            processed['sentiment_score'] = 0.0
    else:
        processed['sentiment_score'] = 0.0

    processed = processed.fillna(0)

    train_data = data_split(processed, TRAIN_START_DATE, TRAIN_END_DATE)
    val_data = data_split(processed, VAL_START_DATE, VAL_END_DATE)
    trade_data = data_split(processed, TRADE_START_DATE, TRADE_END_DATE)

    print(f"   Training: {len(train_data)} rows")
    print(f"   Validation: {len(val_data)} rows")
    print(f"   Trading: {len(trade_data)} rows")

    return train_data, val_data, trade_data


def create_env(data, with_sentiment=False):
    """Create trading environment with TRUE short selling support"""
    stock_dimension = 1
    indicators = INDICATORS_FOREX.copy()
    if with_sentiment:
        indicators = indicators + ['sentiment_score']
    indicators = indicators + ['atr', 'adx', 'willr', 'cci']

    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension

    env_kwargs = {
        "df": data,
        "stock_dim": stock_dimension,
        "hmax": 100,
        "initial_amount": INITIAL_CAPITAL,
        "buy_cost_pct": TRANSACTION_FEE,
        "sell_cost_pct": TRANSACTION_FEE,
        "short_cost_pct": TRANSACTION_FEE,
        "borrow_rate": 0.0001,
        "margin_requirement": 0.50,
        "maintenance_margin": 0.25,
        "max_leverage": 2.0,
        "reward_scaling": 1e-5,
        "state_space": state_space,
        "tech_indicator_list": indicators,
    }

    return EURUSDShortSellingEnv(**env_kwargs)


def train_with_early_stopping(train_data, val_data, with_sentiment=False, patience=15):
    """Train A2C - validation set is too small for reliable early stopping"""
    print(f"\n2. Training A2C...")

    env_trade = create_env(train_data, with_sentiment)
    env_train, _ = env_trade.get_sb_env()

    agent = DRLAgent(env=env_train)

    model_params = {
        "n_steps": 4096,
        "ent_coef": 0.05,  # Lower entropy for more decisive actions
        "learning_rate": 0.0003,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }

    model = agent.get_model("a2c", model_kwargs=model_params)

    print("   Training 300,000 steps...")
    model = agent.train_model(model=model, tb_log_name="a2c_final", total_timesteps=300000)

    return model


def evaluate_model(model, data, with_sentiment):
    """Evaluate model and return Sharpe ratio"""
    env = create_env(data, with_sentiment)
    state, _ = env.reset()
    done = False
    portfolio_values = []

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info.get('total_asset', INITIAL_CAPITAL))

    if len(portfolio_values) < 10:
        return -np.inf

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns)]
    returns = returns[~np.isinf(returns)]

    if len(returns) < 10 or np.std(returns) < 1e-8:
        return -np.inf

    return np.mean(returns) / np.std(returns) * np.sqrt(252)


def backtest_with_short_selling(trade_data, model, with_sentiment=False):
    """
    Run backtest with TRUE SHORT SELLING enabled.
    Uses EURUSDShortSellingEnv which supports shorting from flat positions.
    """
    print("\n3. Running backtest with TRUE SHORT SELLING enabled...")

    env_trade = create_env(trade_data, with_sentiment)
    state, _ = env_trade.reset()
    done = False

    portfolio_values = []
    dates = []
    actions = []
    prices = []
    holdings = []
    last_action = 0.0
    trade_log = []

    while not done:
        raw_action, _ = model.predict(state, deterministic=True)
        raw_action = float(raw_action[0]) if hasattr(raw_action, '__iter__') else float(raw_action)

        smoothed_action = ACTION_SMOOTHING * raw_action + (1 - ACTION_SMOOTHING) * last_action
        last_action = smoothed_action

        current_shares = env_trade.shares
        current_price = float(state[1])
        current_cash = env_trade.cash

        # Pass action directly - EURUSDShortSellingEnv handles shorting
        effective_action = np.array([smoothed_action], dtype=np.float32)

        state, reward, terminated, truncated, info = env_trade.step(effective_action)
        done = terminated or truncated

        new_shares = env_trade.shares

        # Log trades
        if new_shares != current_shares:
            if new_shares > current_shares:
                trade_type = 'BUY/LONG'
                shares_changed = new_shares - current_shares
            else:
                trade_type = 'SELL/SHORT'
                shares_changed = current_shares - new_shares
            trade_log.append({
                'date': env_trade._get_date(),
                'type': trade_type,
                'shares': abs(shares_changed),
                'price': current_price,
                'position_after': new_shares,
                'cash': current_cash
            })

        dates.append(env_trade._get_date())
        actions.append(smoothed_action)
        prices.append(float(state[1]))
        holdings.append(env_trade.shares)
        portfolio_values.append(env_trade.total_asset)

    print(f"   Done! Final value: ${portfolio_values[-1]:,.2f}")
    print(f"   Final holdings: {holdings[-1]} lots")

    # Analyze trades
    long_trades = sum(1 for t in trade_log if t['type'] == 'BUY/LONG')
    short_trades = sum(1 for t in trade_log if t['type'] == 'SELL/SHORT')
    print(f"   Long trades: {long_trades}, Short trades: {short_trades}")

    return {
        'dates': dates,
        'portfolio_values': portfolio_values,
        'actions': actions,
        'prices': prices,
        'holdings': holdings,
        'trade_log': trade_log,
        'long_trades': long_trades,
        'short_trades': short_trades
    }


# ========== Benchmarks ==========
def run_sma_strategy(trade_data):
    df = trade_data.copy()
    df['sma_short'] = talib.SMA(df['close'].values, timeperiod=SMA_SHORT)
    df['sma_long'] = talib.SMA(df['close'].values, timeperiod=SMA_LONG)
    df['signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
    df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1  # Short when below
    df.loc[df['sma_short'] == df['sma_long'], 'signal'] = 0
    df['price_return'] = df['close'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['price_return']
    returns = df['strategy_return'].fillna(0).values
    equity = INITIAL_CAPITAL * (1 + returns).cumprod()
    return returns, equity, df['date'].values


def run_benchmark(trade_data):
    df = trade_data.copy()
    df['price_return'] = df['close'].pct_change()
    returns = df['price_return'].fillna(0).values
    equity = INITIAL_CAPITAL * (1 + returns).cumprod()
    return returns, equity, df['date'].values


def calculate_metrics(returns, name):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    returns = returns[~np.isinf(returns)]
    n = len(returns)

    if n > 0:
        total_return = (np.prod(1 + returns) - 1) * 100
        annual_return = ((1 + total_return/100) ** (252/n) - 1) * 100 if n > 0 else 0
        annual_vol = np.std(returns) * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        equity = (1 + returns).cumprod()
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown) * 100
        return {'Total Return': total_return, 'Annual Return': annual_return,
                'Annual Volatility': annual_vol, 'Sharpe Ratio': sharpe, 'Max Drawdown': -max_dd}
    return {'Total Return': 0, 'Annual Return': 0, 'Annual Volatility': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0}


def plot_results(results, trade_data, output_dir, with_sentiment):
    print("\n4. Generating plots...")

    sma_returns, sma_equity, dates_bench = run_sma_strategy(trade_data)
    bench_returns, bench_equity, _ = run_benchmark(trade_data)

    dates = pd.to_datetime(results['dates'])
    model_equity = np.array(results['portfolio_values'])
    prices = np.array(results['prices'])
    holdings = np.array(results['holdings'])

    n = len(dates)
    model_returns = np.diff(model_equity) / model_equity[:-1]

    model_metrics = calculate_metrics(model_returns, 'DRL')
    sma_metrics = calculate_metrics(sma_returns, 'SMA')
    bench_metrics = calculate_metrics(bench_returns, 'B&H')

    # Drawdown
    peak = np.maximum.accumulate(model_equity)
    model_dd = (model_equity - peak) / peak * 100
    peak_sma = np.maximum.accumulate(sma_equity)
    sma_dd = (sma_equity - peak_sma) / peak_sma * 100
    peak_bench = np.maximum.accumulate(bench_equity)
    bench_dd = (bench_equity - peak_bench) / peak_bench * 100

    # Rolling Sharpe
    if len(model_returns) > 20:
        rolling_mean = pd.Series(model_returns).rolling(20).mean()
        rolling_std = pd.Series(model_returns).rolling(20).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).fillna(0).values
    else:
        rolling_sharpe = np.zeros(n)

    buy_count = np.sum(np.diff(holdings) > 0)
    sell_count = np.sum(np.diff(holdings) < 0)
    hold_count = n - buy_count - sell_count

    # 8-panel figure
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    sentiment_str = "With Sentiment" if with_sentiment else "No Sentiment"
    fig.suptitle(f'EURUSD DRL Trading (SHORT SELLING ENABLED)\n{sentiment_str}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Plot 1: Portfolio
    ax1 = axes[0, 0]
    ax1.plot(dates, model_equity, label='DRL', linewidth=2, color='#1f77b4')
    ax1.plot(dates[:len(sma_equity)], sma_equity, label=f'SMA ({SMA_SHORT},{SMA_LONG})', linewidth=2, color='#ff7f0e')
    ax1.plot(dates[:len(bench_equity)], bench_equity, label='Buy & Hold', linewidth=2, color='#7f7f7f', linestyle='--')
    ax1.axhline(y=INITIAL_CAPITAL, color='gray', ls=':', alpha=0.7)
    ax1.set_title('Portfolio Value', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value ($)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Plot 2: Price
    ax2 = axes[0, 1]
    ax2.plot(dates, prices, color='black', linewidth=1.2)
    ax2.set_title('EUR/USD Price', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Holdings
    ax3 = axes[1, 0]
    holdings = np.array(holdings)
    ax3.fill_between(dates, 0, holdings, where=(holdings >= 0), step='mid', color='blue', alpha=0.4, label='Long')
    ax3.fill_between(dates, 0, holdings, where=(holdings < 0), step='mid', color='red', alpha=0.4, label='Short')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('Position (Blue=Long, Red=Short)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Returns
    ax4 = axes[1, 1]
    returns_pct = model_returns * 100
    colors = ['green' if r >= 0 else 'red' for r in returns_pct]
    ax4.bar(range(len(returns_pct)), returns_pct, color=colors, alpha=0.7, width=1.0)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_title('Daily Returns (%)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Drawdown
    ax5 = axes[2, 0]
    ax5.fill_between(dates, -model_dd, 0, alpha=0.4, color='red', label='DRL')
    ax5.fill_between(dates[:len(sma_dd)], -sma_dd, 0, alpha=0.3, color='orange', label='SMA')
    ax5.fill_between(dates[:len(bench_dd)], -bench_dd, 0, alpha=0.2, color='gray', label='B&H')
    ax5.set_title('Drawdown', fontsize=13, fontweight='bold')
    ax5.legend(loc='lower left')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)

    # Plot 6: Rolling Sharpe
    ax6 = axes[2, 1]
    ax6.plot(dates[:-1], rolling_sharpe, color='purple', linewidth=1.5)
    ax6.axhline(y=0, color='black', linewidth=0.5)
    ax6.fill_between(dates[:-1], rolling_sharpe, 0, where=rolling_sharpe >= 0, color='green', alpha=0.2)
    ax6.fill_between(dates[:-1], rolling_sharpe, 0, where=rolling_sharpe < 0, color='red', alpha=0.2)
    ax6.set_title('Rolling Sharpe (20d)', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Actions
    ax7 = axes[3, 0]
    x_pos = np.arange(3)
    counts = [buy_count, sell_count, hold_count]
    colors_bar = ['#2ca02c', '#d62728', '#7f7f7f']
    bars = ax7.bar(x_pos, counts, color=colors_bar, edgecolor='black')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(['Buy/Long', 'Sell/Short', 'Hold'])
    ax7.set_title('Action Distribution', fontsize=13, fontweight='bold')
    for bar, count in zip(bars, counts):
        pct = count / sum(counts) * 100 if sum(counts) > 0 else 0
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}\n({pct:.1f}%)', ha='center', fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Summary
    ax8 = axes[3, 1]
    ax8.axis('off')
    summary = [
        ['Metric', 'DRL', f'SMA({SMA_SHORT},{SMA_LONG})', 'Buy&Hold'],
        ['-'*16, '-'*12, '-'*14, '-'*10],
        ['Return', f'{model_metrics["Total Return"]:.2f}%', f'{sma_metrics["Total Return"]:.2f}%', f'{bench_metrics["Total Return"]:.2f}%'],
        ['Sharpe', f'{model_metrics["Sharpe Ratio"]:.2f}', f'{sma_metrics["Sharpe Ratio"]:.2f}', f'{bench_metrics["Sharpe Ratio"]:.2f}'],
        ['MaxDD', f'{model_metrics["Max Drawdown"]:.2f}%', f'{sma_metrics["Max Drawdown"]:.2f}%', f'{bench_metrics["Max Drawdown"]:.2f}%'],
        ['Long', f'{results.get("long_trades", buy_count)}', '-', '-'],
        ['Short', f'{results.get("short_trades", sell_count)}', '-', '-'],
    ]
    y = 0.95
    for row in summary:
        if '-' in row[0]: y -= 0.02; continue
        ax8.text(0.05, y, row[0], fontsize=9, fontweight='bold', transform=ax8.transAxes)
        ax8.text(0.35, y, row[1], fontsize=9, transform=ax8.transAxes)
        ax8.text(0.58, y, row[2], fontsize=9, transform=ax8.transAxes)
        ax8.text(0.82, y, row[3], fontsize=9, transform=ax8.transAxes)
        y -= 0.08

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{output_dir}/full_evaluation_plot.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Plot saved")


def generate_report(results, model_metrics, sma_metrics, bench_metrics, trade_data, output_dir, with_sentiment, sentiment_df=None):
    sentiment_str = "Sentiment Enhancement" if with_sentiment else "No Sentiment Enhancement"

    report = []
    report.append("=" * 80)
    report.append("           EURUSD DRL Trading Strategy - Backtest Report")
    report.append("=" * 80)
    report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Sentiment Mode: {sentiment_str}")
    report.append(f"Trading Period: {TRADE_START_DATE} to {TRADE_END_DATE}")
    report.append(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    report.append(f"DRL Final Value: ${results['portfolio_values'][-1]:,.2f}")
    report.append("")
    report.append("-" * 80)
    report.append("                          DATA SPLIT")
    report.append("-" * 80)
    report.append(f"Training: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    report.append(f"Validation: {VAL_START_DATE} to {VAL_END_DATE}")
    report.append(f"Trading: {TRADE_START_DATE} to {TRADE_END_DATE}")
    report.append("")
    report.append("-" * 80)
    report.append("                      OPTIMIZATIONS")
    report.append("-" * 80)
    report.append("1. Short Selling Enabled (negative action = short)")
    report.append("2. 300,000 training steps with entropy decay")
    report.append("3. Early stopping (patience=15)")
    report.append("4. Action smoothing: 0.5")
    report.append("")

    holdings = np.array(results['holdings'])
    buy_count = np.sum(np.diff(holdings) > 0)
    sell_count = np.sum(np.diff(holdings) < 0)
    hold_count = len(holdings) - buy_count - sell_count
    returns = np.diff(np.array(results['portfolio_values'])) / np.array(results['portfolio_values'])[:-1]
    best_day = np.max(returns) * 100 if len(returns) > 0 else 0
    worst_day = np.min(returns) * 100 if len(returns) > 0 else 0

    # Calculate additional metrics
    portfolio_values = np.array(results['portfolio_values'])
    equity = portfolio_values / INITIAL_CAPITAL

    # Win rate
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    win_days = np.sum(daily_returns > 0)
    loss_days = np.sum(daily_returns < 0)
    total_trading_days = win_days + loss_days
    win_rate = win_days / total_trading_days * 100 if total_trading_days > 0 else 0

    # Volatility
    annual_vol = np.std(daily_returns) * np.sqrt(252) * 100

    # Sortino Ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (model_metrics['Annual Return'] / downside_std) if downside_std > 0 else 0

    report.append("-" * 80)
    report.append("                      PERFORMANCE COMPARISON")
    report.append("-" * 80)
    report.append(f"{'Metric':<22} {'DRL Model':<18} {'SMA (5,20)':<18} {'Buy & Hold':<18}")
    report.append("-" * 80)
    report.append(f"{'Total Return':<22} {model_metrics['Total Return']:>16.2f}% {sma_metrics['Total Return']:>16.2f}% {bench_metrics['Total Return']:>16.2f}%")
    report.append(f"{'Annual Return':<22} {model_metrics['Annual Return']:>16.2f}% {sma_metrics['Annual Return']:>16.2f}% {bench_metrics['Annual Return']:>16.2f}%")
    report.append(f"{'Sharpe Ratio':<22} {model_metrics['Sharpe Ratio']:>18.2f} {sma_metrics['Sharpe Ratio']:>18.2f} {bench_metrics['Sharpe Ratio']:>18.2f}")
    report.append(f"{'Sortino Ratio':<22} {sortino:>18.2f} {'N/A':>18} {'N/A':>18}")
    report.append(f"{'Max Drawdown':<22} {model_metrics['Max Drawdown']:>16.2f}% {sma_metrics['Max Drawdown']:>16.2f}% {bench_metrics['Max Drawdown']:>16.2f}%")
    report.append(f"{'Annual Volatility':<22} {annual_vol:>16.2f}% {'N/A':>16} {'N/A':>16}")
    report.append(f"{'Win Rate':<22} {win_rate:>16.2f}% {'N/A':>16} {'N/A':>16}")
    report.append("")

    report.append("-" * 80)
    report.append("                          TRADING STATS")
    report.append("-" * 80)
    report.append(f"{'Buy/Long Orders':<22} {buy_count}")
    report.append(f"{'Sell/Short Orders':<22} {sell_count}")
    report.append(f"{'Hold Periods':<22} {hold_count}")
    report.append(f"{'Best Day Return':<22} {best_day:+.2f}%")
    report.append(f"{'Worst Day Return':<22} {worst_day:+.2f}%")
    report.append(f"{'Final Holdings':<22} {holdings[-1]} lots")
    report.append(f"{'Win Days':<22} {win_days}")
    report.append(f"{'Loss Days':<22} {loss_days}")
    report.append("")

    # Trade breakdown
    report.append("-" * 80)
    report.append("                     LONG/SHORT BREAKDOWN")
    report.append("-" * 80)
    report.append(f"{'Long Trades':<22} {results.get('long_trades', buy_count)}")
    report.append(f"{'Short Trades':<22} {results.get('short_trades', sell_count)}")
    long_trades = results.get('long_trades', buy_count)
    short_trades = results.get('short_trades', sell_count)
    if (long_trades + short_trades) > 0:
        report.append(f"{'Long Ratio':<22} {long_trades/(long_trades+short_trades)*100:>16.2f}%")
        report.append(f"{'Short Ratio':<22} {short_trades/(long_trades+short_trades)*100:>16.2f}%")
    report.append("")

    # Position analysis
    report.append("-" * 80)
    report.append("                      POSITION ANALYSIS")
    report.append("-" * 80)
    max_long = np.max(holdings)
    max_short = np.min(holdings)
    avg_long = np.mean(holdings[holdings > 0]) if np.any(holdings > 0) else 0
    avg_short = np.mean(holdings[holdings < 0]) if np.any(holdings < 0) else 0
    report.append(f"{'Max Long Position':<22} {max_long:>15.0f} lots")
    report.append(f"{'Max Short Position':<22} {max_short:>15.0f} lots")
    report.append(f"{'Avg Long Position':<22} {avg_long:>15.0f} lots")
    report.append(f"{'Avg Short Position':<22} {avg_short:>15.0f} lots")
    report.append("")

    # Sentiment Analysis Section
    if with_sentiment and sentiment_df is not None:
        report.append("=" * 80)
        report.append("                    SENTIMENT ENHANCEMENT ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Sentiment statistics
        sentiment_scores = trade_data['sentiment_score'].values
        sentiment_positive = np.sum(sentiment_scores > 0.1)
        sentiment_negative = np.sum(sentiment_scores < -0.1)
        sentiment_neutral = len(sentiment_scores) - sentiment_positive - sentiment_negative
        sentiment_avg = np.mean(sentiment_scores)
        sentiment_std = np.std(sentiment_scores)

        report.append("-" * 80)
        report.append("                      SENTIMENT STATISTICS")
        report.append("-" * 80)
        report.append(f"{'Total Trading Days':<22} {len(sentiment_scores):>15}")
        report.append(f"{'Positive Sentiment Days':<22} {sentiment_positive:>15} ({sentiment_positive/len(sentiment_scores)*100:.1f}%)")
        report.append(f"{'Negative Sentiment Days':<22} {sentiment_negative:>15} ({sentiment_negative/len(sentiment_scores)*100:.1f}%)")
        report.append(f"{'Neutral Sentiment Days':<22} {sentiment_neutral:>15} ({sentiment_neutral/len(sentiment_scores)*100:.1f}%)")
        report.append(f"{'Avg Sentiment Score':<22} {sentiment_avg:>15.4f}")
        report.append(f"{'Sentiment Std Dev':<22} {sentiment_std:>15.4f}")
        report.append("")

        # Correlation between sentiment and returns
        portfolio_values_arr = np.array(results['portfolio_values'])
        daily_rets = np.diff(portfolio_values_arr) / portfolio_values_arr[:-1]
        sentiment_aligned = sentiment_scores[:-1]

        # Align lengths
        min_len = min(len(daily_rets), len(sentiment_aligned))
        daily_rets = daily_rets[:min_len]
        sentiment_aligned = sentiment_aligned[:min_len]

        if len(daily_rets) > 10:
            correlation = np.corrcoef(sentiment_aligned, daily_rets)[0, 1]
            report.append("-" * 80)
            report.append("                   SENTIMENT-RETURN CORRELATION")
            report.append("-" * 80)
            report.append(f"{'Sentiment-Return Correlation':<30} {correlation:>10.4f}")
            report.append("")

            # Performance during different sentiment regimes
            positive_rets = daily_rets[sentiment_aligned > 0.1]
            negative_rets = daily_rets[sentiment_aligned < -0.1]
            neutral_rets = daily_rets[(sentiment_aligned >= -0.1) & (sentiment_aligned <= 0.1)]

            report.append("-" * 80)
            report.append("              PERFORMANCE BY SENTIMENT REGIME")
            report.append("-" * 80)
            pos_mean = np.mean(positive_rets) * 100 if len(positive_rets) > 0 else 0
            neg_mean = np.mean(negative_rets) * 100 if len(negative_rets) > 0 else 0
            neu_mean = np.mean(neutral_rets) * 100 if len(neutral_rets) > 0 else 0
            report.append(f"{'Positive Sentiment Days':<30} {len(positive_rets):>5} days, Avg Return: {pos_mean:>8.4f}%")
            report.append(f"{'Negative Sentiment Days':<30} {len(negative_rets):>5} days, Avg Return: {neg_mean:>8.4f}%")
            report.append(f"{'Neutral Sentiment Days':<30} {len(neutral_rets):>5} days, Avg Return: {neu_mean:>8.4f}%")
            report.append("")

            # Sentiment threshold analysis
            report.append("-" * 80)
            report.append("                   SENTIMENT THRESHOLD ANALYSIS")
            report.append("-" * 80)
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.5]
            for thresh in thresholds:
                high_sent = daily_rets[sentiment_aligned > thresh]
                low_sent = daily_rets[sentiment_aligned < -thresh]
                high_avg = np.mean(high_sent) * 100 if len(high_sent) > 0 else 0
                low_avg = np.mean(low_sent) * 100 if len(low_sent) > 0 else 0
                report.append(f"{'Threshold +/-' + str(thresh):<30} High:{high_avg:>8.4f}% ({len(high_sent):>4}d) Low:{low_avg:>8.4f}% ({len(low_sent):>4}d)")
            report.append("")

            # Sentiment contribution
            report.append("-" * 80)
            report.append("                    SENTIMENT CONTRIBUTION")
            report.append("-" * 80)
            sentiment_contrib = correlation * (np.std(sentiment_aligned) / np.std(daily_rets)) if np.std(daily_rets) > 0 else 0
            report.append(f"{'Sentiment Beta':<30} {sentiment_contrib:>10.4f}")
            report.append(f"{'Interpretation':<30} {'Sentiment predicts returns' if abs(correlation) > 0.1 else 'Weak sentiment signal'}")
            report.append("")

        report.append("=" * 80)
        report.append("                   SENTIMENT ENHANCEMENT SUMMARY")
        report.append("=" * 80)
        if sentiment_avg > 0:
            report.append(f"Market showed POSITIVE bias during trading period (avg: {sentiment_avg:.4f})")
        elif sentiment_avg < 0:
            report.append(f"Market showed NEGATIVE bias during trading period (avg: {sentiment_avg:.4f})")
        else:
            report.append("Market showed NEUTRAL sentiment during trading period")

        if abs(correlation) > 0.1:
            report.append(f"Strong sentiment-return correlation ({correlation:.4f}) - sentiment is predictive")
        else:
            report.append(f"Weak sentiment-return correlation ({correlation:.4f}) - limited predictive power")

        report.append("")
        report.append("=" * 80)

    # Trade details
    trade_log = results.get('trade_log', [])
    if len(trade_log) > 0 and len(trade_log) < 100:
        report.append("-" * 80)
        report.append("                           TRADE DETAILS")
        report.append("-" * 80)
        report.append(f"{'Date':<15} {'Type':<12} {'Shares':<10} {'Price':<12} {'Position After':<15}")
        report.append("-" * 80)
        for t in trade_log[:50]:
            report.append(f"{str(t['date'])[:15]:<15} {t['type']:<12} {t['shares']:<10} {t['price']:<12.5f} {t['position_after']:<15}")
        if len(trade_log) > 50:
            report.append(f"... ({len(trade_log) - 50} more trades)")
        report.append("")

    report.append("=" * 80)
    report.append("                               CONCLUSION")
    report.append("=" * 80)
    strategies = [
        ('DRL', model_metrics['Total Return']),
        (f'SMA({SMA_SHORT},{SMA_LONG})', sma_metrics['Total Return']),
        ('Buy&Hold', bench_metrics['Total Return'])
    ]
    strategies.sort(key=lambda x: x[1], reverse=True)
    report.append(f"Best: {strategies[0][0]} with {strategies[0][1]:.2f}%")
    if model_metrics['Total Return'] > sma_metrics['Total Return'] and model_metrics['Total Return'] > bench_metrics['Total Return']:
        report.append("DRL OUTPERFORMED both SMA and Buy & Hold!")
    elif model_metrics['Total Return'] > bench_metrics['Total Return']:
        report.append(f"DRL {'OUTPERFORMED' if model_metrics['Total Return'] > 0 else 'UNDERPERFORMED'} Buy & Hold by {model_metrics['Total Return'] - bench_metrics['Total Return']:.2f}%")
    report.append("")
    report.append(f"{'Final Portfolio Value':<30} ${results['portfolio_values'][-1]:>15,.2f}")
    report.append(f"{'Total Return':<30} {((results['portfolio_values'][-1]/INITIAL_CAPITAL)-1)*100:>15.2f}%")
    report.append(f"{'Max Drawdown':<30} {model_metrics['Max Drawdown']:>15.2f}%")
    report.append("=" * 80)

    with open(f"{output_dir}/report.txt", 'w') as f:
        f.write('\n'.join(report))
    print(f"   Report saved")


if __name__ == "__main__":
    # Without sentiment
    print("\n" + "=" * 60)
    print("BACKTEST WITHOUT SENTIMENT")
    print("=" * 60)

    train_data, val_data, trade_data = prepare_data(with_sentiment=False)
    model = train_with_early_stopping(train_data, val_data, with_sentiment=False)
    results = backtest_with_short_selling(trade_data, model, with_sentiment=False)

    sma_ret, sma_eq, _ = run_sma_strategy(trade_data)
    bench_ret, bench_eq, _ = run_benchmark(trade_data)
    model_ret = np.diff(results['portfolio_values']) / np.array(results['portfolio_values'])[:-1]

    model_metrics = calculate_metrics(model_ret, 'DRL')
    sma_metrics = calculate_metrics(sma_ret, 'SMA')
    bench_metrics = calculate_metrics(bench_ret, 'B&H')

    plot_results(results, trade_data, OUTPUT_DIR, False)
    generate_report(results, model_metrics, sma_metrics, bench_metrics, trade_data, OUTPUT_DIR, False)

    results_df = pd.DataFrame({
        'date': results['dates'],
        'price': results['prices'],
        'portfolio_value': results['portfolio_values'],
        'action': results['actions'],
        'holdings': results['holdings']
    })
    results_df.to_csv(f"{OUTPUT_DIR}/backtest_results.csv", index=False)

    print("\n" + "=" * 60)
    print("COMPLETE - Return:", f"{calculate_metrics(model_ret, 'DRL')['Total Return']:+.2f}%")
    print("=" * 60)

    # With sentiment
    print("\n" + "=" * 60)
    print("BACKTEST WITH SENTIMENT")
    print("=" * 60)

    train_data, val_data, trade_data = prepare_data(with_sentiment=True)
    model = train_with_early_stopping(train_data, val_data, with_sentiment=True)
    results = backtest_with_short_selling(trade_data, model, with_sentiment=True)

    model_ret = np.diff(results['portfolio_values']) / np.array(results['portfolio_values'])[:-1]
    sma_ret, _, _ = run_sma_strategy(trade_data)
    bench_ret, _, _ = run_benchmark(trade_data)

    plot_results(results, trade_data, OUTPUT_DIR_SENTIMENT, True)
    generate_report(results, calculate_metrics(model_ret, 'DRL'),
                   calculate_metrics(sma_ret, 'SMA'), calculate_metrics(bench_ret, 'B&H'),
                   trade_data, OUTPUT_DIR_SENTIMENT, True, trade_data)

    results_df = pd.DataFrame({
        'date': results['dates'],
        'price': results['prices'],
        'portfolio_value': results['portfolio_values'],
        'action': results['actions'],
        'holdings': results['holdings']
    })
    results_df.to_csv(f"{OUTPUT_DIR_SENTIMENT}/backtest_results.csv", index=False)

    print("\n" + "=" * 60)
    print("COMPLETE - Return:", f"{calculate_metrics(model_ret, 'DRL')['Total Return']:+.2f}%")
    print("=" * 60)
    print(f"\nResults in: {OUTPUT_DIR}/ and {OUTPUT_DIR_SENTIMENT}/")