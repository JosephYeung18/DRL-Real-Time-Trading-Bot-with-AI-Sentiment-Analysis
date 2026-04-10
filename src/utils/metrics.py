"""
Performance Metrics Calculation and Reporting
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

SMA_SHORT = 5
SMA_LONG = 20
INITIAL_CAPITAL = 100000

def max_drawdown(returns):
    """Calculate maximum drawdown from returns"""
    wealth = (1 + np.array(returns)).cumprod()
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    return np.min(drawdown)

def max_drawdown_from_equity(equity):
    """Calculate maximum drawdown from equity curve"""
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return np.min(drawdown)

def calculate_metrics_numeric(returns, name):
    """Calculate metrics with numeric values"""
    returns = np.array(returns)
    n = len(returns)

    if n > 0:
        total_return = (np.prod(1 + returns) - 1) * 100
        annual_return = ((1 + total_return/100) ** (252/n) - 1) * 100 if n > 0 else 0
        annual_vol = np.std(returns) * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        max_dd = max_drawdown(returns) * 100

        return {
            'name': name,
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': -max_dd
        }
    else:
        return {
            'name': name,
            'Total Return': 0.0,
            'Annual Return': 0.0,
            'Annual Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0
        }

def calculate_metrics(returns, name):
    """Calculate metrics with string formatting"""
    metrics = calculate_metrics_numeric(returns, name)
    return {
        'name': name,
        'Total Return': f"{metrics['Total Return']:.2f}%",
        'Annual Return': f"{metrics['Annual Return']:.2f}%",
        'Annual Volatility': f"{metrics['Annual Volatility']:.2f}%",
        'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.2f}",
        'Max Drawdown': f"{metrics['Max Drawdown']:.2f}%"
    }

def calculate_holding_days(trades):
    """Calculate average holding days from trades"""
    if not trades or len(trades) < 2:
        return 0

    buy_dates = []
    sell_dates = []
    holding_periods = []

    for trade in trades:
        if hasattr(trade, 'side'):
            if trade.side.lower() == 'buy':
                buy_dates.append(trade.time)
            elif trade.side.lower() == 'sell':
                sell_dates.append(trade.time)

    if not buy_dates or not sell_dates:
        return 0

    # Convert to datetime if strings
    buy_dates = sorted([pd.to_datetime(d) for d in buy_dates])
    sell_dates = sorted([pd.to_datetime(d) for d in sell_dates])

    # Match each sell with the earliest unpaired buy before it
    paired_buys = set()
    for sell_date in sell_dates:
        for i, buy_date in enumerate(buy_dates):
            if buy_date < sell_date and i not in paired_buys:
                holding_periods.append((sell_date - buy_date).days)
                paired_buys.add(i)
                break

    if holding_periods:
        return np.mean(holding_periods)
    return 0

def save_metrics_comparison_csv(metrics_list, filepath):
    """Save metrics comparison to CSV"""
    df = pd.DataFrame(metrics_list)
    df = df.set_index('name').T
    df.to_csv(filepath)
    print(f"Metrics saved to: {filepath}")

def generate_text_report(dates, model_equity, sma_equity, bench_equity,
                        prices, model_actions, smoothed_signals, trades,
                        model_returns, sma_returns, bench_returns, filepath,
                        sentiment_mode=0):
    """Generate text report with all metrics"""
    n = min(len(dates), len(model_equity), len(sma_equity), len(bench_equity))
    dates = dates[:n]
    model_equity = np.array(model_equity)[:n]
    sma_equity = np.array(sma_equity)[:n]
    bench_equity = np.array(bench_equity)[:n]

    # Calculate metrics
    model_metrics = calculate_metrics_numeric(model_returns[:n], 'DRL Model')
    sma_metrics = calculate_metrics_numeric(sma_returns[:n], f'SMA ({SMA_SHORT},{SMA_LONG})')
    bench_metrics = calculate_metrics_numeric(bench_returns[:n], 'Benchmark')

    # Calculate additional stats
    avg_holding_days = calculate_holding_days(trades) if trades else 0
    # Count actual trades from trades list
    buy_count = sum(1 for t in trades if getattr(t, 'side', '').lower() == 'buy') if trades else 0
    sell_count = sum(1 for t in trades if getattr(t, 'side', '').lower() == 'sell') if trades else 0
    hold_count = n - buy_count - sell_count

    # Find best and worst days
    if model_returns is not None and len(model_returns) > 0:
        best_day = np.max(model_returns) * 100
        worst_day = np.min(model_returns) * 100
    else:
        best_day = worst_day = 0

    # Sentiment mode label
    sentiment_label = "No Sentiment Enhancement"
    if sentiment_mode == 1:
        sentiment_label = "OpenAI API Sentiment"
    elif sentiment_mode == 2:
        sentiment_label = "Alpaca API Sentiment"
    elif sentiment_mode == 3:
        sentiment_label = "CSV Sentiment"

    report = []
    report.append("=" * 70)
    report.append("TSLA DRL Trading Strategy - Backtest Report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Sentiment Mode: {sentiment_label}")
    report.append(f"Trading Period: {pd.to_datetime(dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}")
    report.append(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    report.append(f"Final Portfolio Value (DRL): ${model_equity[-1]:,.2f}")
    report.append("")
    report.append("-" * 70)
    report.append("PERFORMANCE METRICS COMPARISON")
    report.append("-" * 70)
    report.append(f"{'Metric':<25} {'DRL Model':>15} {f'SMA ({SMA_SHORT},{SMA_LONG})':>20} {'Benchmark':>15}")
    report.append("-" * 70)
    report.append(f"{'Total Return':<25} {model_metrics['Total Return']:>14.2f}% {sma_metrics['Total Return']:>19.2f}% {bench_metrics['Total Return']:>14.2f}%")
    report.append(f"{'Annual Return':<25} {model_metrics['Annual Return']:>14.2f}% {sma_metrics['Annual Return']:>19.2f}% {bench_metrics['Annual Return']:>14.2f}%")
    report.append(f"{'Annual Volatility':<25} {model_metrics['Annual Volatility']:>14.2f}% {sma_metrics['Annual Volatility']:>19.2f}% {bench_metrics['Annual Volatility']:>14.2f}%")
    report.append(f"{'Sharpe Ratio':<25} {model_metrics['Sharpe Ratio']:>15.2f} {sma_metrics['Sharpe Ratio']:>20.2f} {bench_metrics['Sharpe Ratio']:>15.2f}")
    report.append(f"{'Max Drawdown':<25} {model_metrics['Max Drawdown']:>14.2f}% {sma_metrics['Max Drawdown']:>19.2f}% {bench_metrics['Max Drawdown']:>14.2f}%")
    report.append("")
    report.append("-" * 70)
    report.append("TRADING STATISTICS")
    report.append("-" * 70)
    report.append(f"{'Total Buy Orders':<25} {buy_count}")
    report.append(f"{'Total Sell Orders':<25} {sell_count}")
    report.append(f"{'Hold Periods':<25} {hold_count}")
    report.append(f"{'Average Holding Days':<25} {avg_holding_days:.1f}")
    report.append(f"{'Best Day Return':<25} {best_day:.2f}%")
    report.append(f"{'Worst Day Return':<25} {worst_day:.2f}%")
    report.append("")
    report.append("-" * 70)
    report.append("FINAL ASSET COMPARISON")
    report.append("-" * 70)
    report.append(f"{'DRL Model Final Value':<25} ${model_equity[-1]:,.2f} ({model_metrics['Total Return']:+.2f}%)")
    report.append(f"{f'SMA ({SMA_SHORT},{SMA_LONG}) Final Value':<25} ${sma_equity[-1]:,.2f} ({sma_metrics['Total Return']:+.2f}%)")
    report.append(f"{'Benchmark Final Value':<25} ${bench_equity[-1]:,.2f} ({bench_metrics['Total Return']:+.2f}%)")
    report.append("")
    report.append("=" * 70)
    report.append("CONCLUSION")
    report.append("=" * 70)

    # Determine winner
    strategies = [
        ('DRL Model', model_metrics['Total Return']),
        (f'SMA ({SMA_SHORT},{SMA_LONG})', sma_metrics['Total Return']),
        ('Benchmark', bench_metrics['Total Return'])
    ]
    strategies.sort(key=lambda x: x[1], reverse=True)

    best_strategy = strategies[0]
    report.append(f"Best Performing Strategy: {best_strategy[0]} with {best_strategy[1]:.2f}% return")

    if model_metrics['Total Return'] > sma_metrics['Total Return'] and model_metrics['Total Return'] > bench_metrics['Total Return']:
        report.append("The DRL model outperformed both SMA and Buy & Hold strategies.")
    elif model_metrics['Total Return'] > sma_metrics['Total Return']:
        report.append("The DRL model outperformed the SMA strategy but underperformed Buy & Hold.")
    else:
        report.append("The DRL model underperformed compared to benchmark strategies.")

    if model_metrics['Max Drawdown'] > sma_metrics['Max Drawdown']:
        report.append(f"The DRL model had lower maximum drawdown ({model_metrics['Max Drawdown']:.2f}%) compared to SMA ({sma_metrics['Max Drawdown']:.2f}%).")

    # Sentiment analysis section
    if sentiment_mode > 0:
        report.append("")
        report.append("=" * 70)
        report.append("SENTIMENT ENHANCEMENT ANALYSIS")
        report.append("=" * 70)
        sentiment_modes = {
            1: "OpenAI API",
            2: "Alpaca API",
            3: "Pre-saved CSV"
        }
        report.append(f"Sentiment Source: {sentiment_modes.get(sentiment_mode, 'Unknown')}")
        report.append("")
        report.append("The sentiment-enhanced model incorporates news sentiment data into its")
        report.append("decision-making process. This can provide the following benefits:")
        report.append("- Early detection of market sentiment shifts")
        report.append("- Better timing for entry and exit points")
        report.append("- Improved risk management during negative news events")
        report.append("")
        report.append("Note: The actual impact of sentiment enhancement depends on the")
        report.append("quality and relevance of the sentiment data source.")

    report.append("")
    report.append("=" * 70)

    with open(filepath, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved to: {filepath}")


def save_trading_log(trades, account_values, trade_data, model_actions, output_dir, sentiment_mode=0):
    """
    Save detailed trading log to CSV files.

    Args:
        trades: List of trade objects from Lumibot
        account_values: List of account values over time
        trade_data: DataFrame with price data
        model_actions: Array of model actions
        output_dir: Directory to save logs
        sentiment_mode: 0 for no sentiment, 1/2/3 for sentiment modes
    """
    os.makedirs(output_dir, exist_ok=True)

    sentiment_label = "no_sentiment" if sentiment_mode == 0 else f"sentiment{sentiment_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save trades log
    if trades and len(trades) > 0:
        trades_data = []
        for i, trade in enumerate(trades):
            trades_data.append({
                'trade_id': i + 1,
                'time': getattr(trade, 'time', None),
                'symbol': getattr(trade, 'symbol', 'TSLA'),
                'side': getattr(trade, 'side', None),
                'quantity': getattr(trade, 'quantity', 0),
                'price': getattr(trade, 'price', 0),
                'order_value': getattr(trade, 'quantity', 0) * getattr(trade, 'price', 0),
                'commission': getattr(trade, 'commission', 0),
            })
        df_trades = pd.DataFrame(trades_data)
        trades_path = f"{output_dir}/trades_{sentiment_label}_{timestamp}.csv"
        df_trades.to_csv(trades_path, index=False)
        print(f"Trades log saved to: {trades_path}")
    else:
        # Create empty trades file with headers
        df_trades = pd.DataFrame(columns=['trade_id', 'time', 'symbol', 'side', 'quantity', 'price', 'order_value', 'commission'])
        trades_path = f"{output_dir}/trades_{sentiment_label}_{timestamp}.csv"
        df_trades.to_csv(trades_path, index=False)
        print(f"Trades log saved to: {trades_path} (no trades)")

    # 2. Save account value history
    if account_values and len(account_values) > 0:
        account_data = {
            'date': trade_data['date'].values[:len(account_values)],
            'account_value': account_values,
        }
        # Calculate daily returns
        account_values_arr = np.array(account_values)
        account_data['daily_return'] = [0.0] + list(np.diff(account_values_arr) / account_values_arr[:-1])
        df_account = pd.DataFrame(account_data)
        account_path = f"{output_dir}/account_history_{sentiment_label}_{timestamp}.csv"
        df_account.to_csv(account_path, index=False)
        print(f"Account history saved to: {account_path}")

    # 3. Save model actions log
    if model_actions is not None and len(model_actions) > 0:
        actions_data = {
            'date': trade_data['date'].values[:len(model_actions)],
            'target_position_ratio': model_actions,
            'note': 'target_position_ratio is target position (0=cash, 1=full position)'
        }
        df_actions = pd.DataFrame(actions_data)
        actions_path = f"{output_dir}/actions_{sentiment_label}_{timestamp}.csv"
        df_actions.to_csv(actions_path, index=False)
        print(f"Actions log saved to: {actions_path}")

    # 4. Save summary statistics
    summary_path = f"{output_dir}/summary_{sentiment_label}_{timestamp}.csv"
    summary_data = {
        'metric': [
            'total_trades',
            'buy_trades',
            'sell_trades',
            'initial_capital',
            'final_value',
            'total_return_pct',
            'sentiment_mode'
        ],
        'value': [
            len(trades) if trades else 0,
            sum(1 for t in trades if getattr(t, 'side', '').lower() == 'buy') if trades else 0,
            sum(1 for t in trades if getattr(t, 'side', '').lower() == 'sell') if trades else 0,
            INITIAL_CAPITAL,
            account_values[-1] if account_values and len(account_values) > 0 else INITIAL_CAPITAL,
            ((account_values[-1] / INITIAL_CAPITAL - 1) * 100) if account_values and len(account_values) > 0 else 0,
            sentiment_label
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    return {
        'trades_path': trades_path if trades and len(trades) > 0 else None,
        'account_path': account_path if account_values and len(account_values) > 0 else None,
        'actions_path': actions_path if model_actions is not None and len(model_actions) > 0 else None,
        'summary_path': summary_path
    }
