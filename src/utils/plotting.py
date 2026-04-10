"""
Comprehensive Evaluation Plotting Functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SMA_SHORT = 5
SMA_LONG = 20
INITIAL_CAPITAL = 100000

def max_drawdown_from_equity(equity):
    """Calculate drawdown series from equity"""
    equity = np.array(equity)
    # 尝试清理字符串格式（如 '$100,000' -> 100000）
    if equity.dtype.kind in ['U', 'S', 'O']:
        equity = np.array([float(str(v).replace('$', '').replace(',', '')) for v in equity])
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak * 100

def plot_full_evaluation(dates, model_equity, sma_equity, bench_equity, prices,
                         model_actions, smoothed_signals, trades=None,
                         model_returns=None, sma_returns=None, bench_returns=None,
                         output_dir='tesla_backtest_result'):
    """
    Generate comprehensive evaluation plots
    """
    print("\nGenerating full evaluation plots...")

    n = min(len(dates), len(model_equity), len(sma_equity), len(bench_equity))
    dates = dates[:n]
    model_equity = np.array(model_equity)[:n]
    sma_equity = np.array(sma_equity)[:n]
    bench_equity = np.array(bench_equity)[:n]
    prices = prices[:n]

    # Ensure dates are datetime
    if not hasattr(dates, 'astype'):
        dates = pd.to_datetime(dates)
    else:
        dates = dates[:n]

    # Calculate metrics series
    model_cumret = (model_equity / model_equity[0] - 1) * 100
    sma_cumret = (sma_equity / sma_equity[0] - 1) * 100
    bench_cumret = (bench_equity / bench_equity[0] - 1) * 100

    # Calculate drawdown series
    model_dd = max_drawdown_from_equity(model_equity)
    sma_dd = max_drawdown_from_equity(sma_equity)
    bench_dd = max_drawdown_from_equity(bench_equity)

    # Calculate rolling Sharpe (20-day)
    window = 20
    if model_returns is not None and len(model_returns) > window:
        rolling_sharpe = pd.Series(model_returns).rolling(window).mean() / pd.Series(model_returns).rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.fillna(0).values
    else:
        rolling_sharpe = np.zeros(n)

    # Action analysis
    model_actions_arr = np.array(model_actions[:n]).flatten()
    buy_idx = np.where(model_actions_arr > 0)[0]
    sell_idx = np.where(model_actions_arr < 0)[0]
    buy_count = len(buy_idx)
    sell_count = len(sell_idx)
    hold_count = len(model_actions_arr) - buy_count - sell_count

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle('TSLA DRL Trading Strategy - Full Evaluation Report', fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Equity Curves
    ax1 = axes[0, 0]
    ax1.plot(dates, model_equity, label='DRL Model', linewidth=2, color='#1f77b4')
    ax1.plot(dates, sma_equity, label=f'SMA ({SMA_SHORT},{SMA_LONG})', linewidth=2, color='#ff7f0e')
    ax1.plot(dates, bench_equity, label='Buy & Hold', linewidth=2, color='#7f7f7f', linestyle='--')
    ax1.set_title('Portfolio Value Over Time', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Plot 2: Cumulative Returns
    ax2 = axes[0, 1]
    ax2.plot(dates, model_cumret, label='DRL Model', linewidth=2, color='#1f77b4')
    ax2.plot(dates, sma_cumret, label=f'SMA ({SMA_SHORT},{SMA_LONG})', linewidth=2, color='#ff7f0e')
    ax2.plot(dates, bench_cumret, label='Buy & Hold', linewidth=2, color='#7f7f7f', linestyle='--')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.fill_between(dates, model_cumret, 0, where=model_cumret >= 0, color='green', alpha=0.2)
    ax2.fill_between(dates, model_cumret, 0, where=model_cumret < 0, color='red', alpha=0.2)
    ax2.set_title('Cumulative Returns (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=30)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Plot 3: Price with Buy/Sell Signals
    ax3 = axes[1, 0]
    ax3.plot(dates, prices, label='TSLA Price', color='black', linewidth=1.2, alpha=0.8)
    if len(buy_idx) > 0:
        ax3.scatter(dates[buy_idx], prices[buy_idx], color='green', marker='^',
                   s=60, alpha=0.8, label=f'Buy ({buy_count})', zorder=5, edgecolors='darkgreen')
    if len(sell_idx) > 0:
        ax3.scatter(dates[sell_idx], prices[sell_idx], color='red', marker='v',
                   s=60, alpha=0.8, label=f'Sell ({sell_count})', zorder=5, edgecolors='darkred')
    ax3.set_title('DRL Trading Signals on Price', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Price ($)', fontsize=11)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=30)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Plot 4: Position Ratio Over Time
    ax4 = axes[1, 1]
    smoothed = np.array(smoothed_signals[:n]).flatten()
    ax4.fill_between(dates, 0, smoothed * 100, alpha=0.3, color='blue', label='Position (%)')
    ax4.plot(dates, smoothed * 100, color='blue', linewidth=1.5)
    ax4.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax4.set_title('Smoothed Position Ratio Over Time', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Position (%)', fontsize=11)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=30)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Plot 5: Drawdown Comparison
    ax5 = axes[2, 0]
    # 确保drawdown值是数值类型且在合理范围内
    model_dd = np.nan_to_num(model_dd, nan=0.0)
    sma_dd = np.nan_to_num(sma_dd, nan=0.0)
    bench_dd = np.nan_to_num(bench_dd, nan=0.0)

    # 限制最大值，避免极端值影响显示
    max_dd_display = 50  # 最大显示50%
    model_dd_clipped = np.clip(model_dd, -max_dd_display, 0)
    sma_dd_clipped = np.clip(sma_dd, -max_dd_display, 0)
    bench_dd_clipped = np.clip(bench_dd, -max_dd_display, 0)

    ax5.fill_between(dates, -model_dd_clipped, 0, alpha=0.4, color='red', label='DRL Model')
    ax5.fill_between(dates, -sma_dd_clipped, 0, alpha=0.3, color='orange', label=f'SMA ({SMA_SHORT},{SMA_LONG})')
    ax5.fill_between(dates, -bench_dd_clipped, 0, alpha=0.2, color='gray', label='Buy & Hold')
    ax5.set_title('Drawdown Over Time', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Drawdown (%)', fontsize=11)
    ax5.legend(loc='lower left', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=30)
    ax5.xaxis.set_major_locator(plt.MaxNLocator(8))
    # 设置y轴范围，确保drawdown区域可见
    ax5.set_ylim(bottom=0)

    # Plot 6: Rolling Sharpe Ratio
    ax6 = axes[2, 1]
    sharpe_len = min(len(rolling_sharpe), len(dates))
    ax6.plot(dates[:sharpe_len], rolling_sharpe[:sharpe_len], color='purple', linewidth=1.5, label='Rolling Sharpe (20d)')
    ax6.axhline(y=0, color='black', linewidth=0.5)
    ax6.axhline(y=1, color='green', linewidth=0.5, linestyle='--', alpha=0.7)
    ax6.axhline(y=-1, color='red', linewidth=0.5, linestyle='--', alpha=0.7)
    ax6.fill_between(dates[:sharpe_len], rolling_sharpe[:sharpe_len], 0,
                    where=rolling_sharpe[:sharpe_len] >= 0, color='green', alpha=0.2)
    ax6.fill_between(dates[:sharpe_len], rolling_sharpe[:sharpe_len], 0,
                    where=rolling_sharpe[:sharpe_len] < 0, color='red', alpha=0.2)
    ax6.set_title('Rolling Sharpe Ratio (20-day)', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Sharpe Ratio', fontsize=11)
    ax6.legend(loc='upper right', fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=30)
    ax6.xaxis.set_major_locator(plt.MaxNLocator(8))

    # Plot 7: Action Distribution
    ax7 = axes[3, 0]
    x_pos = np.arange(3)
    total_actions = len(model_actions_arr) if len(model_actions_arr) > 0 else 1
    counts = [buy_count, sell_count, hold_count]
    colors = ['#2ca02c', '#d62728', '#7f7f7f']
    bars = ax7.bar(x_pos, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(['Buy', 'Sell', 'Hold'], fontsize=11)
    ax7.set_title('Action Distribution', fontsize=13, fontweight='bold')
    ax7.set_ylabel('Count', fontsize=11)
    for bar, count in zip(bars, counts):
        pct = count / total_actions * 100 if total_actions > 0 else 0
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Metrics Summary Table
    ax8 = axes[3, 1]
    ax8.axis('off')

    # Calculate final metrics
    model_total_ret = model_cumret[-1] if len(model_cumret) > 0 else 0
    sma_total_ret = sma_cumret[-1] if len(sma_cumret) > 0 else 0
    bench_total_ret = bench_cumret[-1] if len(bench_cumret) > 0 else 0

    model_sharpe = model_total_ret / (np.std(model_returns[:n]) * np.sqrt(252) * 100 + 1e-9) if model_returns is not None else 0
    sma_sharpe = sma_total_ret / (np.std(sma_returns[:n]) * np.sqrt(252) * 100 + 1e-9) if sma_returns is not None else 0
    bench_sharpe = bench_total_ret / (np.std(bench_returns[:n]) * np.sqrt(252) * 100 + 1e-9) if bench_returns is not None else 0

    model_max_dd = -np.min(model_dd) if len(model_dd) > 0 else 0
    sma_max_dd = -np.min(sma_dd) if len(sma_dd) > 0 else 0
    bench_max_dd = -np.min(bench_dd) if len(bench_dd) > 0 else 0

    table_data = [
        ['Metric', 'DRL Model', f'SMA ({SMA_SHORT},{SMA_LONG})', 'Buy & Hold'],
        ['─' * 20, '─' * 15, '─' * 18, '─' * 15],
        ['Total Return', f'{model_total_ret:.2f}%', f'{sma_total_ret:.2f}%', f'{bench_total_ret:.2f}%'],
        ['Sharpe Ratio', f'{model_sharpe:.2f}', f'{sma_sharpe:.2f}', f'{bench_sharpe:.2f}'],
        ['Max Drawdown', f'{model_max_dd:.2f}%', f'{sma_max_dd:.2f}%', f'{bench_max_dd:.2f}%'],
        ['Buy Count', f'{buy_count}', '-', '-'],
        ['Sell Count', f'{sell_count}', '-', '-'],
    ]

    y_pos = 0.95
    ax8.text(0.5, y_pos, 'PERFORMANCE SUMMARY', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax8.transAxes)

    y_pos -= 0.08
    for row in table_data:
        if row[0] == '─' * 20:
            y_pos -= 0.02
            continue
        col_positions = [0.05, 0.35, 0.6, 0.82]
        for i, (text, x) in enumerate(zip(row, col_positions)):
            if i == 0:
                ax8.text(x, y_pos, text, fontsize=10, fontweight='bold',
                        ha='left', va='top', transform=ax8.transAxes)
            else:
                ax8.text(x, y_pos, text, fontsize=10,
                        ha='left', va='top', transform=ax8.transAxes)
        y_pos -= 0.06

    y_pos -= 0.05
    try:
        start_date = pd.to_datetime(dates[0]).strftime("%Y-%m-%d") if len(dates) > 0 else "N/A"
        end_date = pd.to_datetime(dates[-1]).strftime("%Y-%m-%d") if len(dates) > 0 else "N/A"
        ax8.text(0.5, y_pos, f'Trading Period: {start_date} to {end_date}',
                 ha='center', va='top', fontsize=9, style='italic', transform=ax8.transAxes)
    except:
        ax8.text(0.5, y_pos, 'Trading Period: N/A',
                 ha='center', va='top', fontsize=9, style='italic', transform=ax8.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{output_dir}/full_evaluation_plot.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Full evaluation plot saved to: {output_dir}/full_evaluation_plot.png")
