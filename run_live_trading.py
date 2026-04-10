"""
Real-Time Paper Trading with Alpaca - 10 Minute Session
======================================================
Uses the new Alpaca paper trading account with trained A2C model.
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
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import A2C
from src.utils.config import TICKER, INITIAL_CAPITAL, TRANSACTION_FEE, ACTION_SMOOTHING

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Alpaca API Keys from environment variables
API_KEY = os.environ.get("ALPACA_API_KEY", "PKGUSBUYZ5DIJLVQ3ADSJ4UF6F")
API_SECRET = os.environ.get("ALPACA_API_SECRET", "AWTGa9Mb1tGspDLwZmHpoAwG9BC51KqMk1NejbXZFW7Q")
OUTPUT_DIR = "live_trading_results"
POLL_INTERVAL = 5  # seconds

def run_trading(duration_minutes=10):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("TSLA DRL Real-Time Paper Trading - 10 Minute Session")
    print("=" * 70)
    print(f"Start Time: {datetime.now()}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Symbol: {TICKER}")
    print(f"API Key: {API_KEY[:8]}...")
    print("=" * 70)

    # Load model
    print("\n[1] Loading A2C model...")
    model = A2C.load("trained_models/best/agent_a2c_best.zip")
    print("    Model loaded successfully!")

    # Initialize Alpaca clients
    print("\n[2] Connecting to Alpaca...")
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    # Get initial account state
    account = trading_client.get_account()
    starting_cash = float(account.equity)
    print(f"    Account connected!")
    print(f"    Starting Equity: ${starting_cash:,.2f}")
    print(f"    Buying Power: ${float(account.buying_power):,.2f}")

    # Clear any existing positions
    positions = trading_client.get_all_positions()
    if positions:
        print(f"\n    Clearing {len(positions)} existing positions...")
        for p in positions:
            order = MarketOrderRequest(
                symbol=p.symbol, qty=int(float(p.qty)),
                side=OrderSide.SELL, time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
        time.sleep(2)

    # Initialize state
    cash = starting_cash
    shares = 0
    last_exec_pos = 0.5
    signal_history = []
    trades = []
    start_time = time.time()
    current_price = 350.0  # Default price

    print("\n[3] Starting Trading Loop...")
    print("-" * 70)
    print(f"{'Time':<12} {'Price':<10} {'Signal':<10} {'Shares':<8} {'Portfolio':<15} {'Action'}")
    print("-" * 70)

    iteration = 0
    while time.time() - start_time < duration_minutes * 60:
        iteration += 1
        elapsed = int(time.time() - start_time)

        try:
            # Get latest price
            quote = stock_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=[TICKER])
            )
            current_price = quote[TICKER].ask_price
            current_time = datetime.now()

            # Prepare observation (16-dim): cash, price, shares, 13 indicators
            obs = np.array([
                cash / INITIAL_CAPITAL,  # 1: normalized cash
                current_price / 500.0,   # 2: normalized price
                shares / 300.0,          # 3: normalized shares
                # 13 technical indicators (default neutral values 0.5)
                0.5,  # macd
                0.5,  # boll_ub
                0.5,  # boll_lb
                0.5,  # rsi_30
                0.5,  # cci_30
                0.5,  # dx_30
                0.5,  # close_30_sma
                0.5,  # close_60_sma
                0.5,  # cci
                0.5,  # willr
                0.5,  # adx
                0.5,  # atr
                0.5   # sentiment_score
            ], dtype=np.float32)

            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            raw_pos = float(np.clip(action[0] if hasattr(action, '__iter__') else action, 0, 1))

            # Apply action smoothing
            smoothed_pos = np.clip(
                ACTION_SMOOTHING * raw_pos + (1 - ACTION_SMOOTHING) * last_exec_pos, 0, 1
            )

            # Calculate target shares
            total_value = cash + shares * current_price
            target_shares = int((total_value * smoothed_pos) / current_price)
            shares_to_trade = target_shares - shares

            action_taken = "HOLD"

            # Execute trade
            if shares_to_trade > 0:
                # Buy
                buy_value = shares_to_trade * current_price
                cost_with_fee = buy_value * (1 + TRANSACTION_FEE)

                if cost_with_fee <= cash:
                    order = MarketOrderRequest(
                        symbol=TICKER, qty=shares_to_trade,
                        side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                    )
                    trading_client.submit_order(order)
                    cash -= cost_with_fee
                    shares += shares_to_trade
                    trades.append({
                        'time': current_time, 'action': 'buy',
                        'shares': shares_to_trade, 'price': current_price
                    })
                    action_taken = f"BUY {shares_to_trade}"

            elif shares_to_trade < 0:
                # Sell
                sell_qty = min(abs(shares_to_trade), shares)
                if sell_qty > 0:
                    proceeds = sell_qty * current_price * (1 - TRANSACTION_FEE)
                    order = MarketOrderRequest(
                        symbol=TICKER, qty=sell_qty,
                        side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                    )
                    trading_client.submit_order(order)
                    cash += proceeds
                    shares -= sell_qty
                    trades.append({
                        'time': current_time, 'action': 'sell',
                        'shares': sell_qty, 'price': current_price
                    })
                    action_taken = f"SELL {sell_qty}"

            # Update position tracking
            total_value = cash + shares * current_price
            last_exec_pos = (shares * current_price) / total_value if total_value > 0 else 0

            # Record history
            signal_history.append({
                'time': current_time,
                'price': current_price,
                'raw_action': raw_pos,
                'smoothed_action': smoothed_pos,
                'portfolio_value': total_value,
                'cash': cash,
                'shares': shares
            })

            # Log
            time_str = current_time.strftime('%H:%M:%S')
            print(f"{time_str:<12} ${current_price:<9.2f} {smoothed_pos:<10.4f} {shares:<8} ${total_value:<14,.2f} {action_taken}")

            # Refresh account for accurate buying power
            account = trading_client.get_account()

        except Exception as e:
            # Still record data even on error
            total_value = cash + shares * current_price
            signal_history.append({
                'time': datetime.now(),
                'price': current_price,
                'raw_action': 0.5,
                'smoothed_action': last_exec_pos,
                'portfolio_value': total_value,
                'cash': cash,
                'shares': shares
            })
            print(f"Error at {datetime.now().strftime('%H:%M:%S')}: {e}")

        time.sleep(POLL_INTERVAL)

    print("-" * 70)
    print("Trading Completed!")

    # Final account state
    account = trading_client.get_account()
    final_equity = float(account.equity)

    # Close any open positions
    positions = trading_client.get_all_positions()
    if positions:
        print("\n[4] Closing positions...")
        for p in positions:
            order = MarketOrderRequest(
                symbol=p.symbol, qty=int(float(p.qty)),
                side=OrderSide.SELL, time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            print(f"    Sold {p.qty} {p.symbol}")
        time.sleep(2)
        account = trading_client.get_account()
        final_equity = float(account.equity)

    # Calculate metrics
    print("\n[5] Calculating Performance Metrics...")

    df = pd.DataFrame(signal_history)

    # Save CSV
    csv_path = f"{OUTPUT_DIR}/live_trading_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Data saved to: {csv_path}")

    return df, trades, timestamp, starting_cash, final_equity, account


def analyze_and_report(df, trades, timestamp, starting_cash, final_equity, account):
    """Generate comprehensive analysis and visualizations"""

    if len(df) == 0:
        print("No data to analyze!")
        return

    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Basic metrics
    initial_capital = 100000.0
    total_return = (final_equity - initial_capital) / initial_capital * 100
    return_vs_start = (final_equity - starting_cash) / starting_cash * 100

    portfolio_values = df['portfolio_value'].values
    prices = df['price'].values
    shares_arr = df['shares'].values
    smoothed_actions = df['smoothed_action'].values
    raw_actions = df['raw_action'].values

    # Returns
    if len(portfolio_values) > 1:
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    else:
        portfolio_returns = np.array([0])

    # Risk metrics
    volatility = np.std(portfolio_returns) * 100 if len(portfolio_returns) > 0 else 0
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252 * 78) if np.std(portfolio_returns) > 0 else 0

    # Max drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    max_drawdown_pct = 0
    for pv in portfolio_values:
        if pv > peak:
            peak = pv
        dd = peak - pv
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd
            max_drawdown_pct = dd_pct

    # Trading metrics
    buy_trades = [t for t in trades if t['action'] == 'buy']
    sell_trades = [t for t in trades if t['action'] == 'sell']
    total_trades = len(trades)

    # Price metrics
    price_change = (prices[-1] - prices[0]) / prices[0] * 100 if prices[-1] > 0 else 0

    # Signal metrics
    avg_signal = np.mean(smoothed_actions)
    signal_std = np.std(smoothed_actions)
    signal_min = np.min(smoothed_actions)
    signal_max = np.max(smoothed_actions)

    # Duration
    duration_min = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 60

    # Final state
    final_shares = shares_arr[-1]
    final_price = prices[-1]

    # Print summary
    print(f"""
Session Summary:
---------------
  Duration:           {duration_min:.1f} minutes ({len(df)} data points)
  Starting Cash:      ${starting_cash:,.2f}
  Initial Capital:    ${initial_capital:,.2f}
  Final Equity:      ${final_equity:,.2f}

Performance:
-----------
  Total Return:      {total_return:+.2f}%
  Return vs Start:   {return_vs_start:+.2f}%
  Sharpe Ratio:      {sharpe:.2f}
  Max Drawdown:      ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)
  Volatility:        {volatility:.4f}%

Trading Activity:
----------------
  Total Trades:      {total_trades}
    Buy Orders:      {len(buy_trades)}
    Sell Orders:     {len(sell_trades)}

Position:
---------
  Final Shares:      {final_shares}
  Final Price:       ${final_price:.2f}
  Final Value:       ${final_shares * final_price:,.2f}

Signal Statistics:
------------------
  Mean Signal:       {avg_signal:.4f}
  Std Dev:           {signal_std:.4f}
  Min Signal:        {signal_min:.4f}
  Max Signal:        {signal_max:.4f}
""")

    # Create visualizations
    print("\n[6] Generating Visualizations...")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'TSLA DRL Real-Time Trading Report\n{datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16, fontweight='bold')

    # 1. Portfolio Value
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.plot(df['time'], df['portfolio_value'], 'b-', linewidth=2)
    ax1.axhline(y=initial_capital, color='gray', ls='--', alpha=0.7)
    ax1.fill_between(df['time'], initial_capital, df['portfolio_value'],
                      where=df['portfolio_value'] >= initial_capital, color='green', alpha=0.3)
    ax1.fill_between(df['time'], initial_capital, df['portfolio_value'],
                      where=df['portfolio_value'] < initial_capital, color='red', alpha=0.3)
    ax1.set_title('Portfolio Value', fontweight='bold')
    ax1.set_ylabel('Value ($)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Price with Trades
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.plot(df['time'], df['price'], 'k-', linewidth=1.5)
    for t in trades:
        c = 'green' if t['action'] == 'buy' else 'red'
        m = '^' if t['action'] == 'buy' else 'v'
        ax2.scatter(t['time'], t['price'], c=c, marker=m, s=150, zorder=5, edgecolors='black')
    ax2.set_title('Price with Trades', fontweight='bold')
    ax2.set_ylabel('Price ($)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Trading Signals
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.plot(df['time'], df['raw_action'], 'b-', alpha=0.5, linewidth=1, label='Raw')
    ax3.plot(df['time'], df['smoothed_action'], 'r-', linewidth=2, label='Smoothed')
    ax3.axhline(y=0.5, color='gray', ls='--', alpha=0.7)
    ax3.fill_between(df['time'], 0.5, df['smoothed_action'],
                      where=df['smoothed_action'] >= 0.5, color='green', alpha=0.2)
    ax3.set_title('Trading Signals', fontweight='bold')
    ax3.set_ylabel('Position Ratio')
    ax3.legend(loc='upper right')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Share Position
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.fill_between(df['time'], 0, df['shares'], step='mid', color='blue', alpha=0.4)
    ax4.plot(df['time'], df['shares'], 'b-', linewidth=2)
    for t in trades:
        offset = 5 if t['action'] == 'buy' else -10
        ax4.annotate(f"{t['action'].upper()}\n{t['shares']}",
                     xy=(t['time'], t['shares']),
                     xytext=(t['time'], t['shares'] + offset),
                     fontsize=7, ha='center', color='green' if t['action'] == 'buy' else 'red')
    ax4.set_title('Share Position', fontweight='bold')
    ax4.set_ylabel('Shares')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    # 5. Returns Distribution
    ax5 = fig.add_subplot(3, 4, 5)
    rets_pct = portfolio_returns * 100
    colors = ['green' if r >= 0 else 'red' for r in rets_pct]
    ax5.bar(range(len(rets_pct)), rets_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.axhline(y=0, color='black', linewidth=1)
    ax5.axhline(y=np.mean(rets_pct), color='blue', ls='--', alpha=0.7)
    ax5.set_title('Period Returns', fontweight='bold')
    ax5.set_ylabel('Return (%)')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Signal Histogram
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.hist(df['smoothed_action'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax6.axvline(x=np.mean(df['smoothed_action']), color='red', ls='--', linewidth=2)
    ax6.axvline(x=0.5, color='gray', ls=':', linewidth=2)
    ax6.set_title('Signal Distribution', fontweight='bold')
    ax6.set_xlabel('Smoothed Action')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Price vs Portfolio
    ax7 = fig.add_subplot(3, 4, 7)
    price_norm = (prices - prices.min()) / (prices.max() - prices.min() + 0.001)
    port_norm = (portfolio_values - portfolio_values.min()) / (portfolio_values.max() - portfolio_values.min() + 0.001)
    ax7.plot(df['time'], price_norm, 'b-', linewidth=1.5, label='Price')
    ax7.plot(df['time'], port_norm, 'r-', linewidth=1.5, label='Portfolio')
    ax7.set_title('Price vs Portfolio (Normalized)', fontweight='bold')
    ax7.legend()
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3)

    # 8. Trade Pie Chart
    ax8 = fig.add_subplot(3, 4, 8)
    if trades:
        counts = [len(buy_trades), len(sell_trades)]
        labels = ['Buy', 'Sell']
        colors_pie = ['#2ecc71', '#e74c3c']
        ax8.pie(counts, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
    ax8.set_title('Trade Distribution', fontweight='bold')

    # 9. Cumulative Returns
    ax9 = fig.add_subplot(3, 4, 9)
    if len(cum_returns := (1 + portfolio_returns).cumprod() - 1) > 0:
        ax9.plot(range(len(cum_returns)), cum_returns * 100, 'b-', linewidth=2)
        ax9.axhline(y=0, color='gray', ls='--')
        ax9.fill_between(range(len(cum_returns)), 0, cum_returns * 100,
                          where=cum_returns >= 0, color='green', alpha=0.3)
        ax9.fill_between(range(len(cum_returns)), 0, cum_returns * 100,
                          where=cum_returns < 0, color='red', alpha=0.3)
    ax9.set_title('Cumulative Returns', fontweight='bold')
    ax9.set_ylabel('Cumulative Return (%)')
    ax9.grid(True, alpha=0.3)

    # 10. Cash vs Position Value
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.fill_between(df['time'], 0, df['cash'].values, alpha=0.5, label='Cash', color='blue')
    ax10.fill_between(df['time'], 0, df['shares'].values * df['price'].values, alpha=0.5, label='Position', color='orange')
    ax10.set_title('Cash vs Position Value', fontweight='bold')
    ax10.set_ylabel('Value ($)')
    ax10.legend()
    ax10.tick_params(axis='x', rotation=45)
    ax10.grid(True, alpha=0.3)

    # 11-12. Summary Box
    ax11 = fig.add_subplot(3, 4, (11, 12))
    ax11.axis('off')

    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    PERFORMANCE SUMMARY                                               ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                        ║
║  SESSION INFORMATION                                                                                   ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────   ║
║  Duration:           {duration_min:>8.1f} minutes                                              ║
║  Data Points:        {len(df):>8}                                                                    ║
║  Symbol:             {TICKER:>8}                                                                    ║
║                                                                                                        ║
║  CAPITAL & RETURNS                                                                                     ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────   ║
║  Initial Capital:    ${initial_capital:>12,.2f}                                                       ║
║  Starting Cash:      ${starting_cash:>12,.2f}                                                       ║
║  Final Equity:       ${final_equity:>12,.2f}                                                       ║
║  Total Return:       {total_return:>12,.2f}%                                                           ║
║                                                                                                        ║
║  RISK METRICS                                                                                          ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────   ║
║  Sharpe Ratio:       {sharpe:>12,.2f}                                                                   ║
║  Max Drawdown:       ${max_drawdown:>12,.2f} ({max_drawdown_pct:.2f}%)                                          ║
║  Volatility:         {volatility:>12,.4f}%                                                               ║
║                                                                                                        ║
║  TRADING ACTIVITY                                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────   ║
║  Total Trades:       {total_trades:>8}                                                                    ║
║  Buy Orders:         {len(buy_trades):>8}                                                                    ║
║  Sell Orders:        {len(sell_trades):>8}                                                                    ║
║                                                                                                        ║
║  POSITION & SIGNAL                                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────   ║
║  Final Shares:       {final_shares:>8}                                                                    ║
║  Avg Signal:         {avg_signal:>8.4f}                                                                    ║
║  Signal Range:       {signal_min:.4f} - {signal_max:.4f}                                                           ║
║                                                                                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

    ax11.text(0.02, 0.98, summary, transform=ax11.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = f"{OUTPUT_DIR}/live_trading_{timestamp}_report.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"    Visualization saved to: {png_path}")

    pdf_path = f"{OUTPUT_DIR}/live_trading_{timestamp}_report.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"    PDF saved to: {pdf_path}")

    plt.close()

    # Generate text report
    report_path = f"{OUTPUT_DIR}/live_trading_{timestamp}_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("        TSLA DRL REAL-TIME PAPER TRADING SESSION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Duration: {duration_min:.1f} minutes\n\n")

        f.write("-" * 80 + "\n")
        f.write("1. ACCOUNT SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Initial Capital:    ${initial_capital:,.2f}\n")
        f.write(f"  Starting Cash:      ${starting_cash:,.2f}\n")
        f.write(f"  Final Equity:       ${final_equity:,.2f}\n")
        f.write(f"  Total Return:       {total_return:+.2f}%\n\n")

        f.write("-" * 80 + "\n")
        f.write("2. PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Sharpe Ratio:       {sharpe:.2f}\n")
        f.write(f"  Max Drawdown:       ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)\n")
        f.write(f"  Volatility:         {volatility:.4f}%\n\n")

        f.write("-" * 80 + "\n")
        f.write("3. TRADING ACTIVITY\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Trades:      {total_trades}\n")
        f.write(f"  Buy Orders:         {len(buy_trades)}\n")
        f.write(f"  Sell Orders:        {len(sell_trades)}\n\n")

        f.write("  Trade History:\n")
        for i, t in enumerate(trades, 1):
            f.write(f"    #{i} {t['time'].strftime('%H:%M:%S')}  {t['action'].upper():4}  {t['shares']:4} shares @ ${t['price']:.2f}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("4. SIGNAL ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean Signal:        {avg_signal:.4f}\n")
        f.write(f"  Std Dev:            {signal_std:.4f}\n")
        f.write(f"  Min Signal:         {signal_min:.4f}\n")
        f.write(f"  Max Signal:         {signal_max:.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("5. POSITION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Final Shares:       {final_shares}\n")
        f.write(f"  Final Price:        ${final_price:.2f}\n")
        f.write(f"  Position Value:     ${final_shares * final_price:,.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("                              END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"    Report saved to: {report_path}")

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'volatility': volatility,
        'total_trades': total_trades,
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'final_equity': final_equity
    }


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run trading session
    df, trades, timestamp, starting_cash, final_equity, account = run_trading(duration_minutes=10)

    # Analyze and generate reports
    metrics = analyze_and_report(df, trades, timestamp, starting_cash, final_equity, account)

    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Return: {metrics['total_return']:+.2f}%")
    print(f"Trades: {metrics['total_trades']}")
    print("=" * 70)