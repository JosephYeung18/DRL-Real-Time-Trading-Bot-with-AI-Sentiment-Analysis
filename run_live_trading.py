"""
TSLA DRL Real-Time Trading with Alpaca
=======================================
Daily trading frequency - one trade per day
Uses trained A2C model with paper trading.

Usage:
    python run_live_trading.py --duration 10
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import A2C

# Alpaca API configuration
API_KEY = "PKTK6WNUCCRVEGWLGINJI4RBYV"
API_SECRET = "2FeyGBBuF4K1yDBgKVqfDycZbWsJHRGzUhnSagzZbwPo"
TICKER = 'TSLA'
INITIAL_CAPITAL = 100000.0
TRANSACTION_FEE = 0.001  # 0.1%
ACTION_SMOOTHING = 0.8
POLL_INTERVAL = 60  # 60 seconds between checks

# Output directories
OUTPUT_DIR = "live_trading_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_trading_day():
    """Check if today is a US stock market trading day (Mon-Fri, not a holiday)"""
    today = datetime.now()
    weekday = today.weekday()

    # Weekend check
    if weekday >= 5:
        return False, "Weekend"

    # Simple holiday check (approximate)
    month = today.month
    day = today.day

    # Major US holidays (simplified)
    if month == 1 and day == 1:
        return False, "New Year's Day"
    if month == 12 and day == 25:
        return False, "Christmas Day"
    if month == 7 and day == 4:
        return False, "Independence Day"
    if month == 11 and day >= 22 and day <= 28 and weekday == 4:
        return False, "Thanksgiving"

    return True, "Trading day"


def run_trading(duration_minutes=10, model_path="trained_models/best/agent_a2c_best.zip"):
    """Run live trading session"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    print("=" * 80)
    print("TSLA DRL Real-Time Trading with Alpaca Paper Trading")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Symbol: {TICKER}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"API Key: {API_KEY[:8]}...")
    print("=" * 80)

    # Import Alpaca after setup
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    # Load model
    print("\n[1] Loading A2C model...")
    try:
        model = A2C.load(model_path)
        print(f"    Model loaded from: {model_path}")
    except Exception as e:
        print(f"    Failed to load model: {e}")
        return None

    # Initialize Alpaca clients
    print("\n[2] Connecting to Alpaca...")
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    # Get initial account state
    account = trading_client.get_account()
    starting_cash = float(account.equity)
    starting_buying_power = float(account.buying_power)

    print(f"    Account connected!")
    print(f"    Starting Equity: ${starting_cash:,.2f}")
    print(f"    Buying Power: ${starting_buying_power:,.2f}")

    # Clear any existing positions
    positions = trading_client.get_all_positions()
    if positions:
        print(f"\n    Clearing {len(positions)} existing positions...")
        for p in positions:
            order = MarketOrderRequest(
                symbol=p.symbol,
                qty=int(float(p.qty)),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
        time.sleep(3)

        # Refresh account after clearing
        account = trading_client.get_account()
        starting_cash = float(account.equity)

    # Trading state
    cash = starting_cash
    shares = 0
    last_signal = 0.5
    signal_history = []
    trades = []
    daily_trades = []  # Only one trade per day
    last_trade_date = None

    print("\n[3] Starting Trading Loop...")
    print("-" * 80)
    print(f"{'Time':<20} {'Price':<10} {'Signal':<10} {'Shares':<10} {'Value':<15} {'Action':<20}")
    print("-" * 80)

    # Main trading loop
    while time.time() - start_time < duration_minutes * 60:
        try:
            current_time = datetime.now()
            current_date = current_time.strftime('%Y-%m-%d')

            # Check trading day
            is_trading, reason = is_trading_day()
            if not is_trading:
                time.sleep(POLL_INTERVAL)
                continue

            # Get latest price
            quote = stock_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=[TICKER])
            )
            current_price = quote[TICKER].ask_price

            # Prepare observation (16-dim)
            obs = np.array([
                cash / INITIAL_CAPITAL,
                current_price / 500.0,
                shares / 300.0,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ], dtype=np.float32)

            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            raw_pos = float(np.clip(action[0] if hasattr(action, '__iter__') else action, 0, 1))

            # Apply action smoothing
            smoothed_pos = np.clip(
                ACTION_SMOOTHING * raw_pos + (1 - ACTION_SMOOTHING) * last_signal,
                0, 1
            )
            last_signal = smoothed_pos

            # Calculate target shares
            total_value = cash + shares * current_price
            target_shares = int((total_value * smoothed_pos) / current_price)
            shares_to_trade = target_shares - shares

            action_taken = "HOLD"
            trade_executed = False

            # Daily trading logic - only one trade per day
            if current_date != last_trade_date:
                if shares_to_trade > 0:
                    # Buy
                    buy_value = shares_to_trade * current_price
                    cost_with_fee = buy_value * (1 + TRANSACTION_FEE)

                    if cost_with_fee <= cash:
                        order = MarketOrderRequest(
                            symbol=TICKER,
                            qty=shares_to_trade,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        trading_client.submit_order(order)

                        cash -= cost_with_fee
                        shares += shares_to_trade

                        trade = {
                            'date': current_date,
                            'time': current_time.strftime('%H:%M:%S'),
                            'action': 'buy',
                            'quantity': shares_to_trade,
                            'price': current_price,
                            'cost': cost_with_fee,
                            'portfolio_value': total_value,
                            'signal': smoothed_pos
                        }
                        trades.append(trade)
                        daily_trades.append(trade)
                        action_taken = f"BUY {shares_to_trade}"
                        last_trade_date = current_date
                        trade_executed = True

                elif shares_to_trade < 0:
                    # Sell
                    sell_qty = min(abs(shares_to_trade), shares)
                    if sell_qty > 0:
                        proceeds = sell_qty * current_price * (1 - TRANSACTION_FEE)
                        order = MarketOrderRequest(
                            symbol=TICKER,
                            qty=sell_qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        trading_client.submit_order(order)

                        cash += proceeds
                        shares -= sell_qty

                        trade = {
                            'date': current_date,
                            'time': current_time.strftime('%H:%M:%S'),
                            'action': 'sell',
                            'quantity': sell_qty,
                            'price': current_price,
                            'proceeds': proceeds,
                            'portfolio_value': total_value,
                            'signal': smoothed_pos
                        }
                        trades.append(trade)
                        daily_trades.append(trade)
                        action_taken = f"SELL {sell_qty}"
                        last_trade_date = current_date
                        trade_executed = True

            # Update position tracking
            total_value = cash + shares * current_price

            # Record history
            signal_history.append({
                'timestamp': current_time,
                'date': current_date,
                'price': current_price,
                'raw_action': raw_pos,
                'smoothed_action': smoothed_pos,
                'portfolio_value': total_value,
                'cash': cash,
                'shares': shares,
                'action_taken': action_taken if not trade_executed else 'HOLD (same day)'
            })

            # Log
            time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"{time_str:<20} ${current_price:<9.2f} {smoothed_pos:<10.4f} {shares:<10} ${total_value:<14,.2f} {action_taken}")

            # Refresh account
            account = trading_client.get_account()

            # Wait for next interval
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(POLL_INTERVAL)

    print("-" * 80)
    print("Trading Completed!")

    # Close positions at end
    positions = trading_client.get_all_positions()
    if positions:
        print("\n[4] Closing positions...")
        for p in positions:
            order = MarketOrderRequest(
                symbol=p.symbol,
                qty=int(float(p.qty)),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            print(f"    Sold {p.qty} {p.symbol}")
        time.sleep(3)

    # Get final account state
    account = trading_client.get_account()
    final_equity = float(account.equity)
    final_cash = float(account.cash)

    return signal_history, trades, timestamp, starting_cash, final_equity, final_cash, account


def analyze_and_generate_report(signal_history, trades, timestamp, starting_cash, final_equity, final_cash, account):
    """Generate comprehensive analysis report"""

    if not signal_history:
        print("No data to analyze!")
        return

    df = pd.DataFrame(signal_history)

    # Calculate metrics
    initial_capital = INITIAL_CAPITAL
    total_return = (final_equity - initial_capital) / initial_capital * 100
    return_vs_start = (final_equity - starting_cash) / starting_cash * 100

    portfolio_values = df['portfolio_value'].values
    prices = df['price'].values
    shares_arr = df['shares'].values
    smoothed_actions = df['smoothed_action'].values
    raw_actions = df['raw_action'].values

    # Calculate returns
    if len(portfolio_values) > 1:
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    else:
        portfolio_returns = np.array([0])

    # Risk metrics
    volatility = np.std(portfolio_returns) * 100 if len(portfolio_returns) > 0 else 0
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)

    # Sharpe ratio (annualized)
    if std_return > 0:
        sharpe = (mean_return / std_return) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    max_drawdown_pct = 0
    max_drawdown_idx = 0
    for i, pv in enumerate(portfolio_values):
        if pv > peak:
            peak = pv
        dd = peak - pv
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd
            max_drawdown_pct = dd_pct
            max_drawdown_idx = i

    # Trading metrics
    buy_trades = [t for t in trades if t['action'] == 'buy']
    sell_trades = [t for t in trades if t['action'] == 'sell']
    total_trades = len(trades)

    # Price metrics
    price_change = (prices[-1] - prices[0]) / prices[0] * 100 if prices[-1] > 0 else 0
    price_start = prices[0]
    price_end = prices[-1]

    # Signal metrics
    avg_signal = np.mean(smoothed_actions)
    signal_std = np.std(smoothed_actions)
    signal_min = np.min(smoothed_actions)
    signal_max = np.max(smoothed_actions)

    # Duration
    start_dt = df['timestamp'].iloc[0]
    end_dt = df['timestamp'].iloc[-1]
    duration_min = (end_dt - start_dt).total_seconds() / 60
    duration_hours = duration_min / 60

    # Final state
    final_shares = shares_arr[-1]
    final_price = prices[-1]
    final_position_value = final_shares * final_price

    # Save detailed trade log CSV
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_csv_path = f"{OUTPUT_DIR}/trade_log_{timestamp}.csv"
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"Trade log saved to: {trades_csv_path}")

    # Save signal history CSV
    signal_csv_path = f"{OUTPUT_DIR}/signal_history_{timestamp}.csv"
    df.to_csv(signal_csv_path, index=False)
    print(f"Signal history saved to: {signal_csv_path}")

    # Generate comprehensive report
    report_path = f"{OUTPUT_DIR}/report_{timestamp}.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("           TSLA DRL REAL-TIME TRADING - COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trading Period: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration_min:.1f} minutes ({duration_hours:.2f} hours)\n")
        f.write(f"Data Points: {len(df)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("1. ACCOUNT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Initial Capital:':<25} ${initial_capital:>15,.2f}\n")
        f.write(f"{'Starting Cash:':<25} ${starting_cash:>15,.2f}\n")
        f.write(f"{'Final Equity:':<25} ${final_equity:>15,.2f}\n")
        f.write(f"{'Final Cash:':<25} ${final_cash:>15,.2f}\n")
        f.write(f"{'Position Value:':<25} ${final_position_value:>15,.2f}\n")
        f.write(f"{'Final Shares:':<25} {final_shares:>15}\n")
        f.write(f"{'Final Price:':<25} ${final_price:>15.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("2. PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Total Return:':<25} {total_return:>15.2f}%\n")
        f.write(f"{'Return vs Start Cash:':<25} {return_vs_start:>15.2f}%\n")
        f.write(f"{'Sharpe Ratio (annualized):':<25} {sharpe:>15.2f}\n")
        f.write(f"{'Max Drawdown:':<25} ${max_drawdown:>14,.2f} ({max_drawdown_pct:.2f}%)\n")
        f.write(f"{'Volatility (daily):':<25} {volatility:>15.4f}%\n")
        f.write(f"{'Mean Daily Return:':<25} {mean_return * 100:>15.4f}%\n")
        f.write(f"{'Std Daily Return:':<25} {std_return * 100:>15.4f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("3. TRADING ACTIVITY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Total Trades:':<25} {total_trades:>15}\n")
        f.write(f"{'Buy Orders:':<25} {len(buy_trades):>15}\n")
        f.write(f"{'Sell Orders:':<25} {len(sell_trades):>15}\n")
        f.write(f"{'Hold Periods:':<25} {len(df) - total_trades:>15}\n\n")

        f.write("=" * 80 + "\n")
        f.write("4. TRADE DETAILS\n")
        f.write("=" * 80 + "\n")

        if trades:
            f.write(f"{'Date':<12} {'Time':<10} {'Action':<6} {'Qty':<8} {'Price':<12} {'Value':<15} {'Signal':<10}\n")
            f.write("-" * 80 + "\n")
            for t in trades:
                value = t['quantity'] * t['price']
                f.write(f"{t['date']:<12} {t['time']:<10} {t['action']:<6} {t['quantity']:<8} "
                       f"${t['price']:<11.2f} ${value:<14,.2f} {t['signal']:<10.4f}\n")
        else:
            f.write("No trades executed during this session.\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("5. PRICE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Starting Price:':<25} ${price_start:>15.2f}\n")
        f.write(f"{'Ending Price:':<25} ${price_end:>15.2f}\n")
        f.write(f"{'Price Change:':<25} {price_change:>15.2f}%\n")
        f.write(f"{'Price Range:':<25} ${np.min(prices):.2f} - ${np.max(prices):.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("6. SIGNAL ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Mean Signal:':<25} {avg_signal:>15.4f}\n")
        f.write(f"{'Std Dev:':<25} {signal_std:>15.4f}\n")
        f.write(f"{'Min Signal:':<25} {signal_min:>15.4f}\n")
        f.write(f"{'Max Signal:':<25} {signal_max:>15.4f}\n")
        f.write(f"{'Signals > 0.5 (Long):':<25} {np.sum(smoothed_actions > 0.5):>15}\n")
        f.write(f"{'Signals < 0.5 (Cash):':<25} {np.sum(smoothed_actions < 0.5):>15}\n\n")

        f.write("=" * 80 + "\n")
        f.write("7. CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Ticker:':<25} {TICKER:>15}\n")
        f.write(f"{'Initial Capital:':<25} ${INITIAL_CAPITAL:>15,.2f}\n")
        f.write(f"{'Transaction Fee:':<25} {TRANSACTION_FEE * 100:>15.2f}%\n")
        f.write(f"{'Action Smoothing (alpha):':<25} {ACTION_SMOOTHING:>15}\n")
        f.write(f"{'Model:':<25} agent_a2c_best.zip\n")
        f.write(f"{'Trading Frequency:':<25} Daily\n\n")

        f.write("=" * 80 + "\n")
        f.write("8. SYSTEM INFORMATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Broker:':<25} Alpaca Markets\n")
        f.write(f"{'Trading Mode:':<25} Paper Trading\n")
        f.write(f"{'API Key:':<25} {API_KEY[:8]}...\n\n")

        f.write("=" * 80 + "\n")
        f.write("                              END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Comprehensive report saved to: {report_path}")

    # Generate visualization
    generate_visualization(df, trades, start_dt, end_dt, initial_capital, final_equity,
                          sharpe, max_drawdown, max_drawdown_pct, total_return,
                          len(buy_trades), len(sell_trades), timestamp)

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'total_trades': total_trades,
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades)
    }


def generate_visualization(df, trades, start_dt, end_dt, initial_capital, final_equity,
                          sharpe, max_drawdown, max_drawdown_pct, total_return,
                          buy_count, sell_count, timestamp):
    """Generate visualization charts"""

    print("\n[7] Generating Visualizations...")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'TSLA DRL Real-Time Trading Report\n{start_dt.strftime("%Y-%m-%d %H:%M")} - {end_dt.strftime("%Y-%m-%d %H:%M")}',
                fontsize=16, fontweight='bold')

    portfolio_values = df['portfolio_value'].values
    prices = df['price'].values
    shares_arr = df['shares'].values
    smoothed_actions = df['smoothed_action'].values

    # 1. Portfolio Value
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.plot(df['timestamp'], df['portfolio_value'], 'b-', linewidth=2)
    ax1.axhline(y=initial_capital, color='gray', ls='--', alpha=0.7)
    ax1.fill_between(df['timestamp'], initial_capital, df['portfolio_value'],
                     where=df['portfolio_value'] >= initial_capital, color='green', alpha=0.3)
    ax1.fill_between(df['timestamp'], initial_capital, df['portfolio_value'],
                     where=df['portfolio_value'] < initial_capital, color='red', alpha=0.3)
    ax1.set_title('Portfolio Value', fontweight='bold')
    ax1.set_ylabel('Value ($)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Price with Trades
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.plot(df['timestamp'], df['price'], 'k-', linewidth=1.5)
    for t in trades:
        trade_time = pd.to_datetime(f"{t['date']} {t['time']}")
        c = 'green' if t['action'] == 'buy' else 'red'
        m = '^' if t['action'] == 'buy' else 'v'
        ax2.scatter(trade_time, t['price'], c=c, marker=m, s=150, zorder=5, edgecolors='black')
    ax2.set_title('Price with Trades', fontweight='bold')
    ax2.set_ylabel('Price ($)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Trading Signals
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.plot(df['timestamp'], df['raw_action'], 'b-', alpha=0.5, linewidth=1, label='Raw')
    ax3.plot(df['timestamp'], df['smoothed_action'], 'r-', linewidth=2, label='Smoothed')
    ax3.axhline(y=0.5, color='gray', ls='--', alpha=0.7)
    ax3.fill_between(df['timestamp'], 0.5, df['smoothed_action'],
                    where=df['smoothed_action'] >= 0.5, color='green', alpha=0.2)
    ax3.set_title('Trading Signals', fontweight='bold')
    ax3.set_ylabel('Position Ratio')
    ax3.legend(loc='upper right')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Share Position
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.fill_between(df['timestamp'], 0, df['shares'], step='mid', color='blue', alpha=0.4)
    ax4.plot(df['timestamp'], df['shares'], 'b-', linewidth=2)
    ax4.set_title('Share Position', fontweight='bold')
    ax4.set_ylabel('Shares')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    # 5. Returns Distribution
    ax5 = fig.add_subplot(3, 4, 5)
    if len(portfolio_values) > 1:
        rets = np.diff(portfolio_values) / portfolio_values[:-1]
        rets_pct = rets * 100
        colors = ['green' if r >= 0 else 'red' for r in rets_pct]
        ax5.bar(range(len(rets_pct)), rets_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax5.axhline(y=0, color='black', linewidth=1)
        ax5.axhline(y=np.mean(rets_pct), color='blue', ls='--', alpha=0.7)
    ax5.set_title('Period Returns (%)', fontweight='bold')
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

    # 7. Cash vs Position Value
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.fill_between(df['timestamp'], 0, df['cash'].values, alpha=0.5, label='Cash', color='blue')
    position_values = df['shares'].values * df['price'].values
    ax7.fill_between(df['timestamp'], 0, position_values, alpha=0.5, label='Position', color='orange')
    ax7.set_title('Cash vs Position Value', fontweight='bold')
    ax7.set_ylabel('Value ($)')
    ax7.legend()
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3)

    # 8. Trade Pie Chart
    ax8 = fig.add_subplot(3, 4, 8)
    if buy_count > 0 or sell_count > 0:
        counts = [buy_count, sell_count]
        labels = ['Buy', 'Sell']
        colors_pie = ['#2ecc71', '#e74c3c']
        ax8.pie(counts, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
    ax8.set_title('Trade Distribution', fontweight='bold')

    # 9-12. Summary Box
    ax9 = fig.add_subplot(3, 4, (9, 12))
    ax9.axis('off')

    summary = f"""
╔════════════════════════════════════════════════════════════════════════════════════╗
║                        PERFORMANCE SUMMARY                                         ║
╠════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  SESSION INFORMATION                                                               ║
║  ──────────────────────────────────────────────────────────────────────────────    ║
║  Duration:           {duration_min:.1f} minutes ({len(df)} data points)                    ║
║  Symbol:             {TICKER}                                                               ║
║  Trading Frequency:  Daily                                                          ║
║                                                                                    ║
║  CAPITAL & RETURNS                                                                  ║
║  ──────────────────────────────────────────────────────────────────────────────    ║
║  Initial Capital:    ${initial_capital:>12,.2f}                                           ║
║  Final Equity:       ${final_equity:>12,.2f}                                           ║
║  Total Return:       {total_return:>12.2f}%                                               ║
║                                                                                    ║
║  RISK METRICS                                                                       ║
║  ──────────────────────────────────────────────────────────────────────────────    ║
║  Sharpe Ratio:       {sharpe:>12.2f}                                                   ║
║  Max Drawdown:       ${max_drawdown:>11,.2f} ({max_drawdown_pct:.2f}%)                          ║
║                                                                                    ║
║  TRADING ACTIVITY                                                                   ║
║  ──────────────────────────────────────────────────────────────────────────────    ║
║  Total Trades:        {buy_count + sell_count:>12}                                                   ║
║  Buy Orders:          {buy_count:>12}                                                   ║
║  Sell Orders:         {sell_count:>12}                                                   ║
║                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
"""

    ax9.text(0.02, 0.98, summary, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    plot_path = f"{OUTPUT_DIR}/trading_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Visualization saved to: {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TSLA DRL Real-Time Trading')
    parser.add_argument('--duration', type=int, default=10, help='Trading duration in minutes')
    parser.add_argument('--model', type=str, default='trained_models/best/agent_a2c_best.zip',
                        help='Path to trained model')
    args = parser.parse_args()

    # Run trading
    result = run_trading(duration_minutes=args.duration, model_path=args.model)

    if result:
        signal_history, trades, timestamp, starting_cash, final_equity, final_cash, account = result

        # Generate report
        metrics = analyze_and_generate_report(
            signal_history, trades, timestamp,
            starting_cash, final_equity, final_cash, account
        )

        print("\n" + "=" * 80)
        print("TRADING SESSION COMPLETE")
        print("=" * 80)
        print(f"Final Equity: ${final_equity:,.2f}")
        if metrics:
            print(f"Total Return: {metrics['total_return']:+.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
        else:
            print("Total Return: 0.00% (no trading data)")
            print("Sharpe Ratio: N/A (insufficient data)")
            print("Total Trades: 0")
        print("=" * 80)