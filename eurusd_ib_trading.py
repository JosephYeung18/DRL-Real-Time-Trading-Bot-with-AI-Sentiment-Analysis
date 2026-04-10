"""
EURUSD DRL Real-Time Trading with Interactive Brokers (ib_insync)
================================================================
Uses trained A2C models for EURUSD forex trading via IB API.

Requirements:
    - Interactive Brokers TWS or IB Gateway running
    - ib_insync package (pip install ib_insync)
    - Valid IB account with forex trading permissions

Configure IB Connection:
    - TWS: File > Settings > API > Enable ActiveX and Socket Clients
    - Socket Port: 7496 (paper trading) or 7497 (live)
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
import talib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ib_insync import IB, Forex, MarketOrder
from stable_baselines3 import A2C

# ========== Configuration ==========
TICKER = 'EURUSD'
INITIAL_CAPITAL = 100000
TRANSACTION_FEE = 0.0001
ACTION_SMOOTHING = 0.5
POLL_INTERVAL = 60  # seconds between trading decisions

# IB Connection Settings
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # 7496 for paper trading, 7497 for live
IB_CLIENT_ID = 1

# Technical indicators
INDICATORS_FOREX = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                    'close_30_sma', 'close_60_sma']

# Output directory
OUTPUT_DIR = 'eurusd_ib_trading_result'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class EURUSDIBTrader:
    """EURUSD trading with IB (ib_insync) and DRL model"""

    def __init__(self, model_path, with_sentiment=False, duration_minutes=10):
        self.model_path = model_path
        self.with_sentiment = with_sentiment
        self.duration_minutes = duration_minutes
        self.ib = None
        self.model = None
        self.contract = None

        # Trading state
        self.cash = INITIAL_CAPITAL
        self.shares = 0  # Positive = long units, negative = short units
        self.last_action = 0.0
        self.signal_history = []
        self.trades = []
        self.start_time = None
        self.iteration = 0
        self.current_price = 0

    def prepare_data(self):
        """Prepare EURUSD data with technical indicators"""
        print("\n" + "=" * 60)
        print("EURUSD DRL Trading with Interactive Brokers (ib_insync)")
        print("=" * 60)

        df_raw = pd.read_csv('data/EURUSD.csv')
        print(f"   Historical data: {len(df_raw)} rows")

        df = pd.DataFrame()
        df['date'] = pd.to_datetime(df_raw['trade_date'])
        df['close'] = (df_raw['bid_close'] + df_raw['ask_close']) / 2
        df['high'] = (df_raw['bid_high'] + df_raw['ask_high']) / 2
        df['low'] = (df_raw['bid_low'] + df_raw['ask_low']) / 2
        df['open'] = (df_raw['bid_open'] + df_raw['ask_open']) / 2
        df['volume'] = df_raw['tick_qty']
        df['tic'] = 'EURUSD'
        df['day'] = df['date'].dt.weekday

        df = df.sort_values('date').reset_index(drop=True)

        # Add technical indicators
        from finrl.meta.preprocessor.preprocessors import FeatureEngineer
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

        # Load sentiment if enabled
        if self.with_sentiment:
            try:
                df_sen = pd.read_csv('data/EURUSD_SEN.csv', on_bad_lines='skip')
                df_sen['date'] = pd.to_datetime(df_sen['date'], errors='coerce')
                df_sen['sentiment_score'] = pd.to_numeric(df_sen['sentiment_score'], errors='coerce')
                df_sen = df_sen.dropna(subset=['sentiment_score'])
                processed = processed.merge(df_sen[['date', 'sentiment_score']], on='date', how='left')
                processed['sentiment_score'] = processed['sentiment_score'].fillna(0.0)
            except:
                processed['sentiment_score'] = 0.0

        processed = processed.fillna(0)
        return processed

    def calculate_indicators_for_state(self, data_row, with_sentiment=False):
        """Calculate indicators for model state"""
        indicators = INDICATORS_FOREX + ['atr', 'adx', 'willr', 'cci']
        if with_sentiment:
            indicators = indicators + ['sentiment_score']

        state = []
        for ind in indicators:
            val = data_row[ind] if ind in data_row else 0
            if ind == 'macd':
                state.append(np.clip(val / 0.01, -1, 1))
            elif ind == 'rsi_30':
                state.append(val / 100.0)
            elif ind in ['cci_30', 'cci']:
                state.append(np.clip(val / 100, -1, 1))
            elif ind in ['dx_30', 'adx']:
                state.append(val / 100.0)
            elif ind in ['boll_ub', 'boll_lb']:
                state.append(np.clip((val - data_row['close']) / data_row['close'] * 100, -1, 1))
            elif ind in ['close_30_sma', 'close_60_sma']:
                state.append(np.clip((data_row['close'] - val) / val, -1, 1))
            elif ind in ['atr', 'willr']:
                state.append(np.clip(val / data_row['close'], -1, 1))
            elif ind == 'sentiment_score':
                state.append(np.clip(val, -1, 1))
            else:
                state.append(0)
        return np.array(state, dtype=np.float32)

    def load_model(self):
        """Load trained A2C model"""
        print(f"\n[1] Loading A2C model from {self.model_path}...")
        try:
            self.model = A2C.load(self.model_path)
            print("    Model loaded successfully!")
            return True
        except Exception as e:
            print(f"    Failed to load model: {e}")
            return False

    def connect_to_ib(self):
        """Connect to Interactive Brokers via ib_insync"""
        print(f"\n[2] Connecting to Interactive Brokers at {IB_HOST}:{IB_PORT}...")

        self.ib = IB()

        try:
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
            print("    Connected to IB!")

            # Create EURUSD forex contract
            self.contract = Forex('EURUSD')
            print(f"    Contract: {self.contract.symbol} {self.contract.currency}")

            # Request market data
            ticker = self.ib.reqMktDepth(self.contract, numRows=5)
            self.ib.sleep(1)

            if ticker:
                print(f"    Market data connected")
            else:
                print("    Warning: Could not get market depth data")

            return True

        except Exception as e:
            print(f"    Connection failed: {e}")
            print("    Make sure TWS or IB Gateway is running with API enabled")
            return False

    def get_current_price(self):
        """Get current EURUSD price"""
        if self.ib is None or not self.ib.isConnected():
            return {'bid': 0, 'ask': 0, 'mid': 0}

        try:
            # Request tick data
            ticker = self.ib.reqTickByTickData(self.contract, 'BidAsk')
            self.ib.sleep(0.5)

            if ticker.bid and ticker.ask:
                bid = ticker.bid
                ask = ticker.ask
                mid = (bid + ask) / 2
                self.current_price = mid
                return {'bid': bid, 'ask': ask, 'mid': mid}
            else:
                # Try last price
                ticker = self.ib.reqTickByTickData(self.contract, 'Last')
                self.ib.sleep(0.5)
                if ticker.last:
                    return {'bid': ticker.last, 'ask': ticker.last, 'mid': ticker.last}

        except Exception as e:
            print(f"    Error getting price: {e}")

        return {'bid': 0, 'ask': 0, 'mid': 0}

    def execute_trade(self, action, target_shares):
        """Execute forex trade via IB"""
        if self.ib is None or not self.ib.isConnected():
            return False

        current_shares = self.shares
        shares_to_trade = target_shares - current_shares

        try:
            if shares_to_trade > 0:
                # Buy
                order = MarketOrder('BUY', shares_to_trade)
                trade = self.ib.placeOrder(self.contract, order)
                self.ib.sleep(1)  # Wait longer for fill

                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice or self.current_price
                    cost = shares_to_trade * fill_price * (1 + TRANSACTION_FEE)
                    self.cash -= cost
                    self.shares += shares_to_trade
                    self.trades.append({
                        'time': datetime.now(),
                        'action': 'buy',
                        'shares': shares_to_trade,
                        'price': fill_price,
                        'cost': cost
                    })
                    print(f"    >>> FILLED: BUY {shares_to_trade} @ {fill_price:.5f}")
                    return True

            elif shares_to_trade < 0:
                # Sell/Short
                order = MarketOrder('SELL', abs(shares_to_trade))
                trade = self.ib.placeOrder(self.contract, order)
                self.ib.sleep(1)  # Wait longer for fill

                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice or self.current_price
                    proceeds = abs(shares_to_trade) * fill_price * (1 - TRANSACTION_FEE)
                    self.cash += proceeds
                    self.shares += shares_to_trade  # shares_to_trade is negative
                    self.trades.append({
                        'time': datetime.now(),
                        'action': 'short',
                        'shares': abs(shares_to_trade),
                        'price': fill_price,
                        'proceeds': proceeds
                    })
                    print(f"    >>> FILLED: SELL/SHORT {abs(shares_to_trade)} @ {fill_price:.5f}")
                    return True

        except Exception as e:
            print(f"    Trade execution error: {e}")

        return False

    def run_trading_loop(self, trade_data):
        """Main trading loop with real IB market data"""
        print(f"\n[3] Starting Live Trading...")
        print(f"    Duration: {self.duration_minutes} minutes")
        print(f"    Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"    Polling Interval: {POLL_INTERVAL} seconds")

        self.start_time = time.time()
        latest_data = trade_data.iloc[-1]

        print("\n" + "-" * 85)
        print(f"{'Time':<12} {'Elapsed':<10} {'Bid':<10} {'Ask':<10} {'Signal':<10} {'Shares':<10} {'Portfolio':<15}")
        print("-" * 85)

        while time.time() - self.start_time < self.duration_minutes * 60:
            self.iteration += 1
            elapsed_seconds = int(time.time() - self.start_time)
            elapsed_str = f"{elapsed_seconds // 60}m {elapsed_seconds % 60}s"

            # Get live market data from IB
            price_data = self.get_current_price()
            current_price = price_data['mid']

            if current_price <= 0:
                time_str = datetime.now().strftime('%H:%M:%S')
                print(f"{time_str:<12} {elapsed_str:<10} Waiting for market data...")
                time.sleep(POLL_INTERVAL)
                continue

            # Calculate model state
            indicators = self.calculate_indicators_for_state(latest_data, self.with_sentiment)
            cash_norm = self.cash / INITIAL_CAPITAL
            shares_norm = self.shares / 1000.0
            price_norm = current_price / 1.5

            state = np.array([cash_norm, price_norm, shares_norm] + list(indicators), dtype=np.float32)

            # Validate state - replace any NaN/inf
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

            # Get model prediction
            action, _ = self.model.predict(state, deterministic=True)
            raw_action = float(action[0]) if hasattr(action, '__iter__') else float(action)

            # Handle NaN in action
            if np.isnan(raw_action) or np.isinf(raw_action):
                raw_action = 0.0
                smoothed_action = self.last_action  # Keep previous action
            else:
                # Apply action smoothing
                smoothed_action = ACTION_SMOOTHING * raw_action + (1 - ACTION_SMOOTHING) * self.last_action

            self.last_action = smoothed_action

            # Validate smoothed_action
            if np.isnan(smoothed_action) or np.isinf(smoothed_action):
                smoothed_action = 0.0

            # Calculate target position
            target_ratio = float(smoothed_action)
            max_position = INITIAL_CAPITAL * 2 / current_price  # 2x leverage

            # Handle NaN/inf
            if np.isnan(target_ratio) or np.isinf(target_ratio):
                target_ratio = 0.0
            if np.isnan(max_position) or np.isinf(max_position):
                max_position = 0.0

            target_shares = int(target_ratio * max_position)

            # Execute trade if position difference is significant
            trade_executed = False
            if abs(target_shares - self.shares) >= 1000:  # Minimum forex unit threshold
                trade_executed = self.execute_trade(smoothed_action, target_shares)

            # Sync with actual IB positions (authoritative source)
            ib_position = 0
            positions = self.ib.positions()
            for pos in positions:
                if pos.contract.symbol == 'EUR' and pos.contract.currency == 'USD':
                    ib_position = pos.position
                    break

            # Only update from IB if trade was not executed (to avoid overwriting during failed orders)
            if not trade_executed:
                self.shares = ib_position

            # Calculate portfolio value using IB positions
            # For long: cash + shares * price
            # For short: cash + shares * price (shares negative, so subtracts)
            total_value = self.cash + self.shares * current_price

            # Record history
            self.signal_history.append({
                'time': datetime.now(),
                'elapsed': elapsed_str,
                'bid': price_data['bid'],
                'ask': price_data['ask'],
                'mid': current_price,
                'signal': smoothed_action,
                'raw_action': raw_action,
                'shares': self.shares,
                'portfolio_value': total_value,
                'cash': self.cash
            })

            # Log
            time_str = datetime.now().strftime('%H:%M:%S')
            print(f"{time_str:<12} {elapsed_str:<10} {price_data['bid']:<10.5f} {price_data['ask']:<10.5f} "
                  f"{smoothed_action:<10.4f} {self.shares:<10} ${total_value:<14,.2f}")

            # Wait for next interval
            time.sleep(POLL_INTERVAL)

        print("-" * 85)
        print(f"\n    Trading loop completed")
        print(f"    Total iterations: {self.iteration}")
        print(f"    Trades executed: {len(self.trades)}")

    def close_position(self):
        """Close any open position"""
        if self.shares != 0:
            print(f"\n[4] Closing position: {self.shares} units")
            if self.ib and self.ib.isConnected():
                try:
                    action = 'SELL' if self.shares > 0 else 'BUY'
                    order = MarketOrder(action, abs(self.shares))
                    trade = self.ib.placeOrder(self.contract, order)
                    self.ib.sleep(1)

                    if trade.orderStatus.status == 'Filled':
                        fill_price = trade.orderStatus.avgFillPrice or self.current_price
                        if self.shares > 0:  # Closing long
                            proceeds = self.shares * fill_price * (1 - TRANSACTION_FEE)
                            self.cash += proceeds
                            print(f"    >>> CLOSED LONG: Sold {self.shares} @ {fill_price:.5f}")
                        else:  # Closing short
                            cost = abs(self.shares) * fill_price * (1 + TRANSACTION_FEE)
                            self.cash -= cost
                            print(f"    >>> CLOSED SHORT: Covered {abs(self.shares)} @ {fill_price:.5f}")
                        self.shares = 0
                    else:
                        print(f"    Position close status: {trade.orderStatus.status}")
                except Exception as e:
                    print(f"    Error closing position: {e}")

    def generate_report(self):
        """Generate trading report"""
        print("\n[5] Generating Report...")

        final_value = self.signal_history[-1]['portfolio_value'] if self.signal_history else INITIAL_CAPITAL
        total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        # Calculate metrics
        returns = []
        portfolio_values = [INITIAL_CAPITAL]
        for i in range(1, len(self.signal_history)):
            portfolio_values.append(self.signal_history[i]['portfolio_value'])
            prev_val = self.signal_history[i-1]['portfolio_value']
            curr_val = self.signal_history[i]['portfolio_value']
            if prev_val > 0:
                ret = (curr_val - prev_val) / prev_val
                returns.append(ret)

        if returns and np.std(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            # Annualize for 10-min data (assume 252 trading days, 24 hours, 6 10-min periods per hour)
            sharpe = (mean_return / std_return) * np.sqrt(252 * 24 * 6) if std_return > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        peak = INITIAL_CAPITAL
        max_drawdown = 0
        max_drawdown_pct = 0
        for val in portfolio_values:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_drawdown:
                max_drawdown = dd
                max_drawdown_pct = dd / peak * 100 if peak > 0 else 0

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(OUTPUT_DIR, f'report_ib_{timestamp}.txt')
        csv_file = os.path.join(OUTPUT_DIR, f'trading_data_ib_{timestamp}.csv')

        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EURUSD DRL TRADING WITH INTERACTIVE BROKERS - REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Duration: {self.duration_minutes} minutes\n")
            f.write(f"IB Connection: {'Connected' if self.ib and self.ib.isConnected() else 'Disconnected'}\n\n")
            f.write("-" * 70 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Initial Capital:     ${INITIAL_CAPITAL:>15,.2f}\n")
            f.write(f"  Final Value:         ${final_value:>15,.2f}\n")
            f.write(f"  Total Return:         {total_return:>15.2f}%\n")
            f.write(f"  Sharpe Ratio:        {sharpe:>15.2f}\n")
            f.write(f"  Max Drawdown:        ${max_drawdown:>14,.2f} ({max_drawdown_pct:.2f}%)\n")
            f.write(f"  Total Trades:        {len(self.trades):>15}\n\n")
            f.write("-" * 70 + "\n")
            f.write("TRADE HISTORY (ACTUAL FILLS)\n")
            f.write("-" * 70 + "\n")

            if self.trades:
                for t in self.trades:
                    f.write(f"  {t['time'].strftime('%Y-%m-%d %H:%M:%S')}  {t['action']:<6} {t['shares']:>10.0f} units @ ${t['price']:.5f}\n")
                    if 'cost' in t:
                        f.write(f"    Cost: ${t['cost']:,.2f}\n")
                    if 'proceeds' in t:
                        f.write(f"    Proceeds: ${t['proceeds']:,.2f}\n")
            else:
                f.write("  No trades executed\n")

            f.write("\n" + "=" * 70 + "\n")

        # Save CSV
        if self.signal_history:
            df = pd.DataFrame(self.signal_history)
            df.to_csv(csv_file, index=False)

        print(f"    Report saved: {report_file}")
        print(f"    Data saved: {csv_file}")

        return {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe': sharpe,
            'trades': len(self.trades)
        }

    def run(self):
        """Main execution"""
        sentiment_str = "With Sentiment" if self.with_sentiment else "Without Sentiment"
        print(f"\n{'='*60}")
        print(f"EURUSD IB Trading - {sentiment_str}")
        print(f"{'='*60}")

        # Prepare data and load model
        trade_data = self.prepare_data()
        if not self.load_model():
            return None

        # Connect to IB
        if not self.connect_to_ib():
            print("Could not connect to IB. Exiting.")
            return None

        # Run trading loop
        self.run_trading_loop(trade_data)

        # Close position
        self.close_position()

        # Generate report
        results = self.generate_report()

        # Disconnect
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            print("\n    Disconnected from IB")

        return results


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='EURUSD DRL Trading with Interactive Brokers')
    parser.add_argument('--model', type=str, default='trained_models/eurusd/agent_a2c_eurusd.zip',
                        help='Path to trained model')
    parser.add_argument('--sentiment', action='store_true', help='Use sentiment enhancement')
    parser.add_argument('--duration', type=int, default=10, help='Trading duration in minutes')
    parser.add_argument('--paper', action='store_true', help='Use paper trading port (7496)')

    args = parser.parse_args()

    # Override port for paper trading
    global IB_PORT
    if args.paper:
        IB_PORT = 7496
        print("Using PAPER TRADING port 7496")
    else:
        IB_PORT = 7497
        print("Using LIVE TRADING port 7497")

    # Create trader
    trader = EURUSDIBTrader(
        model_path=args.model,
        with_sentiment=args.sentiment,
        duration_minutes=args.duration
    )

    # Run
    results = trader.run()

    if results:
        print("\n" + "=" * 50)
        print("TRADING COMPLETE")
        print("=" * 50)
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Return: {results['total_return']:.2f}%")
        print(f"Sharpe: {results['sharpe']:.2f}")
        print(f"Trades: {results['trades']}")


if __name__ == '__main__':
    main()
