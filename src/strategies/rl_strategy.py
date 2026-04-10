"""
RL Trading Strategy for Lumibot
Supports proportional trading and action smoothing
"""

import numpy as np
import pandas as pd
from lumibot.strategies import Strategy

TRANSACTION_FEE = 0.001
TICKER = 'TSLA'

class RLStrategyProportional(Strategy):
    """Reinforcement Learning Trading Strategy with Proportional Trading and Action Smoothing"""

    def initialize(self, trade_data=None, model=None, indicators=None, symbol=None,
                   model_type='a2c', initial_capital=100000, action_smoothing=0.8, **kwargs):
        """Lumibot strategy initialization"""
        self.sleeptime = "1D"
        self.symbol = symbol if symbol is not None else TICKER
        self.model = model
        self.trade_data = trade_data
        self.indicators = indicators if indicators is not None else []
        self.model_type = model_type
        self.initial_capital = initial_capital

        # Action smoothing state
        self.last_executed_position = 0.0
        self.action_smoothing = action_smoothing
        self.last_raw_action = 0.5  # For turnover calculation

        # Statistics
        self.trade_count = 0
        self.total_steps = 0

        # Account value history for tracking
        self.account_value_history = []

        # Detailed action tracking for logging
        self.raw_action_history = []
        self.smoothed_action_history = []
        self.price_history = []
        self.shares_history = []

        print(f"\n{'='*60}")
        print(f"RL Strategy Initialized - Model: {self.model_type}, Symbol: {self.symbol}")
        print(f"Action Smoothing α = {self.action_smoothing}")
        print(f"{'='*60}")

    def write_backtest_settings(self, settings_file):
        """Override to exclude model from serialization"""
        import jsonpickle
        settings = {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'initial_capital': self.initial_capital,
            'action_smoothing': self.action_smoothing,
        }
        with open(settings_file, 'w') as f:
            f.write(jsonpickle.encode(settings))

    def get_state_for_date(self, current_date):
        """Get state DataFrame for a specific date"""
        if self.trade_data is None:
            return None

        if 'date' in self.trade_data.columns:
            date_col = 'date'
        elif 'datetime' in self.trade_data.columns:
            date_col = 'datetime'
        else:
            return None

        state_df = self.trade_data[self.trade_data[date_col].astype(str).str.startswith(current_date)]
        return state_df

    def prepare_observation(self, state_df):
        """Prepare observation vector from state DataFrame"""
        if state_df is None or len(state_df) == 0:
            return None

        # Default indicator list matching FinRL configuration
        default_indicators = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                             'close_30_sma', 'close_60_sma', 'cci', 'willr', 'adx', 'atr',
                             'sentiment_score']
        indicators = self.indicators if self.indicators else default_indicators

        cash = self.portfolio_value * 0.1  # Approximate cash

        price = state_df['close'].values[0] if 'close' in state_df.columns else 0
        shares = 0
        position = self.get_position(self.symbol)
        if position:
            shares = position.quantity

        obs = [cash, price]
        obs.append(shares)

        for ind in indicators:
            if ind in state_df.columns:
                val = state_df[ind].values[0]
                if pd.isna(val):
                    val = 0
                obs.append(val)
            else:
                obs.append(0)

        return np.array(obs, dtype=np.float32)

    def on_trading_iteration(self):
        """Execute trading logic each trading day"""
        self.total_steps += 1
        current_time = self.get_datetime()
        current_date = current_time.strftime('%Y-%m-%d')

        current_price = self.get_last_price(self.symbol)
        if current_price is None or current_price <= 0:
            return

        # Track account value
        self.account_value_history.append(self.portfolio_value)

        state_df = self.get_state_for_date(current_date)
        if state_df is None:
            return

        obs = self.prepare_observation(state_df)
        if obs is None:
            return

        # Model prediction
        raw_action, _ = self.model.predict(obs, deterministic=False)

        # Process action [0, 1] position ratio
        if hasattr(raw_action, '__iter__'):
            raw_position = float(np.clip(raw_action[0], 0, 1))
        else:
            raw_position = float(np.clip(raw_action, 0, 1))

        # Action smoothing
        smoothed_position = (
            self.action_smoothing * raw_position +
            (1 - self.action_smoothing) * self.last_executed_position
        )
        smoothed_position = np.clip(smoothed_position, 0, 1)

        # Track actions for detailed logging
        self.raw_action_history.append(float(raw_position))
        self.smoothed_action_history.append(float(smoothed_position))
        self.price_history.append(float(current_price))

        # Calculate current position
        position = self.get_position(self.symbol)
        current_shares = position.quantity if position else 0
        self.shares_history.append(int(current_shares))

        current_cash = self.portfolio_value - (current_shares * current_price)
        current_total = self.portfolio_value

        current_position_ratio = (current_shares * current_price) / current_total if current_total > 0 else 0

        # Calculate target shares
        target_shares = int((current_total * smoothed_position) / current_price)
        shares_to_trade = target_shares - current_shares

        # Execute trade
        if shares_to_trade > 0:
            buy_value = shares_to_trade * current_price
            buy_cost = buy_value * TRANSACTION_FEE
            if current_cash >= buy_value + buy_cost:
                order = self.create_order(self.symbol, shares_to_trade, "buy")
                self.submit_order(order)
                self.last_executed_position = (current_shares + shares_to_trade) * current_price / current_total
                self.trade_count += 1

        elif shares_to_trade < 0:
            order = self.create_order(self.symbol, abs(shares_to_trade), "sell")
            self.submit_order(order)
            self.last_executed_position = (current_shares + shares_to_trade) * current_price / current_total
            self.trade_count += 1

        # Update last raw action for turnover calculation
        self.last_raw_action = float(raw_position)
