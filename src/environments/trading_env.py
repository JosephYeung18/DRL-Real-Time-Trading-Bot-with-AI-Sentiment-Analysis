"""
Stock Trading Environment with Enhanced Reward Function for Active Timing
Based on FinRL's StockTradingEnv
"""

import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

class StockTradingEnvOptimized(StockTradingEnv):
    """
    Optimized Stock Trading Environment with Enhanced Reward Function
    - Continuous position ratio action space [0, 1]
    - Enhanced reward function:
      - Future return lookahead reward
      - Turnover rate reward
      - Cash holding reward when market is down
      - Momentum and hold rewards
      - Sell encouragement (NEW)
    """

    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.peak_value = self.initial_amount
        self.last_action = 0.5  # For turnover calculation
        self.config = {
            'position_penalty_coef': 0.0,
            'transaction_penalty': 0.0,
            'drawdown_threshold': 0.10,
            'drawdown_penalty_coef': 0.1,
            'reward_scaling': 1.0,
            'momentum_boost': 0.5,
            'hold_bonus': 0.01,
            'future_lookahead': 5,
            # New sell encouragement
            'sell_encouragement_coef': 0.02,
        }

    def _get_future_return(self, current_day, current_price):
        """Calculate future return over the next N days (only for training)"""
        future_days = self.config['future_lookahead']
        max_day = len(self.df.index.unique()) - 1

        future_day = min(current_day + future_days, max_day)
        if future_day <= current_day:
            return 0

        try:
            future_data = self.df.loc[future_day]
            if hasattr(future_data, 'values'):
                future_price = future_data['close'].values[0] if len(future_data) > 0 else current_price
            else:
                future_price = future_data['close']
            future_return = (future_price - current_price) / current_price if current_price > 0 else 0
            return future_return
        except:
            return 0

    def step(self, actions):
        """Process continuous position ratio action with enhanced reward for active timing"""
        # Helper function to convert state values to scalars
        def to_scalar(v):
            """Convert state value to scalar"""
            if isinstance(v, (list, tuple)):
                v = np.array(v)
            if hasattr(v, 'item'):
                return v.item()
            elif hasattr(v, '__iter__') and not isinstance(v, str):
                v = np.array(v)
                if v.ndim > 0:
                    return float(v.flat[0])
            return float(v)

        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            cash = to_scalar(self.state[0])
            if self.stock_dim == 1:
                price = to_scalar(self.state[1])
                shares = to_scalar(self.state[2])
                stock_value = price * shares
            else:
                prices = np.array([to_scalar(self.state[i]) for i in range(1, self.stock_dim + 1)])
                shares = np.array([to_scalar(self.state[self.stock_dim + 1 + i]) for i in range(self.stock_dim)])
                stock_value = np.sum(prices * shares)
            end_total_asset = cash + stock_value
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            total_return = (end_total_asset - self.initial_amount) / self.initial_amount
            self.reward = total_return * 100
            return self.state, self.reward, self.terminal, False, {}

        # Parse action [0, 1] position ratio
        if isinstance(actions, np.ndarray):
            action_val = float(actions.flat[0]) if actions.ndim > 0 else float(actions)
        elif hasattr(actions, '__iter__'):
            action_val = float(list(actions)[0])
        else:
            action_val = float(actions)
        target_position_ratio = np.clip(action_val, 0, 1)

        # Get current state
        current_price = to_scalar(self.state[1])
        current_cash = to_scalar(self.state[0])
        current_shares = to_scalar(self.state[self.stock_dim + 1])
        current_asset_value = current_shares * current_price
        current_total_asset = current_cash + current_asset_value
        current_position_ratio = current_asset_value / current_total_asset if current_total_asset > 0 else 0

        # Calculate target position
        target_asset_value = current_total_asset * target_position_ratio
        target_shares = int(target_asset_value / current_price) if current_price > 0 else 0
        shares_to_trade = target_shares - current_shares

        # Execute trade
        shares_traded = 0
        if shares_to_trade > 0:
            buy_amount = shares_to_trade * current_price * (1 + self.buy_cost_pct[0])
            if buy_amount <= current_cash:
                self.state[0] -= buy_amount
                self.state[self.stock_dim + 1] += shares_to_trade
                self.cost += shares_to_trade * current_price * self.buy_cost_pct[0]
                self.trades += 1
                shares_traded = shares_to_trade
        elif shares_to_trade < 0:
            shares_to_sell = min(abs(shares_to_trade), current_shares)
            if shares_to_sell > 0:
                sell_amount = shares_to_sell * current_price * (1 - self.sell_cost_pct[0])
                self.state[0] += sell_amount
                self.state[self.stock_dim + 1] -= shares_to_sell
                self.cost += shares_to_sell * current_price * self.sell_cost_pct[0]
                self.trades += 1
                shares_traded = -shares_to_sell

        self.actions_memory.append([target_position_ratio])

        # Update state
        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif len(self.df.tic.unique()) > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        self.state = self._update_state()

        # Calculate new position
        if self.stock_dim == 1:
            new_price = to_scalar(self.state[1])
            new_shares = to_scalar(self.state[2])
            end_total_asset = to_scalar(self.state[0]) + new_price * new_shares
        else:
            prices = np.array([to_scalar(self.state[i]) for i in range(1, self.stock_dim + 1)])
            shares = np.array([to_scalar(self.state[self.stock_dim + 1 + i]) for i in range(self.stock_dim)])
            end_total_asset = to_scalar(self.state[0]) + np.sum(prices * shares)

        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        # ========== Enhanced Reward Function ==========

        # 1. Base return
        portfolio_return = (end_total_asset - current_total_asset) / current_total_asset if current_total_asset > 0 else 0
        return_reward = portfolio_return * 100

        # 2. Future Return Lookahead Reward
        future_return = self._get_future_return(self.day - 1, current_price)
        future_reward = 0
        if target_position_ratio < 0.3 and future_return < 0:
            # Low position when market will drop - good timing!
            future_reward = 0.01 * abs(future_return)
        elif target_position_ratio > 0.7 and future_return > 0:
            # High position when market will rise - good timing!
            future_reward = 0.01 * future_return
        elif target_position_ratio > 0.7 and future_return < 0:
            # High position when market will drop - bad timing
            future_reward = -0.02 * abs(future_return)

        # 3. Turnover Rate Reward
        action_change = abs(target_position_ratio - self.last_action)
        turnover_reward = 0.001 * action_change

        # 4. Cash Holding Reward (encourages reducing position when market is down)
        market_return = (self.data['close'] - current_price) / current_price if hasattr(self.data, 'close') else 0
        cash_reward = 0
        if target_position_ratio < 0.2 and market_return < 0:
            cash_reward = 0.005

        # 5. Momentum reward
        momentum_reward = 0
        if portfolio_return > 0 and target_position_ratio > 0.5:
            momentum_reward = 0.3 * portfolio_return * 100

        # 6. Hold reward
        hold_reward = 0
        if target_position_ratio > 0.5 and portfolio_return > 0:
            hold_reward = 0.005

        # 7. Sell Encouragement (NEW - encourage taking profits)
        sell_encouragement = 0
        if shares_to_trade < 0:  # Selling
            if portfolio_return > 0.02:
                # Taking profit - reward it
                sell_encouragement = self.config['sell_encouragement_coef'] * portfolio_return * 100
            elif current_position_ratio > 0.8:
                # Reducing large position
                sell_encouragement = 0.01

        # 8. Drawdown penalty
        self.peak_value = max(self.peak_value, current_total_asset)
        current_drawdown = (self.peak_value - end_total_asset) / self.peak_value
        drawdown_penalty = 0
        if current_drawdown > self.config['drawdown_threshold']:
            drawdown_penalty = -0.1 * (current_drawdown - self.config['drawdown_threshold'])

        # Combine rewards
        self.reward = (return_reward + future_reward + turnover_reward + cash_reward +
                       momentum_reward + hold_reward + drawdown_penalty + sell_encouragement)
        self.reward = self.reward * self.config['reward_scaling']

        # Update last action
        self.last_action = target_position_ratio

        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}