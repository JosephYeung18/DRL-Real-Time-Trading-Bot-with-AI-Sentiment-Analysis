"""
EURUSD Trading Environment with TRUE Short Selling Support

Key differences from FinRL StockTradingEnv:
1. Shares can go negative (short positions)
2. Margin requirements for short positions
3. Borrowing costs for short positions
4. Symmetric P&L for long/short positions
5. Compatible with stable-baselines3 A2C
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class EURUSDShortSellingEnv(gym.Env):
    """
    EURUSD Trading Environment with TRUE Short Selling Support.

    Action Interpretation:
    - action = -1.0: Full short (100% of capital value)
    - action = -0.5: Semi-short (50% of capital value)
    - action =  0.0: Flat (no position)
    - action = +0.5: Semi-long (50% of capital value)
    - action = +1.0: Full long (100% of capital value)

    Short Selling Mechanics:
    - Initial margin: 50% of short position value
    - Maintenance margin: 25%
    - Daily borrowing cost: ~0.01% per day
    - Max leverage: 2x
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int = 1,
        hmax: int = 100,
        initial_amount: float = 1_000_000,
        buy_cost_pct: float = 0.0001,
        sell_cost_pct: float = 0.0001,
        short_cost_pct: float = 0.0001,
        borrow_rate: float = 0.0001,
        margin_requirement: float = 0.50,
        maintenance_margin: float = 0.25,
        max_leverage: float = 2.0,
        reward_scaling: float = 1e-5,
        state_space: int = 15,
        tech_indicator_list: list = None,
        day: int = 0,
    ):
        super().__init__()

        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.short_cost_pct = short_cost_pct
        self.borrow_rate = borrow_rate
        self.margin_requirement = margin_requirement
        self.maintenance_margin = maintenance_margin
        self.max_leverage = max_leverage
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list or []

        # Episode tracking
        self.day = day
        self.terminal = False
        self.episode = 0

        # Portfolio state
        self.cash = initial_amount
        self.shares = 0  # Can be negative for shorts
        self.entry_price = 0.0
        self.total_asset = initial_amount

        # Tracking
        self.asset_memory = [initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.trades = 0
        self.cost = 0
        self.total_borrowing_costs = 0

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )

        # Initialize
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

    def _initiate_state(self):
        """Initialize state vector: [cash, price, shares, indicators...]"""
        state = (
            [self.cash]
            + [self.data.close]
            + [self.shares]
            + [self.data[tech] for tech in self.tech_indicator_list]
        )
        return np.array(state, dtype=np.float32)

    def _update_state(self):
        """Update state vector after a step."""
        state = (
            [self.cash]
            + [self.data.close]
            + [self.shares]
            + [self.data[tech] for tech in self.tech_indicator_list]
        )
        return np.array(state, dtype=np.float32)

    def _get_date(self):
        """Get current date string."""
        if hasattr(self.data, 'date'):
            return self.data.date
        elif hasattr(self.df, 'index') and hasattr(self.df.index, 'get_level_values'):
            return str(self.df.loc[self.day].name)
        else:
            return str(self.day)

    def _calculate_position_value(self, price):
        """Calculate current position value (works for both long and short)."""
        return self.shares * price

    def _open_short(self, num_shares, price):
        """
        Open/add to short position.

        When shorting:
        1. Borrow shares from broker
        2. Sell borrowed shares at current price
        3. Receive cash (after transaction cost)
        4. Post margin requirement
        """
        trade_value = num_shares * price
        cost = trade_value * self.short_cost_pct
        received = abs(trade_value) - cost

        # Save old shares before modifying
        old_shares = self.shares

        # Update cash
        self.cash += received

        # Track short position (shares become negative)
        self.shares -= num_shares

        # Update entry price (weighted average of short positions)
        if old_shares < 0 or (old_shares == 0 and num_shares > 0):
            # Old position value = abs(old_shares) * old_entry_price
            # New position value = num_shares * price
            old_position_value = abs(old_shares) * self.entry_price if old_shares < 0 else 0
            new_position_value = num_shares * price
            total_position_value = old_position_value + new_position_value
            self.entry_price = total_position_value / abs(self.shares) if self.shares != 0 else price

        self.trades += 1
        self.cost += cost

    def _close_short(self, num_shares, price):
        """
        Close/reduce short position.

        When covering:
        1. Buy back shares at current price
        2. Pay transaction cost
        3. Return borrowed shares to broker
        """
        buy_cost = abs(num_shares) * price * (1 + self.sell_cost_pct)
        self.cash -= buy_cost

        # Release proportional margin
        self.shares += num_shares  # shares moves toward zero

        self.trades += 1
        self.cost += abs(num_shares) * price * self.sell_cost_pct

    def _buy_stock(self, num_shares, price):
        """Buy stock (add to long position)."""
        buy_amount = num_shares * price * (1 + self.buy_cost_pct)
        if buy_amount <= self.cash:
            self.cash -= buy_amount
            self.shares += num_shares

            # Update entry price for long
            if self.shares > 0 and num_shares > 0:
                old_value = (self.shares - num_shares) * self.entry_price
                new_value = num_shares * price
                total_value = old_value + new_value
                self.entry_price = total_value / self.shares if self.shares != 0 else 0

            self.trades += 1
            self.cost += num_shares * price * self.buy_cost_pct

    def _sell_stock(self, num_shares, price):
        """Sell/reduce long position."""
        shares_to_sell = min(num_shares, self.shares)
        if shares_to_sell > 0:
            sell_amount = shares_to_sell * price * (1 - self.sell_cost_pct)
            self.cash += sell_amount
            self.shares -= shares_to_sell
            self.trades += 1
            self.cost += shares_to_sell * price * self.sell_cost_pct
            return shares_to_sell
        return 0

    def _apply_borrowing_cost(self, price):
        """Apply daily borrowing cost on short positions."""
        if self.shares < 0:
            position_value = abs(self.shares * price)
            daily_cost = position_value * self.borrow_rate
            self.cash -= daily_cost
            self.total_borrowing_costs += daily_cost

    def _check_margin_call(self, price):
        """Check for margin call and liquidate if necessary."""
        if self.shares < 0:
            # Calculate unrealized loss on short
            if self.entry_price > 0:
                unrealized_pnl = (self.entry_price - price) * abs(self.shares)
            else:
                unrealized_pnl = 0

            # Margin used = initial margin - unrealized P&L
            position_value = abs(self.shares * price)
            initial_margin = position_value * self.margin_requirement
            current_margin = initial_margin + unrealized_pnl
            maintenance_level = position_value * self.maintenance_margin

            if current_margin < maintenance_level:
                # Margin call - liquidate entire short position
                self._liquidate_short_position(price)
                return True
        return False

    def _liquidate_short_position(self, price):
        """Force liquidate short position at current price."""
        if self.shares < 0:
            buy_cost = abs(self.shares) * price * (1 + self.sell_cost_pct)
            self.cash -= buy_cost
            self.shares = 0
            self.entry_price = 0.0
            self.trades += 1

    def step(self, actions):
        """Execute one trading step."""
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        # Get current state values
        current_price = float(self.state[1])
        current_shares = self.shares
        current_total_asset = self.cash + self._calculate_position_value(current_price)

        # Parse action
        if isinstance(actions, np.ndarray):
            action_val = float(actions.flat[0]) if actions.ndim > 0 else float(actions)
        elif hasattr(actions, '__iter__'):
            action_val = float(list(actions)[0])
        else:
            action_val = float(actions)

        action = np.clip(action_val, -1.0, 1.0)

        # Calculate target position ratio
        target_ratio = action  # -1 to 1

        if self.terminal:
            # Final step - calculate terminal reward
            end_total_asset = self.cash + self._calculate_position_value(current_price)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            # Terminal reward based on total return percentage
            total_return_pct = (end_total_asset - current_total_asset) / current_total_asset
            self.reward = total_return_pct
            return self.state, self.reward, self.terminal, False, {}

        # Execute trade based on action
        if abs(target_ratio) < 0.01:
            # Flatten position
            if current_shares > 0:
                self._sell_stock(abs(current_shares), current_price)
            elif current_shares < 0:
                self._close_short(abs(current_shares), current_price)
        elif target_ratio > 0.01:
            # LONG signal
            target_position_value = self.cash * target_ratio * self.max_leverage
            target_shares = int(target_position_value / current_price) if current_price > 0 else 0
            shares_to_trade = target_shares - current_shares

            if shares_to_trade > 0:
                self._buy_stock(shares_to_trade, current_price)
            elif shares_to_trade < 0:
                self._sell_stock(abs(shares_to_trade), current_price)
        else:
            # SHORT signal (target_ratio < -0.01)
            target_position_value = self.cash * abs(target_ratio) * self.max_leverage
            target_shares = int(target_position_value / current_price) if current_price > 0 else 0

            # For short, shares are negative
            target_short_shares = -target_shares  # NEGATIVE for short position
            shares_to_trade = target_short_shares - current_shares  # current_shares may be negative

            if shares_to_trade < 0:
                # Need to short more (open/add to short) - shares become more negative
                self._open_short(abs(shares_to_trade), current_price)
            elif shares_to_trade > 0:
                # Need to cover (buy back to reduce short) - shares move toward zero
                self._close_short(shares_to_trade, current_price)

        # Apply borrowing costs for shorts
        self._apply_borrowing_cost(current_price)

        # Check margin call
        margin_call = self._check_margin_call(current_price)

        # Update state
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate new total asset
        new_price = float(self.state[1])
        new_shares = self.shares
        end_total_asset = self.cash + new_shares * new_price
        self.total_asset = end_total_asset

        # Calculate reward based on price direction and position alignment
        # Get price change from previous step
        price_change_pct = (new_price - current_price) / current_price if current_price > 0 else 0
        portfolio_change_pct = (end_total_asset - current_total_asset) / current_total_asset if current_total_asset > 0 else 0

        # Base reward is portfolio change percentage
        reward = portfolio_change_pct

        # Directional alignment reward: reward when position matches price direction
        position_direction = np.sign(action)
        price_direction = np.sign(price_change_pct)
        directional_alignment = position_direction * price_direction
        if directional_alignment > 0:
            reward += abs(price_change_pct) * 0.1  # Reward for correct direction
        elif directional_alignment < 0:
            reward -= abs(price_change_pct) * 0.1  # Penalty for wrong direction

        # Symmetric P&L reward for position
        position_pnl = new_shares * (new_price - current_price)
        reward += position_pnl * 1e-5  # Scale down

        # Position size reward - encourage using more of the action space
        # Large position = more reward (since we're trading with leverage)
        if abs(self.shares) > 100:
            position_size_bonus = abs(self.shares) / 100000 * 0.0001
            reward += position_size_bonus

        # Leverage penalty to prevent excessive positions
        position_ratio = abs(self.shares * current_price) / current_total_asset if current_total_asset > 0 else 0
        if position_ratio > 2.0:  # Very high leverage
            reward -= 0.0002
        elif position_ratio > 1.5:
            reward -= 0.0001

        # Action diversity bonus - reward for using varied actions
        if len(self.actions_memory) >= 5:
            recent_actions = np.array(self.actions_memory[-5:])
            action_std = np.std(recent_actions)
            if action_std > 0.1:  # Good diversity
                reward += 0.00005

        self.reward = reward

        # Memory
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())
        self.actions_memory.append([action])
        self.rewards_memory.append(self.reward)

        # Update terminal
        self.terminal = self.terminal or margin_call

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.day = 0
        self.cash = self.initial_amount
        self.shares = 0
        self.entry_price = 0.0
        self.total_asset = self.initial_amount
        self.terminal = False
        self.trades = 0
        self.cost = 0
        self.total_borrowing_costs = 0

        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1

        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        return self.state, {}

    def get_sb_env(self):
        """Return stable-baselines3 compatible vectorized environment."""
        from stable_baselines3.common.vec_env import DummyVecEnv

        e = DummyVecEnv([lambda: self])
        result = e.reset()
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        return e, obs
