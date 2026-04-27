# QCS - Quantitative Commodity Trading System

A real-time trading bot with deep reinforcement learning and AI sentiment analysis supporting forex and stock market trading.

---

## Project Overview

This system implements DRL (Deep Reinforcement Learning) trading strategies using stable-baselines3 A2C algorithm based on FinRL and Lumibot. 


---

## Project Structure

```
QCS/
├── src/
│   ├── environments/
│   │   ├── eurusd_short_env.py      # EURUSD short-selling environment
│   │   └── trading_env.py           # FinRL stock environment wrapper
│   ├── strategies/
│   │   └── rl_strategy.py           # DRL Lumibot strategy
│   └── utils/
│       ├── benchmarks.py            # SMA/Buy&Hold benchmarks
│       ├── config.py                # TSLA configuration
│       ├── config_eurusd.py         # EURUSD configuration
│       ├── indicators.py            # Technical indicators
│       ├── metrics.py               # Performance metrics
│       ├── model_loader.py         # Model loading utilities
│       ├── plotting.py              # Visualization
│       └── sentiment.py            # Sentiment analysis
│
├── eurusd_backtest_final.py         # EURUSD backtest (main entry)
├── main.py                         # TSLA backtest (no sentiment)
├── main_sentiment.py              # TSLA backtest (with sentiment)
├── run_live_trading.py            # TSLA real-time trading
├── eurusd_ib_trading.py          # EURUSD IB real-time trading
│
├── data/
│   ├── EURUSD.csv                 # EURUSD historical data
│   ├── EURUSD_SEN.csv             # EURUSD sentiment data
│   ├── TSLA.csv                   # TSLA stock data
│   └── TSLA_SEN.csv               # TSLA sentiment data
│
├── trained_models/
│   ├── best/                      # Best models
│   ├── eurusd/                    # EURUSD models
│   ├── eurusd_no_sentiment/        # EURUSD models (no sentiment)
│   └── eurusd_sentiment/          # EURUSD models (sentiment-enhanced)
│
├── eurusd_backtest_result/        # EURUSD backtest results (no sentiment)
├── eurusd_backtest_result_sentiment/  # EURUSD backtest results (with sentiment)
│
├── tests/
│   └── test_indicators.py        # Technical indicator tests
│
├── docs/
│   ├── ARCHITECTURE_DESIGN.md     # Architecture design document
│   └── architecture.tex          # LaTeX architecture document
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- stable-baselines3 >= 2.0
- finrl >= 0.3
- talib
- torch
- numpy, pandas, matplotlib

### 2. Run EURUSD Backtest

```bash
python eurusd_backtest_final.py
```

The backtest runs both with and without sentiment modes automatically.

### 3. Run TSLA Backtest

```bash
# Without sentiment
python main.py --mode backtest

# With sentiment
python main_sentiment.py --mode backtest --sentiment-mode 3
```

---

## EURUSD Short-Selling Environment

### Action Space

**Continuous space:** `a ∈ [-1, 1]`

| Action | Position | Description |
|--------|----------|-------------|
| a = -1.0 | Full Short | 100% of capital value in short |
| a = -0.5 | Semi-Short | 50% of capital value in short |
| a = 0.0 | Flat | No position |
| a = +0.5 | Semi-Long | 50% of capital value in long |
| a = +1.0 | Full Long | 100% of capital value in long |

**Action Smoothing (EMA):**
```python
smoothed_action = 0.5 * raw_action + 0.5 * last_action
```

### Reward Function

```
R = R_base + R_direction + R_pnl + R_size + R_leverage + R_diversity
```

| Component | Formula | Description |
|-----------|---------|-------------|
| R_base | ΔV/V | Portfolio return |
| R_direction | 0.1 · sign(a) · sign(ΔP/P) · \|ΔP/P\| | Direction alignment |
| R_pnl | 10⁻⁵ · shares · ΔP | Symmetric P&L |
| R_size | [\|shares\| > 100] · \|shares\|/100000 · 0.0001 | Position size bonus |
| R_leverage | -([ratio > 2.0] · 0.0002 + [ratio > 1.5] · 0.0001) | Leverage penalty |
| R_diversity | [std(a_{t-4:t}) > 0.1] · 0.00005 | Action diversity bonus |

### Trading Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_leverage | 2.0 | Maximum 2x leverage |
| short_cost_pct | 0.01% | Transaction fee |
| borrow_rate | 0.01%/day | Daily borrowing cost (~3.65% annually) |
| margin_requirement | 50% | Initial margin |
| maintenance_margin | 25% | Maintenance margin |

### Model Hyperparameters

```python
model_params = {
    "n_steps": 4096,
    "ent_coef": 0.05,
    "learning_rate": 0.00007,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}
```

---

## State Space

**EURUSD State Vector:**
```
state = [cash, price, shares, macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma, atr, adx, willr, cci, sentiment_score]
```

**Technical Indicators:**
- MACD (12, 26, 9)
- Bollinger Bands (20)
- RSI (14)
- CCI (14)
- ADX (14)
- ATR (14)
- Williams %R (14)
- Sentiment Score (-1 to +1, optional)


## Core Modules

### src/environments/eurusd_short_env.py

Custom Gymnasium trading environment featuring:
- True short selling from flat positions
- Symmetric long/short reward mechanism
- Margin and borrowing cost calculation
- Leverage management (2x max)
- Position tracking with entry price

### src/utils/

| Module | Description |
|--------|-------------|
| indicators.py | MACD, RSI, CCI, ADX, ATR, Williams %R calculations |
| sentiment.py | Sentiment score processing |
| metrics.py | Sharpe ratio, max drawdown, win rate calculations |
| benchmarks.py | SMA and Buy&Hold benchmark strategies |
| plotting.py | Portfolio charts, position charts, action distributions |

---

## File Descriptions

### Main Entry Points

| File | Description |
|------|-------------|
| eurusd_backtest_final.py | EURUSD backtest with short-selling support |
| main.py | TSLA backtest (no sentiment) |
| main_sentiment.py | TSLA backtest (with sentiment) |
| run_live_trading.py | TSLA real-time trading via Alpaca API |
| eurusd_ib_trading.py | EURUSD real-time trading via Interactive Brokers |

### Backtest Results

| Directory | Description |
|----------|-------------|
| eurusd_backtest_result/ | EURUSD backtest results (no sentiment) |
| eurusd_backtest_result_sentiment/ | EURUSD backtest results (with sentiment) |

Each directory contains:
- `backtest_results.csv` - Detailed trading data
- `report.txt` - Performance report
- `trade_log_detailed.csv` - Trade log
- `full_evaluation_plot.png` - Visualization

---

## Key Design Features

### 1. Symmetric Reward Design

The reward function treats long and short positions symmetrically:
- Long profit = Short profit when price moves favorably
- Long loss = Short loss when price moves adversely

### 2. Direction Alignment

The `R_direction` component rewards the agent for aligning position direction with price movement:
- Positive alignment: bonus reward
- Negative alignment: penalty

### 3. Action Smoothing

EMA smoothing (α=0.5) reduces noise in action selection, preventing excessive trading.

### 4. Leverage Control

Leverage penalty discourages over-leveraging:
- >2.0x leverage: -0.0002 penalty
- >1.5x leverage: -0.0001 penalty

---

## Important Notes

1. **EURUSD backtest:** Use `eurusd_backtest_final.py`
2. **TSLA backtest:** Use `main.py` or `main_sentiment.py`
3. **RL Non-determinism:** Different training runs may produce different results due to the stochastic nature of reinforcement learning
4. **Past performance does not guarantee future results**

---

## API Configuration

### Interactive Brokers (EURUSD)
```bash
# IB Gateway/TWS Settings
# 1. File > Global Configuration > API > Settings
# 2. Enable Socket Clients
# 3. Port: 7496 (Paper) / 7497 (Live)
```

### Alpaca (TSLA)
```bash
# Register: https://app.alpaca.markets/
# Set API keys as environment variables
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
```

---


## Citation

If you use this system in academic work, please cite:

```bibtex
@software{QCS2024,
  title = {QCS - DRL Quantitative Trading Bot},
  author = {Joseph Yeung},
  year = {2024},
  url = {https://github.com/JosephYeung18/DRL-Real-Time-Trading-Bot-with-AI-Sentiment-Analysis}
}
```

---

## License

MIT License
