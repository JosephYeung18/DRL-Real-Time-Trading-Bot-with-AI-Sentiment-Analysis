# Real-Time Trading Bot with Deep Reinforcement Learning Agents and AI Sentiment Analysis

## 项目架构与设计文档
## Project Architecture and Design Documentation

---

## 1. 系统概述 (System Overview)

本项目(QCS - Quantitative Commodity Trading System)是一个基于深度强化学习的量化交易系统，支持:

- **EURUSD外汇交易**: 通过Interactive Brokers实时交易
- **TSLA股票交易**: 通过Alpaca API实时交易
- **情绪增强**: 支持三种情绪分析模式 (OpenAI API / Alpaca API / 预存CSV)

---

## 2. 系统架构图 (System Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME TRADING BOT WITH DRL & SENTIMENT                      │
│                    深度强化学习与情绪分析实时交易系统                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           DATA LAYER (数据层)                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │
│  │  │  EURUSD.csv │  │TSLA_SEN.csv │  │ EURUSD_     │  │  TSLA.csv   │       │   │
│  │  │  (价格数据) │  │ (情绪数据)  │  │ SEN.csv     │  │  (价格数据) │       │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│                                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        FEATURE ENGINEERING (特征工程)                        │   │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │   │
│  │  │ Technical Indicators (技术指标)                                        │ │   │
│  │  │   MACD, Bollinger Bands, RSI, CCI, ADX, ATR, Williams %R             │ │   │
│  │  │   turbulence_index, momentum_5, volatility_5, price_to_sma           │ │   │
│  │  ├───────────────────────────────────────────────────────────────────────┤ │   │
│  │  │ Sentiment Features (情绪特征) -3 modes:                              │ │   │
│  │  │   Mode 1: OpenAI API + LLM (qwen3-coder:480b via Ollama)            │ │   │
│  │  │   Mode 2: Alpaca API News Sentiment                                  │ │   │
│  │  │   Mode 3: Pre-saved CSV (default, fastest)                           │ │   │
│  │  └───────────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│                                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                     ENVIRONMENT LAYER (环境层)                              │   │
│  │                                                                             │   │
│  │  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │   │
│  │  │   StockTradingEnvOptimized     │  │    EURUSDShortSellingEnv        │  │   │
│  │  │   (TSLA股票交易环境)            │  │    (EURUSD外汇做空环境)          │  │   │
│  │  ├─────────────────────────────────┤  ├─────────────────────────────────┤  │   │
│  │  │ • 连续仓位比例 [0,1]            │  │ • 真正做空支持 [-1,+1]          │  │   │
│  │  │ • 未来收益Lookahead奖励         │  │ • 保证金要求 (50%)             │  │   │
│  │  │ • 动量/持仓奖励                 │  │ • 维护保证金 (25%)             │  │   │
│  │  │ • 回撤惩罚                      │  │ • 借贷成本计算                  │  │   │
│  │  │ • 卖出鼓励机制                  │  │ • 对称多空奖惩                  │  │   │
│  │  └─────────────────────────────────┘  └─────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│                                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                      MODEL LAYER (模型层)                                    │   │
│  │                                                                             │   │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │   │
│  │  │              Stable Baselines3 (A2C / PPO)                           │ │   │
│  │  ├───────────────────────────────────────────────────────────────────────┤ │   │
│  │  │  A2C (Advantage Actor-Critic)                                        │ │   │
│  │  │    • n_steps=2048, ent_coef=0.001                                    │ │   │
│  │  │    • learning_rate=0.0007, gamma=0.99                                │ │   │
│  │  ├───────────────────────────────────────────────────────────────────────┤ │   │
│  │  │  PPO (Proximal Policy Optimization)                                  │ │   │
│  │  │    • n_steps=2048, clip_range=0.2                                    │ │   │
│  │  │    • batch_size=128, learning_rate=0.0007                           │ │   │
│  │  └───────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                             │   │
│  │  State Space:                                                              │   │
│  │    TSLA: 1 + 2*stock_dim + len(indicators)*stock_dim = 15               │   │
│  │    EURUSD: 1 + 2 + num_indicators = 15                                   │   │
│  │                                                                             │   │
│  │  Training: 300,000 steps with early stopping (patience=10)              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│                                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                    STRATEGY LAYER (策略层)                                 │   │
│  │                                                                             │   │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │   │
│  │  │              RLStrategyProportional (Lumibot)                        │ │   │
│  │  ├───────────────────────────────────────────────────────────────────────┤ │   │
│  │  │  • Action Smoothing: α = 0.8                                        │ │   │
│  │  │    final_action = α * raw_action + (1-α) * last_action              │ │   │
│  │  │  • Position Ratio → Share Count Conversion                          │ │   │
│  │  │  • Historical Tracking: account_value, actions, shares              │ │   │
│  │  └───────────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│                         ┌───────────────┼───────────────┐                          │
│                         ▼               ▼               ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                       BROKER LAYER (经纪商层)                                │   │
│  │                                                                             │   │
│  │  ┌─────────────────────────┐    ┌─────────────────────────┐               │   │
│  │  │   Interactive Brokers   │    │        Alpaca           │               │   │
│  │  │   (EURUSD Forex)        │    │   (TSLA Stock)          │               │   │
│  │  ├─────────────────────────┤    ├─────────────────────────┤               │   │
│  │  │ • ib_insync library    │    │ • alpaca-py library    │               │   │
│  │  │ • Port 7496 (Paper)     │    │ • Paper: paper.alpaca  │               │   │
│  │  │ • Port 7497 (Live)      │    │ • Live: api.alpaca     │               │   │
│  │  │ • Real-time bid/ask    │    │ • Free trading         │               │   │
│  │  │ • Market/Limit orders  │    │ • Market orders only   │               │   │
│  │  └─────────────────────────┘    └─────────────────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                │                    │
                                ▼                    ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   BACKTESTING   │    │ LIVE TRADING    │
                    │    (回测)        │    │  (实盘)         │
                    ├─────────────────┤    ├─────────────────┤
                    │ • Lumibot       │    │ • IB TWS/Gateway│
                    │ • Historical    │    │ • Alpaca API    │
                    │   data         │    │ • Real-time     │
                    │ • PyPleurs     │    │   execution     │
                    └─────────────────┘    └─────────────────┘
```

---

## 3. 代码结构 (Code Structure)

```
/Users/josephyoung/Desktop/TestMinimax/
│
├── ════════════════════════════════════════════════════════════════════════════════
│                              SOURCE CODE (src/)
│
├── src/
│   ├── environments/                    # 交易环境 (Gymnasium兼容)
│   │   ├── __init__.py
│   │   ├── trading_env.py              # TSLA股票交易环境 (FinRL封装)
│   │   │   └── StockTradingEnvOptimized
│   │   │       ├── Future return lookahead reward
│   │   │       ├── Turnover rate reward
│   │   │       ├── Momentum & hold bonuses
│   │   │       └── Drawdown penalty
│   │   └── eurusd_short_env.py        # EURUSD做空交易环境
│   │       └── EURUSDShortSellingEnv
│   │           ├── Short selling (margin 50%)
│   │           ├── Borrowing costs
│   │           └── Symmetric P&L
│   │
│   ├── strategies/                     # 交易策略
│   │   ├── __init__.py
│   │   └── rl_strategy.py             # DRL Lumibot策略
│   │       └── RLStrategyProportional
│   │           ├── Action smoothing (α=0.8)
│   │           └── Position management
│   │
│   └── utils/                         # 工具模块
│       ├── __init__.py
│       ├── config.py                   # TSLA配置
│       │       ├── TICKER = 'TSLA'
│       │       ├── INITIAL_CAPITAL = $100,000
│       │       └── TRANSACTION_FEE = 0.1%
│       ├── config_eurusd.py            # EURUSD配置
│       │       ├── TICKER = 'EURUSD'
│       │       ├── INITIAL_CAPITAL = $1,000,000
│       │       └── LOT_SIZE = 100,000
│       ├── indicators.py               # 技术指标 (TA-Lib)
│       │       ├── MACD (12,26,9)
│       │       ├── Bollinger Bands (20)
│       │       ├── RSI (14)
│       │       ├── CCI (14)
│       │       ├── ADX (14)
│       │       ├── ATR (14)
│       │       └── Williams %R (14)
│       ├── sentiment.py                 # 情绪分析
│       │       ├── Mode 1: OpenAI API
│       │       ├── Mode 2: Alpaca API
│       │       └── Mode 3: Pre-saved CSV
│       ├── metrics.py                   # 性能指标
│       │       ├── Sharpe Ratio
│       │       ├── Sortino Ratio
│       │       ├── Max Drawdown
│       │       └── Win Rate
│       ├── plotting.py                  # 可视化
│       │       └── 8-panel evaluation plot
│       ├── benchmarks.py                # 基准策略
│       │       ├── SMA (5,20) crossover
│       │       └── Buy & Hold
│       └── model_loader.py             # 模型加载
│
├── ════════════════════════════════════════════════════════════════════════════════
│                              ENTRY POINTS (主程序)
│
├── main.py                              # TSLA主程序 (无情绪)
│   └── python main.py --mode all --model a2c
│
├── main_sentiment.py                    # TSLA主程序 (有情绪)
│   └── python main_sentiment.py --mode all --sentiment-mode 3
│
├── eurusd_backtest_final.py             # EURUSD回测程序
│   └── python eurusd_backtest_final.py
│
├── eurusd_ib_trading.py                # EURUSD IB实时交易
│   └── python eurusd_ib_trading.py --paper --duration 10
│
├── run_live_trading.py                  # 实时交易运行器
│
├── ════════════════════════════════════════════════════════════════════════════════
│                              DATA FILES (数据文件)
│
├── data/
│   ├── EURUSD.csv                      # EURUSD 1H OHLCV数据
│   ├── EURUSD_SEN.csv                  # EURUSD情绪数据
│   ├── TSLA.csv                        # TSLA日线OHLCV数据
│   ├── TSLA_SEN.csv                    # TSLA情绪数据 (Mode 3)
│   └── TSLA_SEN_1.csv                  # TSLA情绪数据 (Mode 1)
│
├── trained_models/                      # 训练好的模型
│   ├── best/
│   │   ├── agent_a2c_best.zip         # TSLA A2C最佳模型
│   │   └── agent_a2c_sentiment_best.zip # TSLA A2C情绪模型
│   ├── eurusd/
│   │   ├── a2c_no_sentiment/          # EURUSD无情绪模型
│   │   └── a2c_sentiment/             # EURUSD情绪模型
│   └── tsla/
│       ├── a2c/                       # TSLA A2C模型
│       └── ppo/                        # TSLA PPO模型
│
├── ════════════════════════════════════════════════════════════════════════════════
│                              RESULTS (结果输出)
│
├── tesla_backtest_result/              # TSLA回测结果
│   ├── report.txt                      # 回测报告
│   ├── trade_log_detailed.csv          # 详细交易日志
│   ├── full_evaluation_plot.png        # 8-panel可视化
│   └── COMPREHENSIVE_REPORT.txt       # 综合分析报告
│
├── tesla_backtest_result_sentiment3/   # TSLA情绪回测结果
│   ├── report.txt                      # 含情绪分析报告
│   ├── trade_log_detailed.csv          # 详细交易日志
│   └── full_evaluation_plot.png        # 可视化
│
├── eurusd_backtest_result/             # EURUSD回测结果 (无情绪)
├── eurusd_backtest_result_sentiment/   # EURUSD回测结果 (有情绪)
├── eurusd_ib_trading_result/           # EURUSD IB实时交易结果
└── eurusd_ib_sentiment_result/        # EURUSD IB情绪交易结果
```

---

## 4. 功能结构 (Functional Architecture)

### 4.1 训练流程 (Training Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  Raw Data   │
    │  CSV Files   │
    └──────┬───────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA PREPROCESSING                                                     │
│  ├── Add Technical Indicators (TA-Lib)                                 │
│  ├── Add Sentiment Score (if enabled)                                  │
│  ├── Add Turbulence Index                                              │
│  ├── Train/Test Split: 2018-2023 train, 2024 trade                     │
│  └── Validation Split: Q4 2023                                         │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ENVIRONMENT CREATION                                                   │
│  ├── TSLA: StockTradingEnvOptimized                                    │
│  └── EURUSD: EURUSDShortSellingEnv                                     │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODEL TRAINING (Stable Baselines3)                                    │
│  ├── Algorithm: A2C or PPO                                             │
│  ├── Training Steps: 300,000                                            │
│  ├── Early Stopping: patience=10                                        │
│  └── Best Model Checkpointing                                           │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODEL EXPORT                                                           │
│  └── trained_models/best/agent_*.zip                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 回测流程 (Backtesting Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKTESTING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ Trained Model│
    └──────┬───────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LUMIBOT BACKTESTING BROKER                                            │
│  ├── Historical Data Feed (PandasData)                                 │
│  ├── BacktestingBroker                                                  │
│  └── Day-by-day simulation                                              │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STRATEGY EXECUTION                                                     │
│  ├── RLStrategyProportional                                             │
│  ├── Action Smoothing (α=0.8)                                            │
│  └── Position Management                                                │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PERFORMANCE EVALUATION                                                  │
│  ├── Calculate Metrics (Sharpe, Sortino, MaxDD, WinRate)               │
│  ├── Generate Plots                                                     │
│  ├── Save Trade Logs                                                   │
│  └── Compare vs Benchmarks (SMA, Buy&Hold)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 实时交易流程 (Live Trading Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LIVE TRADING PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  BROKER CONNECTION                                                      │
│  ├── IB: ib_insync + TWS/Gateway (port 7496/7497)                      │
│  └── Alpaca: alpaca-py + API keys (paper/live)                        │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  REAL-TIME DATA FEED                                                    │
│  ├── IB: Real-time bid/ask for EURUSD                                  │
│  └── Alpaca: Real-time quotes for TSLA                                │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STATE CONSTRUCTION                                                     │
│  ├── Current price                                                      │
│  ├── Technical indicators (rolling)                                    │
│  ├── Sentiment score (if enabled)                                      │
│  └── Position tracking                                                  │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODEL INFERENCE                                                        │
│  ├── Load trained model                                                │
│  ├── Predict action                                                     │
│  └── Apply action smoothing                                            │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ORDER EXECUTION                                                        │
│  ├── Market orders                                                      │
│  ├── Limit orders (IB only)                                            │
│  └── Position update                                                    │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MONITORING & LOGGING                                                   │
│  ├── Real-time P&L                                                     │
│  ├── Trade logging                                                     │
│  └── Account value tracking                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 关键模块详解 (Key Modules)

### 5.1 EURUSDShortSellingEnv

```python
class EURUSDShortSellingEnv(gym.Env):
    """
    EURUSD外汇交易环境 - 支持真正的做空交易

    Action Interpretation:
    ┌───────────┬────────────────────────────────────────┐
    │  Action   │            Meaning                     │
    ├───────────┼────────────────────────────────────────┤
    │  -1.0     │  Full short (100% capital value)     │
    │  -0.5     │  Semi-short (50% capital value)       │
    │   0.0     │  Flat (no position)                   │
    │  +0.5     │  Semi-long (50% capital value)        │
    │  +1.0     │  Full long (100% capital value)       │
    └───────────┴────────────────────────────────────────┘

    Key Features:
    • 初始保证金: 50% of short position value
    • 维护保证金: 25%
    • 借贷成本: ~0.01%/day (3.65% annual)
    • 最大杠杆: 2x
    """
```

### 5.2 StockTradingEnvOptimized

```python
class StockTradingEnvOptimized(StockTradingEnv):
    """
    TSLA股票交易环境 - 增强奖励函数

    Enhanced Reward Components:
    ┌──────────────────┬───────────────────────────────────┐
    │    Component     │              Effect               │
    ├──────────────────┼───────────────────────────────────┤
    │ Return Reward    │ Base portfolio return × 100      │
    │ Future Reward    │ Lookahead reward for good timing │
    │ Turnover Reward  │ Encourages stability              │
    │ Momentum Reward  │ +0.3 × return × 100 (if winning) │
    │ Hold Bonus       │ +0.005 (if holding winning pos)  │
    │ Sell Encouragement│ Take profits when > 2%          │
    │ Drawdown Penalty │ -0.1 × (dd - threshold)         │
    └──────────────────┴───────────────────────────────────┘
    """
```

### 5.3 RLStrategyProportional

```python
class RLStrategyProportional(Strategy):
    """
    Lumibot DRL交易策略

    Action Smoothing:
    final_action = α × raw_action + (1-α) × last_action
                  = 0.8 × raw_action + 0.2 × last_action

    Position Calculation:
    target_value = portfolio_value × target_ratio
    target_shares = target_value / current_price
    """
```

---

## 6. 情绪分析系统 (Sentiment Analysis System)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ANALYSIS SYSTEM                            │
│                    情绪分析系统 (3 Modes)                               │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  User选择模式   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐        ┌──────────┐        ┌─────────┐
    │ Mode 1  │        │  Mode 2  │        │ Mode 3  │
    │OpenAI API│        │ Alpaca   │        │CSV文件  │
    └────┬────┘        └────┬─────┘        └────┬────┘
         │                   │                  │
         ▼                   ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM SENTIMENT ANALYSIS                          │
│  Model: qwen3-coder:480b (via Ollama) or GPT-4                       │
│  Output: -1.0 (very bearish) to +1.0 (very bullish)                │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SENTIMENT FEATURE VECTOR                           │
│  sentiment_score → [sentiment] → State Space                        │
│  Range: [-1.0, +1.0]                                                │
└─────────────────────────────────────────────────────────────────────┘

MODE COMPARISON:
┌─────────┬──────────────┬──────────────┬────────────────────────────┐
│  Mode   │    Source    │    Speed     │         Accuracy          │
├─────────┼──────────────┼──────────────┼────────────────────────────┤
│    1    │ OpenAI API   │    Slow      │    High (GPT-4)           │
│    2    │ Alpaca API   │   Medium     │    Medium                │
│    3    │ Pre-saved CSV│    Fast      │    Depends on data        │
└─────────┴──────────────┴──────────────┴────────────────────────────┘
```

---

## 7. 性能对比 (Performance Comparison)

### TSLA 2024回测结果

```
┌─────────────────────────┬────────────────┬────────────────┬────────────────┐
│        Metric           │ DRL (无情绪)   │ DRL (有情绪)   │   Buy&Hold    │
├─────────────────────────┼────────────────┼────────────────┼────────────────┤
│ Total Return            │ +48.83%        │ +62.45%        │ +73.76%       │
│ Sharpe Ratio           │ 0.85           │ 1.03           │ 1.18          │
│ Max Drawdown           │ 36.23%         │ 41.46%         │ 42.82%        │
│ Annual Volatility       │ 58.13%         │ 61.54%         │ 63.42%        │
│ Total Trades           │ 197            │ 156            │ 1             │
│ Final Value            │ $148,827       │ $162,453       │ $173,762      │
└─────────────────────────┴────────────────┴────────────────┴────────────────┘
```

### EURUSD 2024回测结果

```
┌─────────────────────────┬────────────────┬────────────────┬────────────────┐
│        Metric           │ DRL (无情绪)   │ DRL (有情绪)   │   SMA(5,20)    │
├─────────────────────────┼────────────────┼────────────────┼────────────────┤
│ Total Return            │ -2.27%         │ -1.89%         │ +1.23%        │
│ Sharpe Ratio           │ -0.78          │ -0.77          │ +0.18         │
│ Max Drawdown           │ 3.31%          │ 2.78%          │ 3.02%         │
│ Annual Volatility       │ 2.22%          │ 1.87%          │ N/A           │
└─────────────────────────┴────────────────┴────────────────┴────────────────┘
```

---

## 8. 目录结构树 (Directory Tree)

```
TestMinimax/
├── src/
│   ├── environments/
│   │   ├── trading_env.py           # StockTradingEnvOptimized
│   │   └── eurusd_short_env.py     # EURUSDShortSellingEnv
│   ├── strategies/
│   │   └── rl_strategy.py          # RLStrategyProportional
│   └── utils/
│       ├── config.py               # TSLA配置
│       ├── config_eurusd.py       # EURUSD配置
│       ├── indicators.py          # 技术指标
│       ├── sentiment.py           # 情绪分析
│       ├── metrics.py             # 性能指标
│       ├── plotting.py            # 可视化
│       ├── benchmarks.py          # 基准策略
│       └── model_loader.py        # 模型加载
│
├── data/
│   ├── EURUSD.csv
│   ├── EURUSD_SEN.csv
│   ├── TSLA.csv
│   ├── TSLA_SEN.csv
│   └── TSLA_SEN_1.csv
│
├── scripts/                        # (已整合到主程序)
│
├── trained_models/
│   ├── best/
│   ├── eurusd/
│   └── tsla/
│
├── results/
│   ├── backtest/
│   ├── live/
│   └── comparison/
│
├── docs/
│   ├── architecture.tex
│   └── QCS1_Profile_副本.pdf
│
├── tests/
│   ├── test_indicators.py
│   └── test_metrics.py
│
├── main.py                        # TSLA主程序 (无情绪)
├── main_sentiment.py             # TSLA主程序 (有情绪)
├── eurusd_backtest_final.py      # EURUSD回测
├── eurusd_ib_trading.py          # EURUSD实时交易
├── run_live_trading.py           # 实时交易运行器
│
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

---

## 9. 依赖关系 (Dependencies)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PYTHON DEPENDENCIES                             │
└─────────────────────────────────────────────────────────────────────────┘

Training & ML:
├── stable-baselines3>=2.0.0          # A2C/PPO algorithms
├── finrl>=0.3.0                     # Deep RL training framework
├── torch>=2.0.0                     # PyTorch backend
└── gymnasium>=0.29.0                # Gymnasium environments

Backtesting:
├── lumibot>=4.0.0                   # Backtesting broker
└── pandas>=2.0.0                    # Data manipulation

Live Trading:
├── ib_insync>=0.9.86                # Interactive Brokers API
└── alpaca-py>=0.20.0                # Alpaca trading API

Indicators:
└── talib-binary>=0.5.0              # Technical analysis library

Visualization:
├── matplotlib>=3.7.0
└── seaborn>=0.12.0

Utilities:
├── numpy>=1.24.0
├── pandas>=2.0.0
└── scikit-learn>=1.3.0
```

---

## 10. 快速开始 (Quick Start)

### TSLA回测

```bash
# 无情绪版本
python main.py --mode all --model a2c

# 有情绪版本 (Mode 3 - CSV)
python main_sentiment.py --mode all --sentiment-mode 3 --model a2c
```

### EURUSD回测

```bash
# 无情绪版本
python eurusd_backtest_final.py

# 有情绪版本
python eurusd_backtest_final.py --sentiment
```

### EURUSD实时交易 (需要IB TWS)

```bash
# Paper交易
python eurusd_ib_trading.py --paper --duration 60

# 实盘交易
python eurusd_ib_trading.py --live --duration 60

# 带情绪
python eurusd_ib_trading.py --sentiment --paper --duration 30
```

---

## END OF DOCUMENT
