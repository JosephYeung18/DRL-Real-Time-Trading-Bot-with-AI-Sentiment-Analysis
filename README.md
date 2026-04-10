# QCS - Quantitative Commodity Trading System
# 量化商品交易系统

基于深度强化学习的量化交易系统，支持EURUSD外汇和TSLA股票交易，支持有无情绪增强两种模式。

---

## 系统架构

```
QCS/
├── src/                            # 源代码
│   ├── environments/               # 交易环境 (Gymnasium)
│   │   ├── __init__.py
│   │   ├── eurusd_short_env.py    # EURUSD做空交易环境
│   │   └── trading_env.py         # FinRL股票交易环境封装
│   │
│   ├── models/                    # 模型定义
│   │   └── __init__.py
│   │
│   ├── strategies/                # 交易策略
│   │   ├── __init__.py
│   │   ├── rl_strategy.py        # DRL Lumibot策略
│   │   ├── benchmarks.py          # SMA/买入持有基准
│   │   └── metrics.py             # 性能评估指标
│   │
│   └── utils/                     # 工具模块
│       ├── __init__.py
│       ├── config.py              # TSLA配置
│       ├── config_eurusd.py       # EURUSD配置
│       ├── indicators.py          # 技术指标
│       ├── benchmarks.py          # 基准策略
│       ├── metrics.py             # 性能指标
│       ├── plotting.py            # 可视化
│       ├── sentiment.py           # 情绪分析
│       └── model_loader.py        # 模型加载
│
├── scripts/                        # 可执行脚本
│   ├── backtest_eurusd.py          # EURUSD回测
│   ├── backtest_eurusd_sentiment.py # EURUSD回测(情绪增强)
│   ├── live_ib_eurusd.py          # EURUSD IB实时交易
│   ├── backtest_tsla.py           # TSLA回测
│   ├── backtest_tsla_sentiment.py # TSLA回测(情绪增强)
│   └── live_trading_tsla.py       # TSLA Alpaca实时交易
│
├── data/                          # 数据文件
│   ├── EURUSD.csv                # EURUSD历史数据
│   ├── EURUSD_SEN.csv            # EURUSD情绪数据
│   ├── TSLA.csv                  # TSLA股票数据
│   ├── TSLA_SEN.csv              # TSLA情绪数据
│   └── TSLA_SEN_1.csv            # TSLA情绪数据(Headlines模式)
│
├── trained_models/                # 训练好的模型
│   ├── eurusd/                   # EURUSD模型
│   ├── eurusd_sentiment/         # EURUSD情绪模型
│   └── tsla/                     # TSLA模型
│
├── results/                       # 实验结果
│   ├── backtest/                 # 回测结果
│   ├── live/                     # 实时交易结果
│   └── comparison/               # 对比分析报告
│
├── docs/                         # 文档
│   ├── architecture.tex          # 系统架构文档
│   ├── QCS1_Profile_副本.pdf    # 系统详细说明
│   └── methodology.pdf           # 方法论文档
│
├── papers/                        # 学术论文参考
│   └── README.md
│
├── tests/                         # 单元测试
│   ├── __init__.py
│   ├── test_indicators.py
│   └── test_metrics.py
│
├── CHANGELOG.md                   # 变更日志
├── CONTRIBUTING.md                # 贡献指南
├── README.md                      # 项目说明
├── requirements.txt               # Python依赖
└── setup.py                      # 安装脚本
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行EURUSD回测

```bash
# 无情绪版本
python scripts/backtest_eurusd.py

# 有情绪版本
python scripts/backtest_eurusd_sentiment.py
```

结果保存在 `results/backtest/eurusd/`

### 3. 运行EURUSD实时交易 (Interactive Brokers)

**前提条件**:
1. 下载安装 [Interactive Brokers TWS](https://www.interactivebrokers.com/en/trading/downloads.php) 或 IB Gateway
2. 启动 TWS/Gateway 并启用 API 连接
3. 配置端口: 7496 (Paper) 或 7497 (Live)

```bash
# Paper交易 (模拟)
python scripts/live_ib_eurusd.py --paper --duration 10

# 实时交易 (需要真实账户)
python scripts/live_ib_eurusd.py --duration 60

# 带情绪增强
python scripts/live_ib_eurusd.py --sentiment --duration 30
```

结果保存在 `results/live/eurusd/`

### 4. 运行TSLA回测

```bash
# 无情绪
python scripts/backtest_tsla.py --mode backtest

# 有情绪 (模式3-CSV)
python scripts/backtest_tsla_sentiment.py --mode backtest --sentiment-mode 3
```

### 5. 运行TSLA实时交易 (Alpaca)

**前提条件**:
1. 注册 [Alpaca 账号](https://app.alpaca.markets/)
2. 获取 API Key 和 Secret Key
3. 设置环境变量或配置文件

```bash
# 设置API密钥
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"

# Paper交易模式
python scripts/live_trading_tsla.py --mode paper --duration 60

# 实时交易模式 (需要真实账户)
python scripts/live_trading_tsla.py --mode live --duration 60

# 带情绪增强
python scripts/live_trading_tsla.py --mode paper --sentiment --duration 30
```

结果保存在 `results/live/tsla/`

---

## 核心功能

### 1. EURUSD Forex交易

**自定义做空交易环境** `EURUSDShortSellingEnv`:
- 支持真正的做空交易(从零仓位开空)
- 对称的多空奖惩机制
- 杠杆/保证金管理(2x杠杆)
- 借贷成本计算

**技术指标**: MACD, Bollinger Bands, RSI, CCI, ADX, ATR, Williams %R

**Interactive Brokers 实时交易**:
- 支持真实EURUSD外汇交易
- 实时市场数据订阅 (Bid/Ask)
- 市价单和限价单执行
- 支持Paper交易 (端口7496) 和 Live交易 (端口7497)

### 2. TSLA股票交易

基于FinRL和Lumibot的交易系统:
- A2C/PPO强化学习算法
- 三种情绪增强模式
- 完整的风险管理系统

**Alpaca API 实时交易**:
- 支持TSLA股票实时交易
- Paper/Live双模式
- 实时订单执行和持仓管理

### 3. 情绪分析

| 模式 | 数据来源 | 配置要求 |
|------|----------|---------|
| 模式1 | OpenAI API + headlines | TSLA_SEN_1.csv |
| 模式2 | Alpaca API + LLM | API Key |
| 模式3 | 预存CSV (默认，最快) | TSLA_SEN.csv |

### 4. 技术指标

| 指标 | 参数 | 说明 |
|------|------|------|
| MACD | 12,26,9 | 趋势跟踪 |
| Bollinger Bands | 20 | 波动性 |
| RSI | 14 | 超买超卖 |
| CCI | 14 | 顺势指标 |
| ADX | 14 | 趋势强度 |
| ATR | 14 | 真实波幅 |
| Williams %R | 14 | 超买超卖 |
| 情绪分数 | -1 to +1 | 新闻情绪 |

---

## 性能表现

### EURUSD 2024回测

| 策略 | 总收益率 | 最大回撤 | 夏普比率 |
|------|----------|----------|----------|
| SMA(5,20) | +1.23% | 3.02% | 0.18 |
| DRL+情绪 | -2.11% | 3.09% | -0.78 |
| Buy&Hold | -5.36% | 7.50% | -0.80 |

### EURUSD IB 实时交易 (10分钟)

| 策略 | 总收益率 | 夏普比率 | 交易次数 |
|------|----------|----------|----------|
| DRL (无情绪) | -0.00% | 30.70 | 1 |
| DRL (带情绪) | -0.00% | 23.36 | 1 |

### TSLA 2024回测

| 策略 | 总收益率 | 夏普比率 | 最大回撤 |
|------|----------|----------|----------|
| **DRL模型** | **+77.67%** | **1.25** | 38.87% |
| SMA(5,20) | +48.05% | 1.04 | 31.69% |
| 买入持有 | +73.76% | 1.18 | 42.82% |

---

## 项目结构说明

### 交易环境 (`src/environments/`)

| 文件 | 说明 |
|------|------|
| `eurusd_short_env.py` | EURUSD自定义做空环境 |
| `trading_env.py` | FinRL StockTradingEnv封装 |

### 策略模块 (`src/strategies/`)

| 文件 | 说明 |
|------|------|
| `rl_strategy.py` | Lumibot DRL交易策略 |
| `benchmarks.py` | SMA/买入持有基准 |
| `metrics.py` | 性能评估指标 |

### 工具模块 (`src/utils/`)

| 文件 | 说明 |
|------|------|
| `indicators.py` | 技术指标计算 |
| `sentiment.py` | 情绪分析处理 |
| `config.py` | TSLA系统配置 |
| `config_eurusd.py` | EURUSD配置 |
| `model_loader.py` | 模型加载 |

### 可执行脚本 (`scripts/`)

| 脚本 | 说明 |
|------|------|
| `backtest_eurusd.py` | EURUSD回测 (无情绪) |
| `backtest_eurusd_sentiment.py` | EURUSD回测 (带情绪) |
| `live_ib_eurusd.py` | EURUSD IB实时交易 |
| `backtest_tsla.py` | TSLA回测 (无情绪) |
| `backtest_tsla_sentiment.py` | TSLA回测 (带情绪) |
| `live_trading_tsla.py` | TSLA Alpaca实时交易 |

---

## 训练配置

| 参数 | EURUSD | TSLA |
|------|--------|------|
| 初始资金 | $100,000 | $100,000 |
| 交易费用 | 0.01% | 0.1% |
| 算法 | A2C | A2C |
| 训练步数 | 300,000 | 300,000 |
| 状态空间 | 15 | 16 |
| 杠杆 | 2x | 1x |
| 做空 | 支持 | 不支持 |
| 交易API | Interactive Brokers | Alpaca |

---

## API配置

### Interactive Brokers (EURUSD)
```bash
# IB Gateway/TWS设置
# 1. File > Global Configuration > API > Settings
# 2. Enable Socket Clients
# 3. Port: 7496 (Paper) / 7497 (Live)
```

### Alpaca (TSLA)
```bash
# 注册: https://app.alpaca.markets/
# 获取API密钥后设置环境变量
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
```

---

## 注意事项

1. EURUSD回测使用自定义环境，需使用 `scripts/backtest_eurusd.py`
2. EURUSD实时交易使用 `scripts/live_ib_eurusd.py` (需IB TWS/Gateway)
3. TSLA回测使用 `scripts/backtest_tsla.py` 或 `scripts/backtest_tsla_sentiment.py`
4. TSLA实时交易使用 `scripts/live_trading_tsla.py` (需Alpaca API密钥)
5. 过去表现不代表未来收益

---

## 学术引用

如果您在学术工作中使用本系统，请引用:

```bibtex
@software{QCS2024,
  title = {QCS - Quantitative Commodity Trading System},
  author = {Joseph Yeung},
  year = {2024},
  url = {https://github.com/JosephYeung18/DRL-Real-Time-Trading-Bot-with-AI-Sentiment-Analysis}
}
```

更多学术参考请见 `papers/README.md`
