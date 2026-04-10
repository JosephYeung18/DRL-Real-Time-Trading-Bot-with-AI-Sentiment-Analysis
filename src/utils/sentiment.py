"""
Technical Indicators and Data Preparation Module with Sentiment
Supports three sentiment modes:
- llm_model=1: Use OpenAI API to get sentiment from headlines
- llm_model=2: Use Alpaca API to get news sentiment
- llm_model=3: Use pre-saved sentiment CSV (default)
"""

import numpy as np
import pandas as pd
import talib
from finrl.config import INDICATORS
from .config import TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE

def add_custom_indicators(df):
    """Add custom technical indicators using TA-Lib matching FinRL conventions"""
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

    # Bollinger Bands (FinRL naming)
    df['boll_ub'], df['boll_lb'], df['boll_middle'] = talib.BBANDS(close, timeperiod=20)

    # RSI (FinRL naming - RSI with 30 period)
    df['rsi_30'] = talib.RSI(close, timeperiod=14)

    # CCI (FinRL naming)
    df['cci_30'] = talib.CCI(high, low, close, timeperiod=14)

    # DX (FinRL naming - ADX component)
    df['dx_30'] = talib.DX(high, low, close, timeperiod=14)

    # Moving Averages (FinRL naming)
    df['close_30_sma'] = talib.SMA(close, timeperiod=30)
    df['close_60_sma'] = talib.SMA(close, timeperiod=60)

    # Additional indicators
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)
    df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)

    # Sentiment score (momentum-based if not already present)
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = (df['close'] - df['open']) / df['open']

    return df

def add_turbulence_index(df):
    """Add turbulence index for risk assessment"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['turbulence'] = (df['returns'] - df['returns'].rolling(20).mean()) / df['returns'].rolling(20).std()
    df['turbulence'] = df['turbulence'].fillna(0)
    return df

def add_user_defined_features(df):
    """Add user-defined features"""
    df = df.copy()

    # Price momentum
    df['momentum_5'] = df['close'].pct_change(5)

    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_5'] = df['returns'].rolling(5).std()

    # Price relative to moving averages
    df['price_to_sma5'] = df['close'] / df['close_30_sma']
    df['price_to_sma20'] = df['close'] / df['close_60_sma']

    return df

def prepare_data_with_sentiment(data_csv='data/TSLA.csv', sentiment_csv='data/TSLA_SEN.csv',
                               llm_model=3, train_data_path=None):
    """
    Prepare data with technical indicators and sentiment

    Args:
        data_csv: Stock price data CSV
        sentiment_csv: Sentiment data CSV
        llm_model: Sentiment mode (1=OpenAI, 2=Alpaca, 3=CSV)
        train_data_path: Optional path to training data (for Alpaca mode)

    Returns:
        train_data, trade_data, indicators
    """
    print("=" * 60)
    print("Preparing data with sentiment...")
    print("=" * 60)

    # Load stock data
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} records from {data_csv}")

    # Standardize column names
    column_mapping = {
        'TradingDate': 'date',
        'ClosePrice': 'close',
        'OpenPrice': 'open',
        'HighPrice': 'high',
        'LowPrice': 'low',
        'Volume': 'volume',
    }
    df = df.rename(columns=column_mapping)

    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.drop('datetime', axis=1)

    df = df.sort_values('date').reset_index(drop=True)

    # Add 'tic' column for FinRL compatibility
    df['tic'] = 'TSLA'

    # ========== Add Sentiment Based on Mode ==========
    if llm_model == 1:
        # Mode 1: Use OpenAI API (requires API key and Ollama)
        # For testing, use TSLA_SEN_1.csv with 10 sample headlines
        print("Sentiment mode 1: Using OpenAI API (LLM re-analysis)")
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key="68580f17da5b453d9f6fabb42a5ec734.Py5Beu8dIUdIYnG3_4wwGT3J",
                base_url="https://ollama.com/v1"
            )

            # Use test file TSLA_SEN_1.csv for mode 1, fallback to sentiment_csv
            sentiment_csv_1 = sentiment_csv.replace('.csv', '_1.csv')
            try:
                df_sen = pd.read_csv(sentiment_csv_1)
            except FileNotFoundError:
                print(f"  Test file {sentiment_csv_1} not found, using {sentiment_csv}")
                df_sen = pd.read_csv(sentiment_csv)

            if 'headline' in df_sen.columns and 'sentiment_score' in df_sen.columns:
                df_sen = df_sen.drop(['sentiment_score'], axis=1)

            # Convert date to datetime to match main data
            df_sen['date'] = pd.to_datetime(df_sen['date'])

            print(f"  Analyzing {len(df_sen)} headlines with LLM...")
            df_sen['sentiment_score'] = df_sen['headline'].apply(
                lambda x: get_sentiment_score_llm(x, client)
            )
            df = pd.merge(df, df_sen, how='left', on=['date'])
            df['sentiment_score'] = df['sentiment_score'].fillna(0)
        except Exception as e:
            print(f"OpenAI sentiment failed: {e}, using momentum-based sentiment")
            df['sentiment_score'] = (df['close'] - df['open']) / df['open']

    elif llm_model == 2:
        # Mode 2: Use Alpaca API (requires API keys)
        print("Sentiment mode 2: Using Alpaca API")
        import os
        import sys
        try:
            from alpaca_trade_api import REST

            # 从环境变量读取API密钥，如果没有则使用默认值
            ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'PKGUSBUYZ5DIJLVQ3ADSJ4UF6F')
            ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET', 'AWTGa9Mb1tGspDLwZmHpoAwG9BC51KqMk1NejbXZFW7Q')
            rest_client = REST(ALPACA_API_KEY, ALPACA_API_SECRET)

            print(f"  Alpaca API Key: {ALPACA_API_KEY[:8]}...")

            # 按月获取新闻，避免请求过大
            time_data = pd.date_range(start=TRAIN_START_DATE, end=TRADE_END_DATE, freq='D').to_frame().reset_index(drop=True).rename(columns={0: 'date'})
            time_data['month'] = pd.to_datetime(time_data['date']).dt.to_period('M')
            time_data1 = time_data.drop_duplicates(subset=['month'], keep='first').rename(columns={'date': 'date1'})
            time_data2 = time_data.drop_duplicates(subset=['month'], keep='last').rename(columns={'date': 'date2'})
            time_data = pd.merge(time_data1, time_data2, how='inner', on=['month'])
            time_data['date1'] = pd.to_datetime(time_data['date1']).dt.date.astype(str)
            time_data['date2'] = pd.to_datetime(time_data['date2']).dt.date.astype(str)
            time_list = list(zip(time_data['date1'].to_list(), time_data['date2'].to_list()))

            print(f"  Fetching news from {len(time_list)} months...")

            df_sen = pd.DataFrame()
            for i, (start, end) in enumerate(time_list):
                try:
                    news = rest_client.get_news(symbol="TSLA", start=start, end=end, limit=1000)
                    if news:
                        newst = pd.DataFrame([[n.created_at, n.headline, n.summary] for n in news],
                                            columns=['created_at', 'headline', 'summary'])
                        newst['date'] = newst['created_at'].apply(lambda x: str(x).split(' ')[0])
                        df_sen = pd.concat([df_sen, newst[['date', 'headline', 'summary']]], axis=0)
                    if (i + 1) % 6 == 0:
                        print(f"  Fetched {i+1}/{len(time_list)} months of news...")
                except Exception as e:
                    print(f"  Warning: Failed to fetch news for {start} to {end}: {e}")
                    continue

            if df_sen.empty:
                raise Exception("No news data fetched from Alpaca API")

            print(f"  Total news articles fetched: {len(df_sen)}")

            # 按日期聚合新闻，合并标题和摘要进行情绪分析
            df_sen['date'] = pd.to_datetime(df_sen['date'])
            df_sen_grouped = df_sen.groupby('date').agg({
                'headline': lambda x: ' '.join(x),
                'summary': lambda x: ' '.join([str(s) for s in x if pd.notna(s)])
            }).reset_index()
            df_sen_grouped['text'] = df_sen_grouped['headline'] + '. ' + df_sen_grouped['summary']

            print(f"  Analyzing sentiment for {len(df_sen_grouped)} unique dates...")

            # 批量获取情绪分数
            sentiments = []
            for j, text in enumerate(df_sen_grouped['text'].values):
                sentiment = get_sentiment_score_llm(text[:500])  # 限制文本长度
                sentiments.append(sentiment)
                if (j + 1) % 10 == 0:
                    print(f"  Processed {j+1}/{len(df_sen_grouped)} dates...")

            df_sen_grouped['sentiment_score'] = sentiments
            df_sen_result = df_sen_grouped[['date', 'sentiment_score']]

            # 保存获取的新闻和情绪分数供以后使用
            news_cache_path = sentiment_csv.replace('.csv', '_alpaca_news.csv')
            df_sen_grouped.to_csv(news_cache_path, index=False)
            print(f"  News cache saved to: {news_cache_path}")

            df = pd.merge(df, df_sen_result, how='left', on=['date'])
            df['sentiment_score'] = df['sentiment_score'].fillna(0)

        except ImportError:
            print("Alpaca SDK not installed. Please run: pip install alpaca-trade-api")
            df['sentiment_score'] = (df['close'] - df['open']) / df['open']
        except Exception as e:
            print(f"Alpaca sentiment failed: {e}, using momentum-based sentiment")
            df['sentiment_score'] = (df['close'] - df['open']) / df['open']

    else:
        # Mode 3: Use pre-saved sentiment CSV (default)
        print("Sentiment mode 3: Using pre-saved sentiment CSV")
        try:
            df_sen = pd.read_csv(sentiment_csv)
            # Keep only date and sentiment_score
            if 'headline' in df_sen.columns:
                df_sen = df_sen.drop(['headline'], axis=1)
            # Convert sentiment date to datetime to match main data
            df_sen['date'] = pd.to_datetime(df_sen['date'])
            df = pd.merge(df, df_sen, how='left', on=['date'])
            df['sentiment_score'] = df['sentiment_score'].fillna(0)
            # Handle potential string format issues
            df['sentiment_score'] = df['sentiment_score'].apply(lambda x: float(str(x).split('\n')[0]) if isinstance(x, str) else float(x))
        except Exception as e:
            print(f"Sentiment CSV load failed: {e}, using momentum-based sentiment")
            df['sentiment_score'] = (df['close'] - df['open']) / df['open']

    print(f"Sentiment score range: {df['sentiment_score'].min():.3f} to {df['sentiment_score'].max():.3f}")

    # Add indicators
    df = add_custom_indicators(df)
    print("Added technical indicators")

    df = add_turbulence_index(df)
    print("Added turbulence index")

    df = add_user_defined_features(df)
    print("Added user defined features")

    # Split data
    from finrl.meta.preprocessor.preprocessors import data_split
    train_data = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
    trade_data = data_split(df, TRADE_START_DATE, TRADE_END_DATE)

    # Indicators list - match original FinRL configuration with sentiment
    indicators = INDICATORS + ['cci', 'willr', 'adx', 'atr', 'sentiment_score']

    print(f"\nTraining data: {len(train_data)} records")
    print(f"Trading data: {len(trade_data)} records")
    print(f"Indicators ({len(indicators)}): {indicators}")

    return train_data, trade_data, indicators


def get_sentiment_score_llm(text, client=None):
    """Get sentiment score from LLM (for modes 1 and 2)

    Args:
        text: News headline or summary to analyze
        client: Optional OpenAI client

    Returns:
        float: Sentiment score between -1 (very negative) and +1 (very positive)
    """
    try:
        if client is None:
            # Default client setup
            from openai import OpenAI
            client = OpenAI(
                api_key="68580f17da5b453d9f6fabb42a5ec734.Py5Beu8dIUdIYnG3_4wwGT3J",
                base_url="https://ollama.com/v1"
            )

        # Optimized system message for financial sentiment analysis
        system_message = """You are an expert financial analyst specializing in stock market sentiment analysis for Tesla (TSLA).

Your task: Analyze the sentiment of financial news headlines and return a single number.

SCORING:
- +1.0 to +0.6: VERY BULLISH - Strong positive news (earnings beat, new contracts, product launches, analyst upgrades)
- +0.5 to +0.3: BULLISH - Moderate positive news (growth indicators, market expansion, partnership announcements)
- +0.2 to +0.1: SLIGHTLY POSITIVE - Minor positive news or optimistic outlook
- 0.0: NEUTRAL - Factual news without clear direction, mixed signals
- -0.1 to -0.2: SLIGHTLY NEGATIVE - Minor concerns or cautious outlook
- -0.3 to -0.5: BEARISH - Moderate negative news (regulatory issues, competition concerns, delivery misses)
- -0.6 to -1.0: VERY BEARISH - Strong negative news (investigation, recalls, major losses, executive departures)

IMPORTANT:
1. Consider the IMPACT on stock price and investor sentiment
2. Weight recent events more heavily
3. Consider both direct and indirect effects on Tesla
4. Return ONLY the numeric score, no explanation, no text
"""

        response = client.chat.completions.create(
            model="qwen3-coder:480b",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Analyze this Tesla news and return sentiment score:\n\n{text}"}
            ],
            temperature=0.1,
            max_tokens=8
        )
        content = response.choices[0].message.content.strip()
        # 尝试提取数字
        import re
        match = re.search(r'[-+]?\d*\.?\d+', content)
        if match:
            score = float(match.group())
            # 确保分数在 [-1, 1] 范围内
            return max(-1, min(1, score))
        return 0.0
    except Exception as e:
        print(f"LLM error: {e}")
        return 0.0
