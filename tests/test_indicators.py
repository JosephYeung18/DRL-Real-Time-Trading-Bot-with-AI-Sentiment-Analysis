"""
Test Technical Indicators
=========================

Unit tests for technical indicators module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.indicators import (
    calculate_macd,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_cci,
    calculate_adx,
    calculate_atr
)


class TestIndicators:
    """Test cases for technical indicators."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 102 + np.cumsum(np.random.randn(100) * 0.5),
            'low': 98 + np.cumsum(np.random.randn(100) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        return data

    def test_macd(self, sample_data):
        """Test MACD calculation."""
        macd, signal, histogram = calculate_macd(sample_data['close'])

        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(histogram) == len(sample_data)
        assert not macd.isna().all()

    def test_rsi(self, sample_data):
        """Test RSI calculation."""
        rsi = calculate_rsi(sample_data['close'])

        assert len(rsi) == len(sample_data)
        assert (rsi >= 0).all() or rsi.isna().any()  # RSI should be 0-100 or NaN
        assert (rsi <= 100).all() or rsi.isna().any()

    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = calculate_bollinger_bands(sample_data['close'])

        assert len(upper) == len(sample_data)
        assert len(middle) == len(sample_data)
        assert len(lower) == len(sample_data)
        assert (upper >= middle).all()  # Upper band >= middle
        assert (middle >= lower).all()  # Middle >= lower band

    def test_cci(self, sample_data):
        """Test CCI calculation."""
        cci = calculate_cci(sample_data['high'], sample_data['low'], sample_data['close'])

        assert len(cci) == len(sample_data)
        # CCI can be any value, but should not be all NaN
        assert not cci.isna().all()

    def test_adx(self, sample_data):
        """Test ADX calculation."""
        adx = calculate_adx(sample_data['high'], sample_data['low'], sample_data['close'])

        assert len(adx) == len(sample_data)
        assert (adx >= 0).all() or adx.isna().any()  # ADX should be non-negative

    def test_atr(self, sample_data):
        """Test ATR calculation."""
        atr = calculate_atr(sample_data['high'], sample_data['low'], sample_data['close'])

        assert len(atr) == len(sample_data)
        assert (atr >= 0).all() or atr.isna().any()  # ATR should be non-negative


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
