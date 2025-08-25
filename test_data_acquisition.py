import pytest
import pandas as pd
from data_acquisition import fetch_data, compute_indicators

def test_fetch_data():
    df = fetch_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Close' in df.columns

def test_compute_indicators():
    df = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [101, 102, 103],
        'Low': [99, 100, 101],
        'Close': [100, 101, 102]
    })
    df = compute_indicators(df)
    assert 'RSI' in df.columns
    assert 'MACD' in df.columns
    assert 'ATR' in df.columns
    assert 'EMA' in df.columns  # 測試補充指標