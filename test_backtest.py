import pytest
import pandas as pd
from trading_strategy import backtest
from utils import check_volatility

def test_backtest():
    df = pd.DataFrame({
        'Close': [100, 101, 102, 103],
        'RSI': [50, 60, 70, 80],
        'MACD': [0, 0.1, 0.2, 0.3]
    })
    result = backtest(df, lambda x: "買入" if x['RSI'] > 60 else "持有")
    assert isinstance(result, dict)
    assert "sharpe_ratio" in result
    assert result['final_capital'] >= 0

def test_high_volatility():
    atr = 0.03
    assert not check_volatility(atr)  # 測試邊緣案例