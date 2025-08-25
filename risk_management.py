import xgboost as xgb
import pandas as pd
import logging

def calculate_stop_loss(current_price: float, atr: float, multiplier: float = 2) -> float:
    """計算止損：基於 ATR 動態調整。
    邏輯：長倉減去 multiplier * ATR。
    """
    return current_price - (multiplier * atr)

def calculate_take_profit(current_price: float, atr: float, multiplier: float = 2) -> float:
    """計算止盈：基於 ATR 動態調整。
    邏輯：長倉加上 multiplier * ATR。
    """
    return current_price + (multiplier * atr)

def calculate_position_size(capital: float, risk_percent: float, stop_loss_distance: float) -> float:
    """計算倉位大小：控制風險。
    邏輯：(資本 * 風險百分比) / 止損距離。
    """
    return (capital * risk_percent) / stop_loss_distance if stop_loss_distance > 0 else 0

def predict_volatility(df: pd.DataFrame) -> float:
    """預測波動性：使用 XGBoost。
    邏輯：訓練簡單模型預測未來 ATR。
    """
    try:
        X = df[['Close', 'RSI', 'MACD']].iloc[:-1]
        y = df['ATR'].shift(-1).dropna()
        model = xgb.XGBRegressor()
        model.fit(X, y)
        pred = model.predict(df[['Close', 'RSI', 'MACD']].iloc[-1:])
        return pred[0]
    except Exception as e:
        logging.error(f"波動預測錯誤: {e}")
        return df['ATR'].mean()