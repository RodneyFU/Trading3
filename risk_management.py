import xgboost as xgb
import pandas as pd
import logging
from psutil import virtual_memory, cpu_percent
import onnxruntime as ort
import numpy as np
from pathlib import Path
import aiosqlite
from datetime import datetime
from ai_models import FEATURES
async def get_current_exposure(db_path: str) -> float:
    """獲取當前總持倉暴露（以美元計）。"""
    # 函數說明：從資料庫查詢當前持倉總暴露值。
    try:
        async with aiosqlite.connect(db_path, timeout=10) as conn:
            cursor = await conn.execute("SELECT SUM(SUM(volume * price)) as total_exposure FROM trades WHERE action IN ('買入', '賣出')")
            result = await cursor.fetchone()
            total_exposure = result[0] or 0.0
            logging.info(f"當前總持倉暴露: {total_exposure}")
            return total_exposure
    except Exception as e:
        logging.error(f"獲取持倉暴露失敗: {e}")
        return 0.0
def calculate_stop_loss(current_price: float, atr: float, action: str, multiplier: float = 2) -> float:
    """計算止損：基於 ATR 動態調整，區分多頭和空頭。"""
    # 函數說明：根據 ATR 和交易方向計算止損價格。
    try:
        if action == "買入":
            stop_loss = current_price - (multiplier * atr)
        elif action == "賣出":
            stop_loss = current_price + (multiplier * atr)
        else:
            stop_loss = current_price
        logging.info(f"計算止損: 當前價格={current_price}, ATR={atr}, 行動={action}, 止損={stop_loss}")
        return stop_loss
    except Exception as e:
        logging.error(f"止損計算錯誤: {e}")
        return current_price
def calculate_take_profit(current_price: float, atr: float, action: str, multiplier: float = 2) -> float:
    """計算止盈：基於 ATR 動態調整，區分多頭和空頭。"""
    # 函數說明：根據 ATR 和交易方向計算止盈價格。
    try:
        if action == "買入":
            take_profit = current_price + (multiplier * atr)
        elif action == "賣出":
            take_profit = current_price - (multiplier * atr)
        else:
            take_profit = current_price
        logging.info(f"計算止盈: 當前價格={current_price}, ATR={atr}, 行動={action}, 止盈={take_profit}")
        return take_profit
    except Exception as e:
        logging.error(f"止盈計算錯誤: {e}")
        return current_price
def check_resources(threshold_mem: float = 0.9, threshold_cpu: float = 80.0) -> bool:
    """檢查系統資源：確保記憶體和 CPU 使用率不過高，記憶體閾值調整為 90%。"""
    # 函數說明：檢查系統資源使用率，若超過閾值則返回 False。
    try:
        mem = virtual_memory()
        cpu = cpu_percent(interval=1)
        if mem.percent > threshold_mem * 100 or cpu > threshold_cpu:
            logging.warning(f"資源使用過高：記憶體 {mem.percent}%，CPU {cpu}%")
            return False
        logging.info(f"資源檢查通過：記憶體 {mem.percent}%，CPU {cpu}%")
        return True
    except Exception as e:
        logging.error(f"資源檢查錯誤: {e}")
        return False
async def calculate_position_size(capital: float, risk_percent: float, stop_loss_distance: float, sentiment: float = 0.0, db_path: str = "C:\\Trading\\data\\trading_data.db") -> float:
    """計算倉位大小：控制風險，檢查總持倉暴露（5% 資本）及槓桿限制（30:1）。"""
    # 函數說明：計算交易倉位大小，考慮風險百分比、情緒調整和暴露限額。
    try:
        if abs(sentiment) > 0.8:
            logging.warning(f"極端情緒分數: {sentiment}，倉位大小設為 0")
            return 0.0
        # 關鍵邏輯：計算基礎倉位並根據情緒調整。
        base_size = (capital * risk_percent) / stop_loss_distance if stop_loss_distance > 0 else 0
        adjustment = 1.2 if sentiment > 0.4 else 0.8 if sentiment < -0.4 else 1.0
        position_size = base_size * adjustment
        # 檢查總持倉暴露
        total_exposure = await get_current_exposure(db_path)
        max_exposure = capital * 0.05 # 最大暴露限額為資本的 5%
        if total_exposure + (position_size * stop_loss_distance) > max_exposure:
            logging.warning(f"超過最大暴露限額: 當前={total_exposure}, 擬新增={position_size * stop_loss_distance}, 限額={max_exposure}")
            return 0.0
        # 檢查槓桿限制
        leverage = (position_size * stop_loss_distance) / capital if capital > 0 else 0
        if leverage > 30:
            logging.warning(f"槓桿超過 30:1: 計算槓桿={leverage:.2f}")
            return 0.0
        logging.info(f"計算倉位大小：基礎={base_size:.2f}，情緒調整={adjustment:.2f}，最終={position_size:.2f}，槓桿={leverage:.2f}")
        return position_size
    except Exception as e:
        logging.error(f"倉位計算錯誤: {e}")
        return 0
def predict_volatility(df: pd.DataFrame, model_path: str = 'models/lightgbm_model_quantized.onnx') -> float:
    """預測波動性：使用 ONNX LightGBM 模型預測 ATR。"""
    # 函數說明：使用 ONNX 模型預測未來波動性，若失敗則回退到平均 ATR。
    try:
        X = df[FEATURES].iloc[-1:].values.astype(np.float32)
        if len(X) == 0:
            logging.error("X 數據為空，回退到平均 ATR")
            return df['ATR'].mean()
        model_dir = Path(model_path).parent
        model_dir.mkdir(exist_ok=True)
        session = ort.InferenceSession(model_path, providers=['VitisAIExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        pred = session.run(None, {'input': X})[0][0]
        logging.info(f"LightGBM 波動性預測: {pred}")
        return pred
    except Exception as e:
        logging.error(f"波動預測錯誤: {e}")
        return df['ATR'].mean()