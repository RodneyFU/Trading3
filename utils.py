import torch
import onnxruntime as ort
import logging
import os
import aiosqlite
import pandas as pd
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# 加密 API 密鑰（補充安全性）
key = b'your_encryption_key_here'  # 安全儲存
cipher = Fernet(key)
def encrypt_key(api_key: str) -> bytes:
    """加密 API 密鑰。
    邏輯：使用 Fernet 加密。
    """
    return cipher.encrypt(api_key.encode())

def decrypt_key(encrypted: bytes) -> str:
    """解密 API 密鑰。
    """
    return cipher.decrypt(encrypted).decode()

def check_hardware():
    """硬體檢測：檢查 GPU/NPU。
    邏輯：DirectML for GPU，VitisAI for NPU，回退 CPU。
    """
    try:
        import torch_directml
        device = torch_directml.device() if torch_directml.is_available() else torch.device('cpu')
        logging.info(f"使用裝置: {device}")
    except ImportError:
        device = torch.device('cpu')
        logging.info("回退到 CPU")
    
    providers = ort.get_available_providers()
    if 'VitisAIExecutionProvider' in providers:
        logging.info("NPU 支援")
        session = ort.InferenceSession('models/lstm_model.onnx', providers=['VitisAIExecutionProvider'])
    else:
        session = ort.InferenceSession('models/lstm_model.onnx', providers=['CPUExecutionProvider'])
    return device, session

def setup_proxy():
    """設置代理：從環境變數載入。
    邏輯：若存在，設置 http/https proxy。
    """
    load_dotenv()
    proxy = os.getenv('HTTP_PROXY')
    if proxy:
        os.environ['http_proxy'] = proxy
        os.environ['https_proxy'] = proxy
        logging.info(f"代理設置: {proxy}")
        print("代理設置成功")
    else:
        logging.info("無代理")
        print("無代理設置")

async def save_data(df: pd.DataFrame, db_path: str = "data/trades.db"):
    """儲存資料到資料庫：使用 aiosqlite。
    邏輯：創建表，若存在則附加。
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    price REAL,
                    action TEXT,
                    volume REAL,
                    stop_loss REAL,
                    take_profit REAL
                )
            """)
            await db.commit()
            await df.to_sql("trades", db, if_exists="append", index=False)
        print("資料儲存成功")
    except Exception as e:
        logging.error(f"資料庫錯誤: {e}")
        print("資料儲存失敗")

def check_volatility(atr: float, threshold: float = 0.02) -> bool:
    """檢查波動：若 ATR > 閾值，暫停。
    邏輯：高波動避免入場。
    """
    if atr > threshold:
        logging.warning("高波動偵測")
        return False
    return True