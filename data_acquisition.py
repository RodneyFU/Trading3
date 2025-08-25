import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import json
import time
import logging
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def fetch_data(primary_api: str = 'yfinance', backup_apis: list = ['polygon', 'fcs']) -> pd.DataFrame:
    """獲取資料：支援多 API，優先使用 primary，失敗則備用。使用 Redis 快取減少呼叫。
    邏輯：檢查快取 → API 呼叫 → 預處理缺失值。
    """
    key = 'usd_jpy_data'
    cached = redis_client.get(key)
    if cached:
        print("從快取載入資料")
        return pd.read_json(cached)
    
    apis = [primary_api] + backup_apis
    for api in apis:
        try:
            if api == 'yfinance':
                df = yf.download("USDJPY=X", start="2025-01-01", end="2025-08-25", interval="1d")
            elif api == 'polygon':
                url = f"https://api.polygon.io/v2/aggs/ticker/C:USDJPY/range/1/day/2023-08-23/2025-08-22?adjusted=true&sort=asc&apiKey={api_keys['polygon_api_key']}"
                response = requests.get(url)
                data = response.json()['results']
                df = pd.DataFrame(data)[['t', 'o', 'h', 'l', 'c']].rename(columns={'t':'Date', 'o':'Open', 'h':'High', 'l':'Low', 'c':'Close'})
                df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                df.set_index('Date', inplace=True)
            elif api == 'fcs':
                url = f"https://fcsapi.com/api-v3/forex/history?symbol=USD/JPY&access_key={api_keys['FCS_API_Key']}"
                response = requests.get(url)
                data = response.json()['response']
                df = pd.DataFrame(data)[['datetime', 'open', 'high', 'low', 'close']].rename(columns={'datetime':'Date', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'})
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            if not df.empty:
                redis_client.setex(key, 3600, df.to_json())
                print(f"從 {api} 獲取資料成功")
                return df
            time.sleep(5)
        except Exception as e:
            logging.error(f"{api} API 失敗: {e}")
    print("所有 API 失敗")
    return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標：RSI、MACD、ATR 等，用於模型輸入。
    邏輯：使用 pandas_ta 計算，填補缺失值。
    """
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        # 添加更多指標如 EMA、Bollinger Bands
        df['EMA'] = ta.ema(df['Close'], length=20)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.bbands(df['Close'], length=20, std=2).iloc[:, :3]
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    except Exception as e:
        logging.error(f"計算指標錯誤: {e}")
        return df