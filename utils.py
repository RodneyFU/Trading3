import torch
import onnxruntime as ort
import logging
import os
import aiosqlite
import pandas as pd
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from pathlib import Path
from datetime import datetime
import json
import traceback
import aiofiles
import aiohttp
import random

# 全局快取變數，用於確保配置僅載入一次
_config_cache = None

# 加密 API 密鑰（補充安全性）
key = b'_eIKG0YhiJCyBQ-VvxAsx8LT3Vow-k0hE-i0iwK9wwM='  # 安全儲存
cipher = Fernet(key)

def encrypt_key(api_key: str) -> bytes:
    """加密 API 密鑰。"""
    return cipher.encrypt(api_key.encode())

def decrypt_key(encrypted: bytes) -> str:
    """解密 API 密鑰。"""
    return cipher.decrypt(encrypted).decode()

async def initialize_db(db_path: str):
    """初始化 SQLite 資料庫，創建 OHLC、indicators、economic_calendar、sentiment_data 和 trades 表格。"""
    try:
        async with aiosqlite.connect(db_path, timeout=10) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlc (
                    date DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    timeframe TEXT,
                    PRIMARY KEY (date, timeframe)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS indicators (
                    date DATETIME,
                    indicator TEXT,
                    value REAL,
                    timeframe TEXT,
                    PRIMARY KEY (date, indicator, timeframe)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS economic_calendar (
                    date DATETIME,
                    event TEXT,
                    impact TEXT,
                    PRIMARY KEY (date, event)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    date DATETIME,
                    sentiment REAL,
                    PRIMARY KEY (date)
                )
            """)
            await conn.execute("""
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
            await conn.commit()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫初始化成功：{db_path}")
            await backup_database(db_path, root_dir)
            logging.info(f"Database initialized: {db_path}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫初始化失敗：{str(e)}")
        logging.error(f"Database initialization failed: {str(e)}, traceback={traceback.format_exc()}")

async def save_data(df: pd.DataFrame, timeframe: str, db_path: str, data_type: str = 'ohlc') -> bool:
    """將數據增量儲存到 SQLite 資料庫。"""
    try:
        async with aiosqlite.connect(db_path, timeout=10) as conn:
            if data_type == 'ohlc':
                ohlc_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                df_to_save = df[ohlc_columns].copy()
                df_to_save['timeframe'] = timeframe
                cursor = await conn.execute("SELECT MAX(date) FROM ohlc WHERE timeframe = ?", (timeframe,))
                last_date = (await cursor.fetchone())[0]
                if last_date:
                    df_to_save = df_to_save[df_to_save['date'] > pd.to_datetime(last_date)]
                if not df_to_save.empty:
                    await df_to_save.to_sql('ohlc', conn, if_exists='append', index=False)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 增量儲存 {len(df_to_save)} 行 OHLC 數據至 SQLite：{timeframe}")
                    logging.info(f"Incrementally saved {len(df_to_save)} OHLC rows to SQLite: timeframe={timeframe}")
            elif data_type == 'indicators':
                ohlc_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                indicator_columns = [col for col in df.columns if col not in ohlc_columns + ['event', 'impact', 'sentiment']]
                if indicator_columns:
                    indicators_df = df[['date'] + indicator_columns].melt(id_vars=['date'], var_name='indicator', value_name='value')
                    indicators_df['timeframe'] = timeframe
                    cursor = await conn.execute("SELECT MAX(date) FROM indicators WHERE timeframe = ?", (timeframe,))
                    last_date = (await cursor.fetchone())[0]
                    if last_date:
                        indicators_df = indicators_df[indicators_df['date'] > pd.to_datetime(last_date)]
                    if not indicators_df.empty:
                        await indicators_df.to_sql('indicators', conn, if_exists='append', index=False)
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 增量儲存 {len(indicators_df)} 行技術指標至 SQLite：{timeframe}")
                        logging.info(f"Incrementally saved {len(indicators_df)} indicator rows to SQLite: timeframe={timeframe}")
            elif data_type == 'economic':
                economic_columns = ['date', 'event', 'impact']
                df_to_save = df[economic_columns].copy()
                cursor = await conn.execute("SELECT MAX(date) FROM economic_calendar")
                last_date = (await cursor.fetchone())[0]
                if last_date:
                    df_to_save = df_to_save[df_to_save['date'] > pd.to_datetime(last_date)]
                if not df_to_save.empty:
                    await df_to_save.to_sql('economic_calendar', conn, if_exists='append', index=False)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 增量儲存 {len(df_to_save)} 行經濟日曆數據至 SQLite")
                    logging.info(f"Incrementally saved {len(df_to_save)} economic calendar rows to SQLite")
            elif data_type == 'sentiment':
                sentiment_columns = ['date', 'sentiment']
                df_to_save = df[sentiment_columns].copy()
                cursor = await conn.execute("SELECT MAX(date) FROM sentiment_data")
                last_date = (await cursor.fetchone())[0]
                if last_date:
                    df_to_save = df_to_save[df_to_save['date'] > pd.to_datetime(last_date)]
                if not df_to_save.empty:
                    await df_to_save.to_sql('sentiment_data', conn, if_exists='append', index=False)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 增量儲存 {len(df_to_save)} 行情緒數據至 SQLite")
                    logging.info(f"Incrementally saved {len(df_to_save)} sentiment rows to SQLite")
            await conn.commit()
        return True
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {data_type} 數據儲存失敗：{str(e)}")
        logging.error(f"Failed to save {data_type} data to SQLite: {str(e)}, traceback={traceback.format_exc()}")
        return False

async def backup_database(db_path: str, root_dir: str):
    """備份 SQLite 資料庫。"""
    backup_dir = Path(root_dir) / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"trading_data_{datetime.now().strftime('%Y%m%d')}.db"
    try:
        async with aiofiles.open(db_path, mode='rb') as src, aiofiles.open(backup_file, mode='wb') as dst:
            content = await src.read()
            await dst.write(content)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫已備份至 {backup_file}")
        logging.info(f"Database backed up to {backup_file}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫備份失敗：{str(e)}")
        logging.error(f"Database backup failed: {str(e)}, traceback={traceback.format_exc()}")

async def save_periodically(df_buffer: pd.DataFrame, timeframe: str, db_path: str, root_dir: str, data_type: str = 'ohlc'):
    """定期將緩衝區數據保存到 SQLite 並進行每日備份。"""
    save_interval = 1800 if timeframe == '1 hour' else 3 * 3600
    while True:
        try:
            if not df_buffer.empty:
                await save_data(df_buffer, timeframe, db_path, data_type)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {data_type} 數據已增量保存至 SQLite")
                logging.info(f"Data incrementally saved to SQLite: timeframe={timeframe}, data_type={data_type}")
            if datetime.now().hour == 0 and datetime.now().minute < 5:
                await backup_database(db_path, root_dir)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫已備份")
                logging.info("Database backed up")
            await asyncio.sleep(save_interval)
        except Exception as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 定期保存失敗：{str(e)}")
            logging.error(f"Periodic save failed: {str(e)}, traceback={traceback.format_exc()}")

def load_settings():
    """載入所有設定檔案並生成 requirements.txt。"""
    global _config_cache
    if _config_cache is not None:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從快取載入配置")
        logging.info("Loaded config from cache")
        return _config_cache
    config = {}
    root_dir = "C:\\Trading"
    config_dir = Path(root_dir) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    default_api_key = {}
    default_trading_params = {
        "max_position_size": 10000,
        "risk_per_trade": 0.01,
        "price_diff_threshold": {"high_volatility": 0.005, "normal": 0.003},
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "stoch_overbought": 80,
        "stoch_oversold": 20,
        "adx_threshold": 25,
        "obv_window": 14,
        "capital": 10000,
        "risk_percent": 0.01,
        "atr_threshold": 0.02,
        "min_backtest_days": 180,
        "ppo_learning_rate": 0.0003,
        "ppo_timesteps": 10000
    }
    default_system_config = {
        "data_source": "yfinance",
        "symbol": "USDJPY=X",
        "timeframe": "1d",
        "root_dir": "C:\\Trading",
        "db_path": "C:\\Trading\\data\\trading_data.db",
        "proxies": {
            "http": "http://proxy1.scig.gov.hk:8080",
            "https": "http://proxy1.scig.gov.hk:8080"
        },
        "dependencies": [],
        "model_dir": "models",
        "model_periods": ["short_term", "medium_term", "long_term"]
    }
    config_files = {
        'api_key': config_dir / 'api_key.json',
        'trading_params': config_dir / 'trading_params.json',
        'system_config': config_dir / 'system_config.json'
    }
    try:
        for key, file_path in config_files.items():
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config[key] = json.load(f)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功載入設定檔：{file_path}")
                logging.info(f"Successfully loaded config file: {file_path}")
            else:
                if key == 'api_key':
                    default = default_api_key
                elif key == 'trading_params':
                    default = default_trading_params
                elif key == 'system_config':
                    default = default_system_config
                config[key] = default
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default, f, indent=4, ensure_ascii=False)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定檔不存在，已創建預設：{file_path}")
                logging.info(f"Created default config file: {file_path}")
        if 'api_key' in config:
            for k, v in config['api_key'].items():
                config['api_key'][k] = encrypt_key(v)
        system_config = config.get('system_config', {})
        dependencies = system_config.get('dependencies', [])
        if not dependencies:
            dependencies = [
                "torch-directml", "onnxruntime", "transformers", "stable-baselines3", "pandas-ta",
                "lightgbm", "aiosqlite", "requests", "gym", "pytest", "python-dotenv", "redis",
                "streamlit", "prometheus-client", "ib_insync", "cryptography", "scipy", "yfinance",
                "pandas", "numpy", "torch", "sklearn", "aiofiles", "aiohttp", "beautifulsoup4", "textblob",
                "joblib", "psutil", "onnxmltools", "onnxconverter-common"
            ]
            system_config['dependencies'] = dependencies
            with open(config_files['system_config'], 'w', encoding='utf-8') as f:
                json.dump(system_config, f, indent=4, ensure_ascii=False)
            logging.info("Filled default dependencies in system_config.json")
        requirements_path = Path(root_dir) / 'requirements.txt'
        with open(requirements_path, 'w', encoding='utf-8') as f:
            for dep in dependencies:
                f.write(f"{dep}\n")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已生成/更新 requirements.txt：{requirements_path}")
        logging.info(f"Generated requirements.txt: {requirements_path}")
        _config_cache = config
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定檔載入成功！")
        logging.info("Config files loaded successfully")
        return config
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定檔載入失敗：{str(e)}")
        logging.error(f"Failed to load config files: {str(e)}")
        return {}

def check_hardware():
    """硬體檢測：檢測 GPU/NPU/CPU 並為不同模型指定設備。"""
    try:
        import torch_directml
        gpu_available = torch_directml.is_available()
        device = torch_directml.device() if gpu_available else torch.device('cpu')
        logging.info(f"使用裝置: {device}")
    except ImportError:
        device = torch.device('cpu')
        logging.info("回退到 CPU")
    providers = ort.get_available_providers()
    onnx_provider = 'VitisAIExecutionProvider' if 'VitisAIExecutionProvider' in providers else 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in providers else 'CPUExecutionProvider'
    logging.info(f"ONNX provider: {onnx_provider}")
    device_config = {
        'lstm': device,
        'finbert': device,
        'xgboost': torch.device('cpu'),
        'randomforest': torch.device('cpu'),
        'lightgbm': torch.device('cpu'),
        'timeseries_transformer': device,
        'distilbert': device,
        'ppo': device
    }
    try:
        session = ort.InferenceSession('models/lstm_model_quantized.onnx', providers=[onnx_provider])
    except Exception as e:
        logging.warning(f"Failed to load ONNX session: {e}, using CPU")
        session = None
    return device_config, session

def get_proxy(config: dict) -> dict:
    """獲取代理設置，支援多個備用代理。"""
    proxies = config.get('system_config', {}).get('proxies', {})
    if not proxies:
        logging.info("無代理設置")
        return {}
    return proxies

async def test_proxy(proxy: dict) -> bool:
    """測試代理是否可用。"""
    if not proxy:
        return True
    test_url = "https://www.google.com"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url, proxy=proxy.get('http'), timeout=5) as response:
                if response.status == 200:
                    logging.info(f"代理測試成功: {proxy}")
                    return True
                else:
                    logging.warning(f"代理測試失敗: {proxy}, 狀態碼={response.status}")
                    return False
    except Exception as e:
        logging.error(f"代理測試失敗: {proxy}, 錯誤={str(e)}")
        return False

def setup_proxy():
    """設置代理：從環境變數載入。"""
    load_dotenv()
    proxy = os.getenv('HTTP_PROXY')
    if proxy:
        os.environ['http_proxy'] = proxy
        os.environ['https_proxy'] = proxy
        logging.info(f"代理設置: {proxy}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 代理設置成功")
    else:
        logging.info("無代理")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無代理設置")

def check_volatility(atr: float, threshold: float = 0.02) -> bool:
    """檢查波動：若 ATR > 閾值，暫停。"""
    if atr > threshold:
        logging.warning("高波動偵測")
        return False
    return True

def clear_config_cache():
    """清除配置快取。"""
    global _config_cache
    _config_cache = None
    logging.info("Config cache cleared")

def filter_future_dates(df: pd.DataFrame) -> pd.DataFrame:
    """過濾未來日期數據。"""
    if not df.empty and 'date' in df.columns:
        current_time = pd.to_datetime(datetime.now())
        initial_rows = len(df)
        df = df[df['date'] <= current_time].copy()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已過濾未來日期，初始行數={initial_rows}，剩餘行數={len(df)}")
        logging.info(f"Filtered future dates, initial_rows={initial_rows}, remaining_rows={len(df)}")
    return df