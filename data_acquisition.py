import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import json
import time
import logging
import redis
import aiohttp
import investpy
from utils import initialize_db, save_data, get_proxy, test_proxy, filter_future_dates
from datetime import datetime, timedelta
import asyncio
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import traceback

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def fetch_data(primary_api: str = 'polygon', backup_apis: list = ['yfinance', 'fcs', 'fixer'], date_range: dict = None, timeframe: str = '1d', db_path: str = "C:\\Trading\\data\\trading_data.db", config: dict = None) -> pd.DataFrame:
    """獲取資料：支援多 API，優先使用 primary，失敗則備用。使用 Redis 和 SQLite 快取減少呼叫，加入缺失值填補。"""
    key = f'usd_jpy_data_{timeframe}'
    start_date = pd.to_datetime(date_range['start'] if date_range else "2025-01-01")
    end_date = pd.to_datetime(date_range['end'] if date_range else "2025-08-25")
    CSV_PATH = Path(config['system_config']['root_dir']) / 'data' / f'usd_jpy_{timeframe}.csv'
    interval_map = {'1 hour': '1h', '4 hours': '4h', '1 day': '1d', '1min': '1m', '5min': '5m'}
    if timeframe not in interval_map:
        logging.error(f"無效的時間框架: {timeframe}")
        return pd.DataFrame()
    interval = interval_map[timeframe]
    # 第一層快取：Redis
    cached = redis_client.get(key)
    if cached:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 Redis 快取載入資料")
        logging.info(f"Loaded data from Redis cache: key={key}")
        df = pd.read_json(cached)
        df = fill_missing_values(df)
        return df
    # 第二層快取：SQLite
    await initialize_db(db_path)
    try:
        async with aiosqlite.connect(db_path, timeout=10) as conn:
            cursor = await conn.execute("SELECT * FROM ohlc WHERE timeframe = ? AND date BETWEEN ? AND ?",
                                     (timeframe, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            await cursor.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = filter_future_dates(df)
                df = fill_missing_values(df)
                redis_client.setex(key, 3600, df.to_json())
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 快取載入資料：{timeframe}")
                logging.info(f"Loaded data from SQLite: shape={df.shape}, timeframe={timeframe}")
                return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SQLite 快取查詢失敗：{str(e)}")
        logging.error(f"SQLite cache query failed: {str(e)}, traceback={traceback.format_exc()}")
    # API 呼叫：分批獲取
    apis = [primary_api] + backup_apis
    df_list = []
    for api in apis:
        proxies = get_proxy(config)
        if not await test_proxy(proxies):
            logging.warning(f"代理不可用，無代理模式")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 代理不可用，無代理模式")
        current_start = start_date
        while current_start < end_date:
            if timeframe == '1 hour':
                batch_end = current_start + timedelta(days=30)
            elif timeframe == '4 hours':
                batch_end = current_start + timedelta(days=60)
            else:
                batch_end = current_start + timedelta(days=730)
            batch_end = min(batch_end, end_date)
            batch_range = {'start': current_start.strftime('%Y-%m-%d'), 'end': batch_end.strftime('%Y-%m-%d')}
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 分批提取 {api}：{batch_range['start']} 至 {batch_range['end']}")
            logging.info(f"Batch fetch from {api}: {batch_range['start']} to {batch_range['end']}")
            try:
                if api == 'yfinance':
                    for attempt in range(5):
                        try:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}")
                            ticker = yf.Ticker('USDJPY=X')
                            df = ticker.history(start=batch_range['start'], end=batch_range['end'], interval=interval)
                            if df.empty:
                                logging.warning(f"Yahoo Finance batch empty: timeframe={timeframe}")
                                break
                            df = df.reset_index().rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                            df['date'] = pd.to_datetime(df['date'])
                            df = fill_missing_values(df)
                            df_list.append(df[['date', 'open', 'high', 'low', 'close', 'volume']])
                            break
                        except Exception as e:
                            if attempt == 4:
                                logging.error(f"Yahoo Finance batch failed: {str(e)}, traceback={traceback.format_exc()}")
                                break
                            await asyncio.sleep(2 ** attempt * 4)
                elif api == 'polygon':
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}")
                    api_key = config['api_key'].get('polygon_api_key', '')
                    if not api_key:
                        logging.error("Polygon API key not configured")
                        continue
                    url = f"https://api.polygon.io/v2/aggs/ticker/C:USDJPY/range/1/day/{batch_range['start']}/{batch_range['end']}?adjusted=true&sort=asc&apiKey={api_key}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, proxy=proxies.get('http'), timeout=10) as response:
                            data = await response.json()
                            if 'results' not in data:
                                logging.warning(f"Polygon API batch empty: timeframe={timeframe}")
                                continue
                            df = pd.DataFrame(data['results'])[['t', 'o', 'h', 'l', 'c', 'v']].rename(columns={'t': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                            df['date'] = pd.to_datetime(df['date'], unit='ms')
                            df = fill_missing_values(df)
                            df_list.append(df)
                elif api == 'fcs':
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}")
                    api_key = config['api_key'].get('fcs_api_key', '')
                    if not api_key:
                        logging.error("FCS API key not configured")
                        continue
                    url = f"https://fcsapi.com/api-v3/forex/history?symbol=USD/JPY&access_key={api_key}&period={interval}&from={batch_range['start']}&to={batch_range['end']}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, proxy=proxies.get('http'), timeout=10) as response:
                            data = await response.json()
                            if 'response' not in data:
                                logging.warning(f"FCS API batch empty: timeframe={timeframe}")
                                continue
                            df = pd.DataFrame(data['response'])[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'datetime': 'date'})
                            df['date'] = pd.to_datetime(df['date'])
                            df = fill_missing_values(df)
                            df_list.append(df)
                elif api == 'fixer':
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}")
                    api_key = config['api_key'].get('fixer_API_Key', '')
                    if not api_key:
                        logging.error("Fixer API key not configured")
                        continue
                    url = f"http://data.fixer.io/api/timeseries?access_key={api_key}&start_date={batch_range['start']}&end_date={batch_range['end']}&symbols=USD,JPY"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, proxy=proxies.get('http'), timeout=10) as response:
                            data = await response.json()
                            if not data.get('success') or 'rates' not in data:
                                logging.warning(f"Fixer API batch empty or failed: timeframe={timeframe}")
                                continue
                            rates = data['rates']
                            df_data = []
                            for date, rate in rates.items():
                                if 'USD' in rate and 'JPY' in rate:
                                    usd_jpy = rate['JPY'] / rate['USD']
                                    df_data.append({'date': date, 'close': usd_jpy, 'open': usd_jpy, 'high': usd_jpy, 'low': usd_jpy, 'volume': 0})
                            df = pd.DataFrame(df_data)
                            df['date'] = pd.to_datetime(df['date'])
                            df = fill_missing_values(df)
                            df_list.append(df[['date', 'open', 'high', 'low', 'close', 'volume']])
                current_start = batch_end
                await asyncio.sleep(12)
            except Exception as e:
                logging.error(f"{api} API batch failed: {str(e)}, traceback={traceback.format_exc()}")
                continue
        if df_list:
            df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['date'])
            df = filter_future_dates(df)
            df = fill_missing_values(df)
            await save_data(df, timeframe, db_path, data_type='ohlc')
            if not os.path.exists(CSV_PATH.parent):
                os.makedirs(CSV_PATH.parent)
            df.to_csv(CSV_PATH, index=False)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} OHLC 數據已儲存到 CSV：{CSV_PATH}")
            redis_client.setex(key, 3600, df.to_json())
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 {api} 獲取資料成功")
            logging.info(f"Successfully fetched data from {api}: shape={df.shape}, timeframe={timeframe}")
            return df
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 所有 API 和 CSV 後備失敗")
    logging.error("All APIs and CSV fallback failed")
    return pd.DataFrame()

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """填補缺失值：使用前向填補法，確保數據連續性。"""
    if df.empty:
        return df
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'fed_funds_rate']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已填補缺失值")
    logging.info("Missing values filled using forward and backward fill")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標：使用多線程計算 RSI, MACD, ATR, Stochastic, ADX, Ichimoku, Bollinger, EMA。"""
    try:
        def calc_rsi(df): return ta.rsi(df['close'], length=14)
        def calc_macd(df): return ta.macd(df['close'])[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']].rename(columns={'MACD_12_26_9': 'MACD', 'MACDs_12_26_9': 'MACD_signal', 'MACDh_12_26_9': 'MACD_hist'})
        def calc_atr(df): return ta.atr(df['high'], df['low'], df['close'], length=14)
        def calc_stoch(df): return ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)['STOCHk_14_3_3']
        def calc_adx(df): return ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        def calc_ichimoku(df): return ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)[0][['ISA_9', 'ISB_26']]
        def calc_bbands(df): return ta.bbands(df['close'], length=20, std=2)[['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']]
        def calc_ema_12(df): return ta.ema(df['close'], length=12)
        def calc_ema_26(df): return ta.ema(df['close'], length=26)
       
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(calc_rsi, df),
                executor.submit(calc_macd, df),
                executor.submit(calc_atr, df),
                executor.submit(calc_stoch, df),
                executor.submit(calc_adx, df),
                executor.submit(calc_ichimoku, df),
                executor.submit(calc_bbands, df),
                executor.submit(calc_ema_12, df),
                executor.submit(calc_ema_26, df)
            ]
            results = [f.result() for f in futures]
       
        df['RSI'] = results[0]
        df[['MACD', 'MACD_signal', 'MACD_hist']] = results[1]
        df['ATR'] = results[2]
        df['Stoch_k'] = results[3]
        df['ADX'] = results[4]
        df['Ichimoku_span_a'] = results[5]['ISA_9']
        df['Ichimoku_span_b'] = results[5]['ISB_26']
        df['Ichimoku_cloud_top'] = results[5][['ISA_9', 'ISB_26']].max(axis=1)
        df['BB_upper'] = results[6]['BBU_20_2.0']
        df['BB_middle'] = results[6]['BBM_20_2.0']
        df['BB_lower'] = results[6]['BBL_20_2.0']
        df['EMA_12'] = results[7]
        df['EMA_26'] = results[8]
        return df.dropna()
    except Exception as e:
        logging.error(f"指標計算錯誤: {e}")
        return df

async def fetch_economic_calendar(date_range: dict, db_path: str, config: dict) -> pd.DataFrame:
    """獲取經濟日曆數據並儲存到 SQLite，新增 FRED API 獲取聯邦基金利率。"""
    key = 'usd_jpy_economic_calendar'
    CSV_PATH = Path(config['system_config']['root_dir']) / 'data' / 'economic_calendar.csv'
    start_date = pd.to_datetime(date_range['start']) - timedelta(days=7)
    end_date = pd.to_datetime(date_range['end']) + timedelta(days=7)
    # 第一層快取：Redis
    cached = redis_client.get(key)
    if cached:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 Redis 快取載入經濟日曆數據")
        logging.info(f"Loaded economic calendar from Redis cache: key={key}")
        df = pd.read_json(cached)
        df['date'] = pd.to_datetime(df['date'])
        df = fill_missing_values(df)
        return df[['date', 'event', 'impact', 'fed_funds_rate']]
    # 第二層快取：SQLite
    try:
        async with aiosqlite.connect(db_path, timeout=10) as conn:
            cursor = await conn.execute("SELECT * FROM economic_calendar WHERE date BETWEEN ? AND ?",
                                     (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            await cursor.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = filter_future_dates(df)
                df = fill_missing_values(df)
                redis_client.setex(key, 3600, df.to_json())
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 載入經濟日曆數據")
                logging.info(f"Loaded economic calendar from SQLite: shape={df.shape}")
                return df[['date', 'event', 'impact', 'fed_funds_rate']]
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SQLite 經濟日曆查詢失敗：{str(e)}")
        logging.error(f"SQLite economic calendar query failed: {str(e)}, traceback={traceback.format_exc()}")
    # 使用 investpy 獲取經濟日曆數據
    try:
        proxies = get_proxy(config)
        if not await test_proxy(proxies):
            logging.warning("主代理不可用，嘗試備用代理")
            for backup_proxy in config.get('system_config', {}).get('backup_proxies', []):
                if await test_proxy(backup_proxy):
                    proxies = backup_proxy
                    break
            else:
                proxies = {}
                logging.warning("所有代理不可用，無代理模式")
        if proxies:
            os.environ['http_proxy'] = proxies.get('http', '')
            os.environ['https_proxy'] = proxies.get('https', '')
        importances = ['high', 'medium']
        time_zone = 'GMT +8:00'
        from_date = start_date.strftime('%d/%m/%Y')
        to_date = end_date.strftime('%d/%m/%Y')
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 investpy 獲取經濟日曆數據：{from_date} 至 {to_date}")
        logging.info(f"Fetching economic calendar from investpy: start={from_date}, end={to_date}")
        calendar = investpy.economic_calendar(
            importances=importances,
            time_zone=time_zone,
            from_date=from_date,
            to_date=to_date,
            countries=['united states', 'japan']
        )
        if calendar.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} investpy 經濟日曆數據為空")
            logging.warning("investpy economic calendar data is empty")
            df = pd.DataFrame()
        else:
            calendar['date'] = pd.to_datetime(calendar['date'], format='%d/%m/%Y')
            calendar = calendar[calendar['importance'].notnull()]
            calendar['event'] = calendar['currency'] + ' ' + calendar['event']
            calendar['impact'] = calendar['importance'].str.capitalize()
            df = calendar[['date', 'event', 'impact']]
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} investpy 經濟日曆數據獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch investpy economic calendar: {str(e)}, traceback={traceback.format_exc()}")
        df = pd.DataFrame()
    # 使用 FRED API 獲取聯邦基金利率
    try:
        fred_api_key = config['api_key'].get('fred_api_key', '')
        if not fred_api_key:
            logging.error("FRED API key not configured")
        else:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={fred_api_key}&file_type=json&observation_start={start_date.strftime('%Y-%m-%d')}&observation_end={end_date.strftime('%Y-%m-%d')}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, proxy=proxies.get('http'), timeout=10) as response:
                    data = await response.json()
                    if 'observations' not in data:
                        logging.warning("FRED API returned no observations")
                    else:
                        fred_data = pd.DataFrame(data['observations'])[['date', 'value']].rename(columns={'value': 'fed_funds_rate'})
                        fred_data['date'] = pd.to_datetime(fred_data['date'])
                        fred_data['fed_funds_rate'] = fred_data['fed_funds_rate'].astype(float)
                        if not df.empty:
                            df = df.merge(fred_data[['date', 'fed_funds_rate']], on='date', how='left')
                        else:
                            df = fred_data[['date', 'fed_funds_rate']]
                            df['event'] = 'FEDFUNDS'
                            df['impact'] = 'High'
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FRED API 獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch FRED API data: {str(e)}, traceback={traceback.format_exc()}")
    # 儲存數據
    if not df.empty:
        df = filter_future_dates(df)
        df = fill_missing_values(df)
        await save_data(df, timeframe='1 day', db_path=db_path, data_type='economic')
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 經濟日曆數據已儲存到 SQLite")
        logging.info(f"Economic calendar data saved to SQLite: shape={df.shape}")
        if not os.path.exists(CSV_PATH.parent):
            os.makedirs(CSV_PATH.parent)
        df.to_csv(CSV_PATH, index=False)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 經濟日曆數據已儲存到 CSV：{CSV_PATH}")
        logging.info(f"Economic calendar data saved to CSV: {CSV_PATH}")
        redis_client.setex(key, 3600, df.to_json())
        return df[['date', 'event', 'impact', 'fed_funds_rate']]
    # 從 CSV 載入備份
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無 investpy/FRED 數據，嘗試從 CSV 載入")
    logging.info("No investpy/FRED data, trying to load from CSV")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        required_columns = ['date', 'event', 'impact']
        if all(col in df.columns for col in required_columns):
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = filter_future_dates(df)
            df = fill_missing_values(df)
            await save_data(df, timeframe='1 day', db_path=db_path, data_type='economic')
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 CSV 載入經濟日曆數據並儲存到 SQLite")
            logging.info(f"Loaded and saved economic calendar from CSV: shape={df.shape}")
            redis_client.setex(key, 3600, df.to_json())
            return df[['date', 'event', 'impact', 'fed_funds_rate']]
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CSV 檔案缺少必要欄位：{required_columns}")
            logging.warning(f"Invalid or missing columns in {CSV_PATH}, columns={df.columns.tolist()}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 警告：經濟日曆數據為空")
    logging.warning("Economic calendar is empty")
    return pd.DataFrame()