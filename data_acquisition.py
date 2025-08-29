import yfinance as yf
import pandas as pd
import torch
import pandas_ta as ta
import requests
import json
import time
import logging
import redis
import aiohttp
import aiosqlite
import investpy
from utils import initialize_db, save_data, get_proxy, test_proxy, fetch_api_data, get_redis_client, filter_future_dates
from datetime import datetime, timedelta
import asyncio
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import traceback
import numpy as np
# 速率限制器類
class RateLimiter:
    def __init__(self, calls: int, period: float):
        """初始化速率限制器：calls 次/period 秒"""
        self.calls = calls
        self.period = period
        self.timestamps = []

    async def wait(self):
        """等待直到允許下一次請求"""
        current_time = time.time()
        # 移除超出時間窗口的時間戳
        self.timestamps = [t for t in self.timestamps if current_time - t < self.period]
        if len(self.timestamps) >= self.calls:
            # 等待直到最早的時間戳超出窗口
            sleep_time = self.period - (current_time - self.timestamps[0])
            if sleep_time > 0:
                logging.info(f"Rate limiter: Waiting {sleep_time:.2f} seconds for next request")
                print(f"Rate limiter: Waiting {sleep_time:.2f} seconds for next request")
                await asyncio.sleep(sleep_time)
        self.timestamps.append(time.time())

# 全局 Polygon API 速率限制器：每 60 秒 5 次
polygon_rate_limiter = RateLimiter(calls=5, period=60.0)

async def fetch_sentiment_data(date: str, db_path: str, config: dict) -> pd.DataFrame:
    """從 X API 獲取推文並儲存到 SQLite 的 tweets 表。"""
    try:
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=1)
        cache_key = f"tweets_{end_date.strftime('%Y-%m-%d')}"
        redis_client = get_redis_client(config)
        
        # 檢查 Redis 快取
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    logging.info(f"從 Redis 快取載入推文數據: {cache_key}")
                    return pd.read_json(cached)
            except redis.RedisError as e:
                logging.warning(f"Redis 快取查詢失敗: {str(e)}")
        
        # X API 請求
        start_time = start_date.strftime('%Y-%m-%dT00:00:00Z')
        end_time = end_date.strftime('%Y-%m-%dT23:59:59Z')
        X_BEARER_TOKEN = config['api_key'].get('x_bearer_token', '')
        if not X_BEARER_TOKEN:
            logging.error("X Bearer Token 未配置")
            return pd.DataFrame()
        
        url = "https://api.x.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        params = {'query': "(USDJPY OR USD/JPY OR 'Federal Reserve') lang:en", 'start_time': start_time, 'end_time': end_time, 'max_results': 100}
        data = await fetch_api_data(url, headers=headers, params=params, proxy=get_proxy(config))
        
        if 'data' not in data or not data['data']:
            logging.warning(f"X API 數據為空或格式錯誤: {data}")
            return pd.DataFrame()
        
        # 構建推文數據
        tweets = [
            {'date': end_date, 'tweet_id': tweet['id'], 'text': tweet['text']}
            for tweet in data['data']
        ]
        tweets_df = pd.DataFrame(tweets)
        # 儲存結果
        if not tweets_df.empty:
            if redis_client:
                try:
                    redis_client.setex(cache_key, 24 * 3600, tweets_df.to_json())
                    logging.info(f"已快取推文數據至 Redis: {cache_key}")
                except redis.RedisError as e:
                    logging.warning(f"無法快取至 Redis: {str(e)}")
            await save_data(tweets_df, timeframe='1 day', db_path=db_path, data_type='tweets')
            logging.info(f"儲存 {len(tweets_df)} 條推文到 SQLite")
        else:
            logging.warning("無推文數據")
        
        sentiment_df = pd.DataFrame({'date': [end_date], 'sentiment': [avg_score]})
        await save_data(sentiment_df, timeframe='1 day', db_path=db_path, data_type='sentiment')
        return sentiment_df
    except Exception as e:
        logging.error(f"情緒分析錯誤: {e}")
        return pd.DataFrame()

async def fetch_data(primary_api: str = 'polygon', backup_apis: list = ['yfinance', 'fcs', 'fixer'], date_range: dict = None, timeframe: str = '1d', db_path: str = "C:\\Trading\\data\\trading_data.db", config: dict = None) -> pd.DataFrame:
    """獲取資料：支援多 API，優先使用 primary，失敗則備用。使用 Redis 和 SQLite 快取減少呼叫，加入缺失值填補。"""
    def normalize_timeframe(tf: str) -> str:
        mapping = {'1 hour': '1h', '4 hours': '4h', '1 day': '1d'}
        return mapping.get(tf, tf)
    timeframe = normalize_timeframe(timeframe)
    use_redis = config.get('system_config', {}).get('use_redis', True)
    
    redis_client = get_redis_client(config)
    key = f'usd_jpy_data_{timeframe}'
    start_date = pd.to_datetime(date_range['start'] if date_range else "2025-01-01")
    end_date = pd.to_datetime(date_range['end'] if date_range else "2025-08-25")
    CSV_PATH = Path(config['system_config']['root_dir']) / 'data' / f'usd_jpy_{timeframe}.csv'
    # 定義時間框架映射和各 API 的參數
    interval_map = {
        '1min': {'multiplier': 1, 'timespan': 'minute', 'yfinance': '1m', 'fcs': '1min', 'fixer': None},
        '5min': {'multiplier': 5, 'timespan': 'minute', 'yfinance': '5m', 'fcs': '5min', 'fixer': None},
        '1h': {'multiplier': 1, 'timespan': 'hour', 'yfinance': '1h', 'fcs': '1hour', 'fixer': None},
        '4h': {'multiplier': 4, 'timespan': 'hour', 'yfinance': '4h', 'fcs': '4hour', 'fixer': None},
        '1d': {'multiplier': 1, 'timespan': 'day', 'yfinance': '1d', 'fcs': '1day', 'fixer': '1d'}
    }
    if timeframe not in interval_map:
        logging.error(f"無效的時間框架: {timeframe}")
        return pd.DataFrame()
    interval = interval_map[timeframe]
    # 第一層快取：Redis
    if use_redis and redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 Redis 快取載入資料")
                logging.info(f"Loaded data from Redis cache: key={key}")
                df = pd.read_json(cached)
                df = fill_missing_values(df)
                return df
        except redis.RedisError as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Redis 快取查詢失敗：{str(e)}，跳過 Redis")
            logging.warning(f"Redis cache query failed: {str(e)}，falling back to SQLite", extra={'mode': 'fetch_data'})
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
                # 防護：如果缺少 'n' 或 'vw'，手動設定
                if 'n' not in df.columns:
                    df['n'] = 0
                if 'vw' not in df.columns:
                    df['vw'] = df['close']
                if use_redis and redis_client:
                    try:
                        redis_client.setex(key, 3600, df.to_json())
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已將 SQLite 數據快取至 Redis")
                        logging.info(f"Cached SQLite data to Redis: key={key}")
                    except redis.RedisError as e:
                        logging.warning(f"無法快取至 Redis: {str(e)}")
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 快取載入資料：{timeframe}")
                logging.info(f"Loaded data from SQLite: shape={df.shape}, timeframe={timeframe}")
                return df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']]
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SQLite 快取查詢失敗：{str(e)}")
        logging.error(f"SQLite cache query failed: {str(e)}, traceback={traceback.format_exc()}")
    # API 呼叫：分批獲取
    apis = [primary_api] + backup_apis
    df_list = []
    for api in apis:
        # 檢查 API 是否支持指定的 timeframe
        #if api == 'fixer' and interval['fixer'] is None:
        #    logging.warning(f"Fixer API 不支持 {timeframe} 時間框架，跳過")
        #    continue
        proxies = get_proxy(config)
        if not await test_proxy(proxies):
            logging.warning(f"代理不可用，無代理模式")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 代理不可用，無代理模式")
        current_start = start_date
        while current_start < end_date:
            # 根據 timespan 動態設置分批範圍
            if interval['timespan'] == 'minute':
                batch_end = current_start + timedelta(days=7) # 分鐘級數據分批較小
            elif interval['timespan'] == 'hour':
                batch_end = current_start + timedelta(days=30) # 小時級數據
            else: # day
                batch_end = current_start + timedelta(days=365) # 天級數據
            batch_end = min(batch_end, end_date)
            batch_range = {'start': current_start.strftime('%Y-%m-%d'), 'end': batch_end.strftime('%Y-%m-%d')}
            # 檢查 yfinance 的時間範圍限制
            if api == 'yfinance':
                delta = batch_end - current_start
                if timeframe == '1min' and delta > timedelta(days=7):
                    logging.warning(f"yfinance 不支持 {timeframe} 超過 7 天，跳過此批次")
                    current_start = batch_end
                    continue
                if timeframe == '5min' and delta > timedelta(days=60):
                    logging.warning(f"yfinance 不支持 {timeframe} 超過 60 天，跳過此批次")
                    current_start = batch_end
                    continue
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 分批提取 {api}：{batch_range['start']} 至 {batch_range['end']}")
            logging.info(f"Batch fetch from {api}: {batch_range['start']} to {batch_range['end']}")
            try:
                if api == 'yfinance':
                    for attempt in range(5):
                        try:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}")
                            ticker = yf.Ticker('USDJPY=X')
                            df = ticker.history(start=batch_range['start'], end=batch_range['end'], interval=interval['yfinance'])
                            if df.empty:
                                logging.warning(f"Yahoo Finance batch empty: timeframe={timeframe}")
                                break
                            df = df.reset_index().rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                            df['date'] = pd.to_datetime(df['date'])
                            df['n'] = 0 # yfinance 不提供交易次數
                            df['vw'] = df['close'] # 模擬成交量加權平均價格
                            df = fill_missing_values(df)
                            df_list.append(df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']])
                            logging.info(f"yfinance data fetched: shape={df.shape}, timeframe={timeframe}")
                            break
                        except Exception as e:
                            if attempt == 4:
                                logging.error(f"Yahoo Finance batch failed: {str(e)}, traceback={traceback.format_exc()}")
                                break
                            await asyncio.sleep(2 ** attempt * 4)
                elif api == 'polygon':
                    # 根據 polygon 及 timespan 限制動態設置分批範圍
                    if interval['timespan'] == 'minute':
                        batch_end = current_start + timedelta(days=30) # 分鐘級數據分批較小
                    elif interval['timespan'] == 'hour':
                        batch_end = current_start + timedelta(days=730) # 小時級數據
                    else: # day
                        batch_end = current_start + timedelta(days=730) # 天級數據
                    batch_end = min(batch_end, end_date)
                    batch_range = {'start': current_start.strftime('%Y-%m-%d'), 'end': batch_end.strftime('%Y-%m-%d')}
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}")
                    api_key = config['api_key'].get('polygon_api_key', '')
                    if not api_key:
                        logging.error("Polygon API key not configured")
                        continue
                    # 計算預期數據點數以設置 limit
                    delta = batch_end - current_start
                    if interval['timespan'] == 'minute':
                        expected_points = delta.total_seconds() // (interval['multiplier'] * 60) # 分鐘級數據分批較小
                    elif interval['timespan'] == 'hour':
                        expected_points = delta.total_seconds() // (interval['multiplier'] * 3600)# 小時級數據
                        print(f"{delta.total_seconds()} /{(interval['multiplier'] * 3600)}=expected_points")
                    else: # day
                        expected_points = delta.days // interval['multiplier']
                    limit = min(int(expected_points) + 100, 50000) # 設置 limit，留有餘量
                    url = f"https://api.polygon.io/v2/aggs/ticker/C:USDJPY/range/{interval['multiplier']}/{interval['timespan']}/{batch_range['start']}/{batch_range['end']}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}: {url}")
                    logging.info(f"Attempting Polygon API request: {url}")
                    for attempt in range(3):  # 重試 3 次
                        await polygon_rate_limiter.wait()  # 等待速率限制器
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, proxy=proxies.get('http'), timeout=10) as response:
                                data = await response.json()
                                if data.get('status') != 'OK' and data.get('status') != "DELAYED":
                                    logging.warning(f"Polygon API failed: status={data.get('status')}, message={data.get('error', 'Unknown error')}")
                                    continue
                                if data.get('ticker') != 'C:USDJPY':
                                    logging.warning(f"Polygon API returned unexpected ticker: {data.get('ticker')}")
                                    continue
                                if 'results' not in data or not data['results']:
                                    logging.warning(f"Polygon API batch empty: timeframe={timeframe}")
                                    continue
                                logging.info(f"Polygon API metadata: queryCount={data.get('queryCount', 0)}, resultsCount={data.get('resultsCount', 0)}, request_id={data.get('request_id', 'N/A')}")
                                df = pd.DataFrame(data['results'])[['t', 'o', 'h', 'l', 'c', 'v', 'n', 'vw']].rename(
                                    columns={'t': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'n', 'vw': 'vw'}
                                )
                                df['date'] = pd.to_datetime(df['date'], unit='ms')
                                df = fill_missing_values(df)
                                df_list.append(df)
                                logging.info(f"Polygon data fetched: shape={df.shape}, timeframe={timeframe}")
                elif api == 'fcs':
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 提取{api}")
                    api_key = config['api_key'].get('fcs_api_key', '')
                    if not api_key:
                        logging.error("FCS API key not configured")
                        continue
                    url = f"https://fcsapi.com/api-v3/forex/history?symbol=USD/JPY&access_key={api_key}&period={interval['fcs']}&from={batch_range['start']}&to={batch_range['end']}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, proxy=proxies.get('http'), timeout=10) as response:
                            data = await response.json()
                            if 'response' not in data or not data['response']:
                                logging.warning(f"FCS API batch empty: timeframe={timeframe}")
                                continue
                            if data.get('code') != 200:
                                logging.warning(f"FCS API failed: code={data.get('code')}, message={data.get('msg', 'Unknown error')}")
                                continue
                            df = pd.DataFrame(data['response'])[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'datetime': 'date'})
                            df['date'] = pd.to_datetime(df['date'])
                            df['n'] = 0 # FCS 不提供交易次數
                            df['vw'] = df['close'] # 模擬成交量加權平均價格
                            df = fill_missing_values(df)
                            df_list.append(df)
                            logging.info(f"FCS data fetched: shape={df.shape}, timeframe={timeframe}")
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
                                    df_data.append({'date': date, 'close': usd_jpy, 'open': usd_jpy, 'high': usd_jpy, 'low': usd_jpy, 'volume': 0, 'n': 0, 'vw': usd_jpy})
                            df = pd.DataFrame(df_data)
                            df['date'] = pd.to_datetime(df['date'])
                            df = fill_missing_values(df)
                            df_list.append(df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']])
                            logging.info(f"Fixer data fetched: shape={df.shape}, timeframe={timeframe}")
                current_start = batch_end
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
            if use_redis and redis_client:
                try:
                    redis_client.setex(key, 3600, df.to_json())
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已將 API 數據快取至 Redis")
                    logging.info(f"Cached API data to Redis: key={key}")
                except redis.RedisError as e:
                    logging.warning(f"無法快取至 Redis: {str(e)}")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 {api} 獲取資料成功")
            logging.info(f"Successfully fetched data from {api}: shape={df.shape}, timeframe={timeframe}")
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']]
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 所有 API 和 CSV 後備失敗")
    logging.error("All APIs and CSV fallback failed")
    return pd.DataFrame()
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """填補缺失值：使用前向填補法，確保數據連續性。"""
    if df.empty:
        return df
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'n', 'vw', 'fed_funds_rate']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已填補缺失值")
    logging.info("Missing values filled using forward and backward fill")
    return df
async def compute_indicators(df: pd.DataFrame, db_path: str, timeframe: str, config: dict = None) -> pd.DataFrame:
    """計算技術指標：使用多線程計算 RSI, MACD, ATR, Stochastic, ADX, Ichimoku, Bollinger, EMA，支援分批計算並檢查重複。"""
    try:
        # 動態數據長度檢查，根據時間框架設置最小數據長度
        min_data_length = {'1min': 200, '5min': 150, '1h': 100, '4h': 60, '1d': 60}
        required_length = min_data_length.get(timeframe, 100)
        if len(df) < required_length:
            logging.warning(f"數據長度不足 ({len(df)} < {required_length}) for timeframe {timeframe}，跳過指標計算")
            return df

        # 檢查 SQLite 是否已存儲最新指標
        async with aiosqlite.connect(db_path, timeout=10) as conn:
            cursor = await conn.execute(
                "SELECT MAX(date) FROM indicators WHERE timeframe = ? AND indicator IN (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (timeframe, 'RSI', 'MACD', 'ATR', 'Stoch_k', 'ADX', 'Ichimoku_tenkan', 'Ichimoku_kijun', 'BB_upper', 'EMA_12', 'EMA_26')
            )
            last_date = await cursor.fetchone()
            last_date = pd.to_datetime(last_date[0]) if last_date[0] else None
            if last_date and df['date'].max() <= last_date:
                logging.info(f"發現最新指標已存儲 for timeframe {timeframe}，從 SQLite 載入")
                cursor = await conn.execute(
                    "SELECT date, indicator, value FROM indicators WHERE timeframe = ? AND date >= ?",
                    (timeframe, df['date'].min().strftime('%Y-%m-%d'))
                )
                rows = await cursor.fetchall()
                if rows:
                    indicators_df = pd.DataFrame(rows, columns=['date', 'indicator', 'value'])
                    indicators_df = indicators_df.pivot(index='date', columns='indicator', values='value')
                    indicators_df['date'] = pd.to_datetime(indicators_df['date'])
                    df = df.merge(indicators_df, on='date', how='left')
                    df = fill_missing_values(df)
                    return df

        # 分批計算（若數據量大）
        batch_size = 10000 if timeframe in ['1min', '5min'] else None
        if batch_size and len(df) > batch_size:
            logging.info(f"數據量大 ({len(df)})，分批計算，batch_size={batch_size}")
            df_list = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
        else:
            df_list = [df]

        result_dfs = []
        for batch_df in df_list:
            def calc_rsi(df):
                rsi = ta.rsi(df['close'], length=14)
                return rsi if rsi is not None else pd.Series(np.nan, index=df.index, name='RSI')

            def calc_macd(df):
                macd = ta.macd(df['close'])
                if macd is None:
                    return pd.DataFrame({'MACD': np.nan, 'MACD_signal': np.nan, 'MACD_hist': np.nan}, index=df.index)
                return macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']].rename(
                    columns={'MACD_12_26_9': 'MACD', 'MACDs_12_26_9': 'MACD_signal', 'MACDh_12_26_9': 'MACD_hist'}
                )

            def calc_atr(df):
                atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                return atr if atr is not None else pd.Series(np.nan, index=df.index, name='ATR')

            def calc_stoch(df):
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
                if stoch is None:
                    return pd.Series(np.nan, index=df.index, name='Stoch_k')
                return stoch['STOCHk_14_3_3']

            def calc_adx(df):
                adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                if adx is None:
                    return pd.Series(np.nan, index=df.index, name='ADX')
                return adx['ADX_14']

            def calc_ichimoku(df):
                if not all(col in df.columns for col in ['high', 'low', 'close']):
                    logging.warning(f"Missing required columns for Ichimoku calculation: {df.columns.tolist()}")
                    return pd.DataFrame({
                        'Ichimoku_tenkan': np.nan, 'Ichimoku_kijun': np.nan, 'Ichimoku_span_a': np.nan, 'Ichimoku_span_b': np.nan
                    }, index=df.index)
                if df[['high', 'low', 'close']].isna().any().any():
                    logging.warning(f"NaN values detected in high, low, or close columns for timeframe {timeframe}")
                    df = df.fillna(method='ffill').fillna(method='bfill')
                ich = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
                if ich[0] is None:
                    logging.warning(f"Ichimoku calculation returned None for timeframe {timeframe}")
                    return pd.DataFrame({
                        'Ichimoku_tenkan': np.nan, 'Ichimoku_kijun': np.nan, 'Ichimoku_span_a': np.nan, 'Ichimoku_span_b': np.nan
                    }, index=df.index)
                try:
                    # Replace with actual column names from debug output
                    return ich[0][['TENKAN_9', 'KIJUN_26', 'SENKOU_A_9', 'SENKOU_B_26']].rename(
                        columns={
                            'TENKAN_9': 'Ichimoku_tenkan',
                            'KIJUN_26': 'Ichimoku_kijun',
                            'SENKOU_A_9': 'Ichimoku_span_a',
                            'SENKOU_B_26': 'Ichimoku_span_b'
                        }
                    )
                except KeyError as e:
                    logging.error(f"Ichimoku column error: {str(e)}")
                    return pd.DataFrame({
                        'Ichimoku_tenkan': np.nan, 'Ichimoku_kijun': np.nan, 'Ichimoku_span_a': np.nan, 'Ichimoku_span_b': np.nan
                    }, index=df.index)

            def calc_bbands(df):
                bb = ta.bbands(df['close'], length=20, std=2)
                if bb is None:
                    return pd.DataFrame({'BB_upper': np.nan, 'BB_middle': np.nan, 'BB_lower': np.nan}, index=df.index)
                return bb[['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']].rename(
                    columns={'BBU_20_2.0': 'BB_upper', 'BBM_20_2.0': 'BB_middle', 'BBL_20_2.0': 'BB_lower'}
                )

            def calc_ema_12(df):
                ema = ta.ema(df['close'], length=12)
                return ema if ema is not None else pd.Series(np.nan, index=df.index, name='EMA_12')

            def calc_ema_26(df):
                ema = ta.ema(df['close'], length=26)
                return ema if ema is not None else pd.Series(np.nan, index=df.index, name='EMA_26')

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(calc_rsi, batch_df),
                    executor.submit(calc_macd, batch_df),
                    executor.submit(calc_atr, batch_df),
                    executor.submit(calc_stoch, batch_df),
                    executor.submit(calc_adx, batch_df),
                    executor.submit(calc_ichimoku, batch_df),
                    executor.submit(calc_bbands, batch_df),
                    executor.submit(calc_ema_12, batch_df),
                    executor.submit(calc_ema_26, batch_df)
                ]
                results = [f.result() for f in futures]

            batch_df = batch_df.copy()
            batch_df['RSI'] = results[0]
            batch_df[['MACD', 'MACD_signal', 'MACD_hist']] = results[1]
            batch_df['ATR'] = results[2]
            batch_df['Stoch_k'] = results[3]
            batch_df['ADX'] = results[4]
            batch_df[['Ichimoku_tenkan', 'Ichimoku_kijun', 'Ichimoku_span_a', 'Ichimoku_span_b']] = results[5]
            batch_df['Ichimoku_cloud_top'] = results[5][['Ichimoku_span_a', 'Ichimoku_span_b']].max(axis=1)
            batch_df[['BB_upper', 'BB_middle', 'BB_lower']] = results[6]
            batch_df['EMA_12'] = results[7]
            batch_df['EMA_26'] = results[8]

            # 改進缺失值處理：前向填補代替 dropna
            batch_df = fill_missing_values(batch_df)
            result_dfs.append(batch_df)

        df = pd.concat(result_dfs, ignore_index=True).drop_duplicates(subset=['date'])

        # 存入 DB
        await save_data(df, timeframe, db_path, data_type='indicators')
        logging.info(f"指標數據已存入 DB: timeframe={timeframe}")

        # 存入 CSV
        if config:
            indicators_csv_path = Path(config['system_config']['root_dir']) / 'data' / f'usd_jpy_{timeframe}_indicators.csv'
            if not os.path.exists(indicators_csv_path.parent):
                os.makedirs(indicators_csv_path.parent)
            df.to_csv(indicators_csv_path, index=False)
            logging.info(f"指標數據已存入 CSV: {indicators_csv_path}")

        return df
    except Exception as e:
        logging.error(f"指標計算錯誤: {e}")
        return df
async def fetch_economic_calendar(date_range: dict, db_path: str, config: dict) -> pd.DataFrame:
    """獲取經濟日曆數據並儲存到 SQLite，新增 FRED API 獲取聯邦基金利率。"""
    use_redis = config.get('system_config', {}).get('use_redis', True)
   
    redis_client = get_redis_client(config)
    key = 'usd_jpy_economic_calendar'
    CSV_PATH = Path(config['system_config']['root_dir']) / 'data' / 'economic_calendar.csv'
    start_date = pd.to_datetime(date_range['start']) - timedelta(days=7)
    end_date = pd.to_datetime(date_range['end']) + timedelta(days=7)
    # 第一層快取：Redis
    if use_redis and redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 Redis 快取載入經濟日曆數據")
                logging.info(f"Loaded economic calendar from Redis cache: key={key}")
                df = pd.read_json(cached)
                df['date'] = pd.to_datetime(df['date'])
                df = fill_missing_values(df)
                return df[['date', 'event', 'impact', 'fed_funds_rate']]
        except redis.RedisError as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Redis 快取查詢失敗：{str(e)}，跳過 Redis")
            logging.warning(f"Redis cache query failed: {str(e)}，falling back to SQLite", extra={'mode': 'fetch_economic_calendar'})
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
                if use_redis and redis_client:
                    try:
                        redis_client.setex(key, 3600, df.to_json())
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已將 SQLite 數據快取至 Redis")
                        logging.info(f"Cached SQLite data to Redis: key={key}")
                    except redis.RedisError as e:
                        logging.warning(f"無法快取至 Redis: {str(e)}")
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
    # 儲存數據前去重，防止UNIQUE constraint failed
    if not df.empty:
        df = df.drop_duplicates(subset=['date', 'event'], keep='last')  # 去重，保留最後一筆
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
        if use_redis and redis_client:
            try:
                redis_client.setex(key, 3600, df.to_json())
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已將經濟日曆數據快取至 Redis")
                logging.info(f"Cached economic calendar data to Redis: key={key}")
            except redis.RedisError as e:
                logging.warning(f"無法快取至 Redis: {str(e)}")
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
            df = df.drop_duplicates(subset=['date', 'event'], keep='last')  # 去重
            df = filter_future_dates(df)
            df = fill_missing_values(df)
            await save_data(df, timeframe='1 day', db_path=db_path, data_type='economic')
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 CSV 載入經濟日曆數據並儲存到 SQLite")
            logging.info(f"Loaded and saved economic calendar from CSV: shape={df.shape}")
            if use_redis and redis_client:
                try:
                    redis_client.setex(key, 3600, df.to_json())
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已將 CSV 數據快取至 Redis")
                    logging.info(f"Cached CSV data to Redis: key={key}")
                except redis.RedisError as e:
                    logging.warning(f"無法快取至 Redis: {str(e)}")
            return df[['date', 'event', 'impact', 'fed_funds_rate']]
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CSV 檔案缺少必要欄位：{required_columns}")
            logging.warning(f"Invalid or missing columns in {CSV_PATH}, columns={df.columns.tolist()}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 警告：經濟日曆數據為空")
    logging.warning("Economic calendar is empty")
    return pd.DataFrame()