import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import onnx
import onnxruntime as ort
from onnxmltools import convert_xgboost, convert_sklearn, convert_lightgbm
from onnxconverter_common import FloatTensorType, convert_float_to_float16
from transformers import pipeline, TimeSeriesTransformerModel, TimeSeriesTransformerConfig, DistilBertTokenizer, DistilBertForSequenceClassification
import logging
from scipy.stats import ks_2samp
import os
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from pathlib import Path
import joblib
import redis
from sklearn.metrics import mean_squared_error, r2_score
from utils import save_data, check_hardware, get_redis_client
from datetime import timedelta, datetime
import asyncio
import aiohttp
from textblob import TextBlob
import aiosqlite

# 全局特徵常量，避免重複定義
FEATURES = ['close', 'RSI', 'MACD', 'Stoch_k', 'ADX', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26', 'fed_funds_rate']

class LSTMModel(nn.Module):
    """LSTM 模型：用於價格預測，捕捉時間序列模式。"""
    def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_size=1, device=torch.device('cpu')):
        """初始化 LSTM 模型，支援多個技術指標和經濟數據的輸入大小。"""
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)
 
    def forward(self, x):
        # 關鍵邏輯：將輸入數據移動到指定設備，並通過 LSTM 和全連接層進行前向傳播。
        x = x.to(self.device)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])
def validate_data(df: pd.DataFrame, required_features: list) -> bool:
    """驗證輸入數據是否包含必要的特徵欄位。"""
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        logging.error(f"缺少必要特徵: {missing_features}")
        return False
    if df[required_features].isna().any().any():
        logging.error("數據包含 NaN 值")
        return False
    return True
def train_lstm_model(df: pd.DataFrame, epochs: int = 50, device=torch.device('cpu')):
    """訓練 LSTM：使用歷史數據訓練，保存為 ONNX 格式。"""
    # 函數說明：此函數負責訓練 LSTM 模型，使用指定的特徵進行價格預測，並將模型轉換為 ONNX 格式以便跨平台使用。
    try:
        if not validate_data(df, FEATURES):
            return None
        # 關鍵邏輯：移除 NaN 值並準備輸入 X 和目標 y（下一期的 Close 值）。
        X = df[FEATURES].dropna().values[:-1]
        y = df['close'].shift(-1).dropna().values
        if len(X) != len(y):
            logging.error("X 和 y 長度不匹配")
            return None
        if len(X) == 0:
            logging.error("X 數據為空，無法訓練 LSTM")
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LSTMModel(input_size=len(FEATURES), device=device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        # 關鍵邏輯：訓練循環，使用反向傳播更新模型參數。
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(torch.tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1))
            loss = criterion(output.squeeze(), torch.tensor(y_train, dtype=torch.float32, device=device))
            loss.backward()
            optimizer.step()
     
        # 保存 PyTorch 模型
        torch.save(model.state_dict(), 'models/lstm_model.pth')
        # 轉換為 ONNX
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32).unsqueeze(1)
        torch.onnx.export(model, dummy_input, "models/lstm_model.onnx", input_names=["input"], output_names=["output"], opset_version=11)
        # 量化為 FP16
        model_fp32 = onnx.load("models/lstm_model.onnx")
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, "models/lstm_model_quantized.onnx")
        logging.info("LSTM 模型訓練完成並轉換為 ONNX")
        return model
    except Exception as e:
        logging.error(f"LSTM 訓練錯誤: {e}")
        return None
def train_xgboost_model(df: pd.DataFrame):
    """訓練 XGBoost 模型：用於短期價格預測，保存為 ONNX 格式。"""
    # 函數說明：訓練 XGBoost 模型，用於短期價格預測，並計算性能指標後轉換為 ONNX 格式。
    try:
        if not validate_data(df, FEATURES):
            return None
        X = df[FEATURES].dropna().values[:-1]
        y = df['close'].shift(-1).dropna().values
        if len(X) != len(y):
            logging.error("X 和 y 長度不匹配")
            return None
        if len(X) == 0:
            logging.error("X 數據為空，無法訓練 XGBoost")
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        # 關鍵邏輯：計算模型性能指標，如 RMSE 和 R² 分數。
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"XGBoost 價格預測性能：RMSE={rmse:.4f}, R²={r2:.4f}")
        joblib.dump(model, 'models/xgboost_model.pkl')
        # 轉換為 ONNX
        onnx_model = convert_xgboost(model, initial_types=[('input', FloatTensorType([None, len(FEATURES)]))])
        onnx.save(onnx_model, "models/xgboost_model.onnx")
        # 量化為 FP16
        model_fp32 = onnx.load("models/xgboost_model.onnx")
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, "models/xgboost_model_quantized.onnx")
        logging.info("XGBoost 模型訓練完成並轉換為 ONNX")
        return model
    except Exception as e:
        logging.error(f"XGBoost 訓練錯誤: {e}")
        return None
def train_random_forest_model(df: pd.DataFrame):
    """訓練隨機森林模型：用於短期價格預測，保存為 ONNX 格式。"""
    # 函數說明：訓練 RandomForest 模型，用於中期價格預測，並轉換為 ONNX 格式。
    try:
        if not validate_data(df, FEATURES):
            return None
        X = df[FEATURES].dropna().values[:-1]
        y = df['close'].shift(-1).dropna().values
        if len(X) != len(y):
            logging.error("X 和 y 長度不匹配")
            return None
        if len(X) == 0:
            logging.error("X 數據為空，無法訓練 RandomForest")
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'models/rf_model.pkl')
        # 轉換為 ONNX
        onnx_model = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, len(FEATURES)]))])
        onnx.save(onnx_model, "models/rf_model.onnx")
        # 量化為 FP16
        model_fp32 = onnx.load("models/rf_model.onnx")
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, "models/rf_model_quantized.onnx")
        logging.info("RandomForest 模型訓練完成並轉換為 ONNX")
        return model
    except Exception as e:
        logging.error(f"RandomForest 訓練錯誤: {e}")
        return None
def train_lightgbm_model(df: pd.DataFrame):
    """訓練 LightGBM 模型：用於波動性預測（ATR），保存為 ONNX 格式。"""
    # 函數說明：訓練 LightGBM 模型，用於波動性預測，並與 XGBoost 比較性能後轉換為 ONNX 格式。
    try:
        if not validate_data(df, FEATURES):
            return None
        X = df[FEATURES].dropna().values[:-1]
        y = df['ATR'].shift(-1).dropna().values
        if len(X) != len(y):
            logging.error("X 和 y 長度不匹配")
            return None
        if len(X) == 0:
            logging.error("X 數據為空，無法訓練 LightGBM")
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        # 關鍵邏輯：計算 LightGBM 性能指標，並訓練 XGBoost 進行比較。
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"LightGBM 波動性預測性能：RMSE={rmse:.4f}, R²={r2:.4f}")
        # 比較 XGBoost 的波動性預測性能
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)
        xgb_r2 = r2_score(y_test, xgb_pred)
        logging.info(f"XGBoost 波動性預測性能：RMSE={xgb_rmse:.4f}, R²={xgb_r2:.4f}")
        logging.info(f"性能比較：LightGBM RMSE={rmse:.4f} vs XGBoost RMSE={xgb_rmse:.4f}, LightGBM R²={r2:.4f} vs XGBoost R²={xgb_r2:.4f}")
        joblib.dump(model, 'models/lightgbm_model.pkl')
        # 轉換為 ONNX
        onnx_model = convert_lightgbm(model, initial_types=[('input', FloatTensorType([None, len(FEATURES)]))])
        onnx.save(onnx_model, "models/lightgbm_model.onnx")
        # 量化為 FP16
        model_fp32 = onnx.load("models/lightgbm_model.onnx")
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, "models/lightgbm_model_quantized.onnx")
        logging.info("LightGBM 模型訓練完成並轉換為 ONNX")
        return model
    except Exception as e:
        logging.error(f"LightGBM 訓練錯誤: {e}")
        return None
def train_timeseries_transformer(df: pd.DataFrame, epochs: int = 10, device=torch.device('cpu')):
    """訓練 TimeSeriesTransformer 模型：用於時間序列預測，保存為 ONNX 格式。"""
    # 函數說明：訓練 TimeSeriesTransformer 模型，用於時間序列預測，處理序列數據並轉換為 ONNX 格式。
    try:
        seq_len = 60
        # 關鍵邏輯：移除 NaN 值並構造時間序列輸入 X 和目標 y。
        df_clean = df.dropna(subset=FEATURES + ['close'])
        num_seq = len(df_clean) - seq_len
        if num_seq <= 0:
            logging.error("數據不足以構造時間序列，無法訓練 TimeSeriesTransformer")
            return None
        X = np.array([df_clean[FEATURES].iloc[i:i+seq_len].values for i in range(num_seq)])
        y = df_clean['close'].iloc[seq_len:].values
        if len(X) != len(y):
            logging.error("X 和 y 長度不匹配")
            return None
        config = TimeSeriesTransformerConfig(
            input_size=len(FEATURES), time_series_length=seq_len, prediction_length=1, d_model=64
        )
        model = TimeSeriesTransformerModel(config).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 關鍵邏輯：訓練循環，使用反向傳播更新模型參數。
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(torch.tensor(X_train, dtype=torch.float32, device=device)).logits
            loss = criterion(output.squeeze(), torch.tensor(y_train, dtype=torch.float32, device=device))
            loss.backward()
            optimizer.step()
        # 保存 PyTorch 模型
        torch.save(model.state_dict(), 'models/timeseries_transformer.pth')
        # 轉換為 ONNX
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32, device=device)
        torch.onnx.export(model, dummy_input, "models/timeseries_transformer.onnx", input_names=["input"], output_names=["output"], opset_version=11)
        # 量化為 FP16
        model_fp32 = onnx.load("models/timeseries_transformer.onnx")
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, "models/timeseries_transformer_quantized.onnx")
        logging.info("TimeSeriesTransformer 模型訓練完成並轉換為 ONNX")
        return model
    except Exception as e:
        logging.error(f"TimeSeriesTransformer 訓練錯誤: {e}")
        return None
def train_distilbert(df: pd.DataFrame, device=torch.device('cpu')):
    """訓練 DistilBERT 模型：用於情緒分析，保存為 ONNX 格式。"""
    # 函數說明：訓練 DistilBERT 模型，用於情緒分析，處理文本數據並轉換為 ONNX 格式。
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
        texts = df['tweets'].dropna().tolist()[:100]
        labels = df['sentiment'].dropna().tolist()[:100]
        if not texts or len(texts) != len(labels):
            logging.error("無效的文本或標籤數據")
            return None, None
        # 關鍵邏輯：對文本進行 tokenization 並準備輸入。
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        optimizer = Adam(model.parameters(), lr=5e-5)
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(**inputs).logits
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
        # 保存 PyTorch 模型
        torch.save(model.state_dict(), 'models/distilbert_model.pth')
        # 轉換為 ONNX
        dummy_input = {
            'input_ids': tokenizer(['dummy text'], return_tensors="pt", padding=True, truncation=True)['input_ids'].to(device),
            'attention_mask': tokenizer(['dummy text'], return_tensors="pt", padding=True, truncation=True)['attention_mask'].to(device)
        }
        torch.onnx.export(model, (dummy_input['input_ids'], dummy_input['attention_mask']),
                          "models/distilbert_model.onnx", input_names=["input_ids", "attention_mask"], output_names=["output"], opset_version=11)
        # 量化為 FP16
        model_fp32 = onnx.load("models/distilbert_model.onnx")
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, "models/distilbert_model_quantized.onnx")
        logging.info("DistilBERT 模型訓練完成並轉換為 ONNX")
        return model, tokenizer
    except Exception as e:
        logging.error(f"DistilBERT 訓練錯誤: {e}")
        return None, None
async def predict_sentiment(date: str, db_path: str, config: dict) -> float:
    """情緒分析：從 X API 獲取推文，計算 polarity 並儲存結果，使用 DistilBERT，加入 Redis 快取。"""
    # 函數說明：進行情緒分析，從 X (Twitter) API 獲取推文，使用 DistilBERT 和 TextBlob 計算情緒分數，並快取結果。
    use_redis = config.get('system_config', {}).get('use_redis', True)
   
    redis_client = get_redis_client(config)
    try:
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=1)
        cache_key = f"sentiment_{end_date.strftime('%Y-%m-%d')}"
        # 關鍵邏輯：檢查 Redis 快取，若存在則直接返回。
        # 檢查 Redis 快取
        if use_redis and redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    logging.info(f"從 Redis 快取載入情緒分數: {cache_key}")
                    return float(cached)
            except redis.RedisError as e:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Redis 快取查詢失敗：{str(e)}，跳過 Redis")
                logging.warning(f"Redis cache query failed: {str(e)}，falling back to API", extra={'mode': 'predict_sentiment'})
        start_time = start_date.strftime('%Y-%m-%dT00:00:00Z')
        end_time = end_date.strftime('%Y-%m-%dT23:59:59Z')
        X_BEARER_TOKEN = config['api_key'].get('x_bearer_token', '')
        if not X_BEARER_TOKEN:
            logging.error("X Bearer Token 未配置")
            return 0.0
        query = "(USDJPY OR USD/JPY OR 'Federal Reserve') lang:en"
        logging.info(f"從 X API 獲取推文：query={query}, start={start_time}, end={end_time}")
        url = "https://api.x.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        params = {'query': query, 'start_time': start_time, 'end_time': end_time, 'max_results': 100}
        polarities = []
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.load_state_dict(torch.load('models/distilbert_model.pth'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        async with aiohttp.ClientSession() as session:
            for attempt in range(5):
                try:
                    async with session.get(url, headers=headers, params=params, timeout=10) as response:
                        data = await response.json()
                        if response.status != 200 or 'data' not in data or not data['data']:
                            logging.warning(f"X API 數據為空或格式錯誤: {data}")
                            return 0.0
                        tweets = data['data']
                        # 關鍵邏輯：對每條推文計算情緒分數，結合 DistilBERT 和 TextBlob。
                        for tweet in tweets:
                            text = tweet['text']
                            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                            with torch.no_grad():
                                outputs = model(**inputs).logits
                            score = torch.softmax(outputs, dim=1)[0][1].item() - torch.softmax(outputs, dim=1)[0][0].item()
                            tb_polarity = TextBlob(text).sentiment.polarity
                            combined_score = 0.7 * score + 0.3 * tb_polarity
                            polarities.append(combined_score)
                        break
                except Exception as e:
                    if attempt == 4:
                        logging.error(f"X API 獲取失敗: {str(e)}")
                        return 0.0
                    await asyncio.sleep(2 ** attempt * 4)
        if polarities:
            avg_score = sum(polarities) / len(polarities)
            logging.info(f"計算平均 polarity 分數: {avg_score} (DistilBERT 70%, TextBlob 30%)")
            # 存入 Redis 快取，設置 24 小時過期
            if use_redis and redis_client:
                try:
                    redis_client.setex(cache_key, 24 * 3600, str(avg_score))
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已將情緒分數快取至 Redis")
                    logging.info(f"Cached sentiment score to Redis: key={cache_key}")
                except redis.RedisError as e:
                    logging.warning(f"無法快取至 Redis: {str(e)}")
        else:
            avg_score = 0.0
            logging.warning("無推文數據，使用預設值 0.0")
        sentiment_df = pd.DataFrame({'date': [end_date], 'sentiment': [avg_score]})
        await save_data(sentiment_df, timeframe='1 day', db_path=db_path, data_type='sentiment')
        return avg_score
    except Exception as e:
        logging.error(f"情緒分析錯誤: {e}")
        return 0.0
def integrate_sentiment(polarity: float) -> float:
    """整合情緒分數：將 polarity 轉換為決策調整值，並檢查極端情緒。"""
    # 函數說明：根據情緒分數調整決策值，若極端則暫停交易。
    if abs(polarity) > 0.8:
        logging.warning(f"極端情緒分數: {polarity}，建議暫停交易")
        return 0.0
    if polarity > 0.1:
        return 0.1
    elif polarity < -0.1:
        return -0.1
    return 0.0
def detect_drift(old_data: pd.DataFrame, new_data: pd.DataFrame, threshold: float = 0.05) -> bool:
    """檢測數據漂移：使用 KS 檢驗比較分佈。"""
    # 函數說明：使用 Kolmogorov-Smirnov 檢驗檢測數據分佈漂移。
    stat, p_value = ks_2samp(old_data['close'], new_data['close'])
    return p_value < threshold
def predict(model_path: str, input_data: pd.DataFrame, provider='VitisAIExecutionProvider'):
    """使用 ONNX 模型進行推理，支援 NPU。"""
    # 函數說明：載入 ONNX 模型並進行預測，支持多種執行提供者。
    try:
        session = ort.InferenceSession(model_path, providers=[provider, 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        X = input_data[FEATURES].iloc[-1:].values.astype(np.float32)
        return session.run(None, {'input': X})[0][0]
    except Exception as e:
        logging.error(f"ONNX 推理錯誤: {e}")
        return 0.0
def update_model(df: pd.DataFrame, model_path: str = 'models', session: str = 'normal', device_config: dict = None) -> dict:
    """更新多模型：個別檢查並訓練或載入模型，根據時段加權預測。"""
    # 函數說明：更新多個模型，若數據漂移則重新訓練，並定義加權預測函數。
    try:
        model_dir = Path(model_path)
        model_dir.mkdir(exist_ok=True)
        models = {}
        # 關鍵邏輯：檢測數據漂移，若存在則重新訓練模型。
        old_data = df.iloc[:-1000] if len(df) > 1000 else df
        new_data = df.iloc[-1000:]
        data_drift = detect_drift(old_data, new_data)
        device = device_config.get('lstm', torch.device('cpu')) if device_config else torch.device('cpu')
        df_clean = df[FEATURES].dropna()
        X = df_clean.values[:-1]
        y = df['close'].shift(-1).dropna().values
        # LSTM 模型
        lstm_path = model_dir / 'lstm_model.pth'
        lstm_onnx_path = model_dir / 'lstm_model_quantized.onnx'
        if not lstm_path.exists() or data_drift:
            models['lstm'] = train_lstm_model(df, device=device)
        else:
            models['lstm'] = lstm_onnx_path
            logging.info("載入現有 LSTM ONNX 模型")
        # XGBoost 模型
        xgboost_path = model_dir / 'xgboost_model.pkl'
        xgboost_onnx_path = model_dir / 'xgboost_model_quantized.onnx'
        if not xgboost_path.exists() or data_drift:
            models['xgboost'] = train_xgboost_model(df)
        else:
            models['xgboost'] = xgboost_onnx_path
            logging.info("載入現有 XGBoost ONNX 模型")
        # RandomForest 模型
        rf_path = model_dir / 'rf_model.pkl'
        rf_onnx_path = model_dir / 'rf_model_quantized.onnx'
        if not rf_path.exists() or data_drift:
            models['rf_model'] = train_random_forest_model(df)
        else:
            models['rf_model'] = rf_onnx_path
            logging.info("載入現有 RandomForest ONNX 模型")
        # LightGBM 模型
        lightgbm_path = model_dir / 'lightgbm_model.pkl'
        lightgbm_onnx_path = model_dir / 'lightgbm_model_quantized.onnx'
        if not lightgbm_path.exists() or data_drift:
            models['lightgbm'] = train_lightgbm_model(df)
        else:
            models['lightgbm'] = lightgbm_onnx_path
            logging.info("載入現有 LightGBM ONNX 模型")
        # TimeSeriesTransformer 模型
        transformer_path = model_dir / 'timeseries_transformer.pth'
        transformer_onnx_path = model_dir / 'timeseries_transformer_quantized.onnx'
        if not transformer_path.exists() or data_drift:
            models['timeseries_transformer'] = train_timeseries_transformer(df, device=device)
        else:
            models['timeseries_transformer'] = transformer_onnx_path
            logging.info("載入現有 TimeSeriesTransformer ONNX 模型")
        # DistilBERT 模型
        distilbert_path = model_dir / 'distilbert_model.pth'
        distilbert_onnx_path = model_dir / 'distilbert_model_quantized.onnx'
        if not distilbert_path.exists() or data_drift:
            models['distilbert'], _ = train_distilbert(df, device=device)
        else:
            models['distilbert'] = distilbert_onnx_path
            logging.info("載入現有 DistilBERT ONNX 模型")
        # 加權預測
        weights = {
            'lstm': 0.2, 'xgboost': 0.3, 'rf_model': 0.2, 'lightgbm': 0.2, 'timeseries_transformer': 0.1
        } if session == 'high_volatility' else {
            'lstm': 0.3, 'xgboost': 0.2, 'rf_model': 0.2, 'lightgbm': 0.2, 'timeseries_transformer': 0.1
        }
        def predict_price(input_data: pd.DataFrame):
            # 內部函數說明：根據輸入數據進行加權預測，忽略情緒模型。
            if input_data.empty or FEATURES[0] not in input_data.columns:
                logging.error("輸入數據為空或缺少必要欄位")
                return 0.0
            predictions = {}
            for name, model in models.items():
                if name == 'distilbert':
                    continue
                if isinstance(model, str):
                    predictions[name] = predict(model, input_data)
                else:
                    X = input_data[FEATURES].iloc[-1:].values
                    if name == 'lstm' or name == 'timeseries_transformer':
                        X_tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
                        model.eval()
                        with torch.no_grad():
                            predictions[name] = model(X_tensor).item()
                    else:
                        predictions[name] = model.predict(X)[0]
            final_pred = sum(weights[name] * pred for name, pred in predictions.items())
            return final_pred
        models['predict'] = predict_price
        logging.info("多模型更新完成")
        return models
    except Exception as e:
        logging.error(f"多模型更新錯誤: {e}")
        return {}