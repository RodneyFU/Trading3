# USD/JPY 自動投資軟件 Python 設計指南

## 摘要
- **目標**：開發針對 USD/JPY 外匯交易的自動投資軟件，利用 AI 模型進行價格預測、情緒分析、交易策略制定和風險管理，整合免費 API 提供實時與歷史數據。軟件支援回測、模擬交易和實時交易，強調模塊化設計以便擴展。
- **硬體**：Aoostar GT37（AMD Ryzen AI 9 HX 370，12 核 24 線程 CPU、Radeon 890M GPU、XDNA 2 NPU）。
- **市場特性**：USD/JPY 高波動性，對宏觀經濟數據（利率、通脹、GDP）敏感，需結合新聞/社群媒體情緒和技術指標分析。
- **焦點**：AI 模型選擇、API 整合、硬體優化、軟件架構、程式碼範例、測試指南和模型更新策略。

## 1. 軟件架構
軟件採用模塊化設計，遵循 MVC（Model-View-Controller）模式變體：資料模塊（Model）、AI 處理模塊（Controller）和報告模塊（View）。主要資料流：API → 資料預處理 → AI 模型 → 交易決策 → 日誌/報告。

### 主要模塊
- `data_acquisition.py`：處理 API 呼叫、資料獲取和預處理。
- `ai_models.py`：定義和訓練 AI 模型（價格預測、情緒分析等）。
- `trading_strategy.py`：整合模型輸出制定交易決策，使用 PPO 等。
- `risk_management.py`：計算止損/止盈、倉位大小，監控波動性。
- `main.py`：入口點，載入配置、硬體檢測、執行回測/實時模式。
- `utils.py`：公用函數，如錯誤處理、日誌、資料庫操作。
- `config/`：存放 JSON 配置檔。
- `reports/`：輸出報告檔（回測/實時）。
- `models/`：儲存訓練好的模型檔（ONNX 格式）。
- `tests/`：存放單元測試和回測測試。

### 資料流範例
1. 從 API 獲取資料 → 儲存到資料庫。
2. 預處理（計算指標、填補缺失值） → 輸入 AI 模型。
3. 模型輸出（預測、情緒分數） → 交易策略決策。
4. 執行模擬交易或實時交易 → 生成報告。

### 檔案結構
```
usd_jpy_trader/
├── main.py
├── data_acquisition.py
├── ai_models.py
├── trading_strategy.py
├── risk_management.py
│   test_data_acquisition.py
│   test_backtest.py
├── utils.py
├── config/
│   ├── system_config.json
│   └── api_key.json
├── models/
│   ├── lstm_model.onnx
│   ├── TimeSeriesTransformer.onnx
│   ├── FinBERT.onnx
│   ├── DistilBERT.onnx
│   ├── PPO.onnx
│   ├── XGBoost.onnx
│   └── LightGBM.onnx
├── reports/
│   ├── backtest_report_[日期]_[時間].csv
│   └── live_report_[日期]_[時間].csv
├── logs/
│   ├── backtest_log_[日期]_[時間].csv
│   └── live_log_[日期]_[時間].log
├── data/
│   └── trades.db
└── requirements.txt
```

## 2. AI 模型選擇
以下模型組合適用於 USD/JPY 的價格預測、情緒分析、交易策略和風險管理，優化於 HX 370 硬體。

| 功能         | 模型                     | 適用性                                       | HX 370 優化                              |
|--------------|--------------------------|----------------------------------------------|------------------------------------------|
| 價格預測     | LSTM/TimeSeriesTransformer | 捕捉時間序列模式，預測短期/中期走勢         | GPU（DirectML）加速矩陣運算；NPU（ONNX）實時推理 |
| 情緒分析     | FinBERT/DistilBERT       | 分析新聞/X 貼文，判斷央行政策影響           | CPU 批量文本處理；GPU 加速 BERT 推理     |
| 交易策略     | PPO（強化學習）          | 動態調整買賣決策，優化利潤/Sharpe Ratio    | GPU 加速訓練；NPU 低延遲決策            |
| 風險管理     | XGBoost/LightGBM         | 預測波動性，生成止損/止盈建議               | CPU 多線程訓練；GPU（LightGBM）加速      |

### 輸入資料
- **歷史價格**：OHLC（開盤、最高、最低、收盤）。
- **技術指標**：RSI、EMA、MACD、布林帶、ATR、ADX、Ichimoku Cloud、Stochastic Oscillator。
- **經濟數據**：聯邦基金利率、日本央行政策、GDP、通脹。
- **情緒來源**：新聞文章、X 貼文、央行聲明。

### 框架
- PyTorch/TensorFlow，搭配 Hugging Face Transformers。
- Stable-Baselines3（PPO），Pandas_TA/TA-Lib（技術指標）。
- ONNX 格式支援 DirectML 和 Vitis AI 優化。

### 模型訓練流程
- **數據分割**：80% 訓練集，10% 驗證集，10% 測試集。
- **損失函數**：MSE（價格預測）、CrossEntropy（情緒分析）。
- **超參數範圍**：
  - LSTM：層數（2-4）、隱藏單元（50-200）、學習率（0.0001-0.001）。
  - PPO：學習率（0.0003）、總步數（10,000-100,000）。

### 範例程式碼（ai_models.py）
```python
import torch
from torch import nn
from transformers import pipeline
from sklearn.model_selection import train_test_split
from torch.optim import Adam

class LSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train_lstm_model(df: pd.DataFrame, epochs: int = 50):
    X = df[['Close', 'RSI', 'MACD']].values
    y = df['Close'].shift(-1).dropna().values
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y, test_size=0.2)
    model = LSTMModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(output, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'models/lstm_model.pth')
    return model

# 情緒分析範例
sentiment_pipeline = pipeline('sentiment-analysis', model='finbert')
def predict_sentiment(text):
    return sentiment_pipeline(text)[0]['label']  # 返回 'positive', 'negative' 或 'neutral'

# 情緒分數整合
def integrate_sentiment(sentiment: str) -> float:
    if sentiment == 'positive':
        return 0.1  # 增加買入傾向
    elif sentiment == 'negative':
        return -0.1  # 增加賣出傾向
    return 0.0
```

## 3. API 選擇與應用
以下免費 API 提供 USD/JPY 數據，適用於 Python 3.10 環境。

| API          | 描述                                       | 適用性                                       | HX 370 優化                       | 限制                        | 設置範例                                                                 |
|--------------|--------------------------------------------|----------------------------------------------|-----------------------------------|-----------------------------|--------------------------------------------------------------------------|
| **yfinance** | 免費獲取 USD/JPY OHLC 數據，支持多時間框架 | 價格預測（OHLC 輸入 LSTM）；技術指標（搭配 Pandas_TA） | CPU 處理數據；GPU 加速指標計算 | 數據可能有延遲，偶有缺失   | `df = yf.download("USDJPY=X", start="2025-01-01", end="2025-08-25")`    |
| **fixer.io** | 170+ 貨幣歷史匯率，每小時更新（歐洲央行） | 回測（日收盤數據輸入 Transformer）           | CPU 處理回應；GPU 清理數據       | 免費 100 請求/月，無 OHLC  | `url = "https://data.fixer.io/api/timeseries?access_key=KEY&symbols=USDJPY"` |
| **Polygon.io** | 外匯 OHLC 數據，免費計劃有限             | 價格預測（OHLC 輸入模型）；交易策略（實時數據） | CPU/GPU 處理大規模數據          | 免費 5 次/分鐘             | `url = "https://api.polygon.io/v2/aggs/ticker/C:USDJPY/range/1/day/2023-08-23/2025-08-22?adjusted=true&sort=asc&apiKey=KEY"` |
| **FRED API** | 美國經濟數據（利率、GDP 等）              | 情緒分析（數據輸入 FinBERT）；風險管理       | CPU 批量處理；GPU 加速分析       | 更新頻率低（日/月）        | `url = "https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key=KEY"` |
| **X API**    | X 貼文數據，用於情緒分析                  | 情緒分析（貼文輸入 FinBERT）                | CPU 處理文本；GPU 加速推理       | 免費 10,000 次/月，僅公開貼文 | `url = "https://api.x.com/2/tweets/search/recent?query=USDJPY"`        |
| **investpy** | 經濟事件日曆，支援日期/國家篩選           | 情緒分析（事件輸入 FinBERT）；風險管理       | CPU 處理事件；GPU 加速分析       | 依網站速率，需遵守條款     | `investpy.get_economic_calendar(from_date="2025-01-01")`               |
| **FCS API**  | 外匯歷史/即時匯率，免費計劃有限          | 價格預測（歷史數據）；交易策略（即時匯率）   | CPU 處理數據；GPU 清理           | 免費 500 次/月，無技術指標 | `url = "https://fcsapi.com/api-v3/forex/history?symbol=USD/JPY&access_key=KEY"` |

### 備註
- **技術指標計算**：使用 Pandas_TA 計算技術指標：
  ```python
  import pandas_ta as ta
  import pandas as pd

  def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
      df['RSI'] = ta.rsi(df['Close'], length=14)
      df['MACD'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
      df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
      df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.bbands(df['Close'], length=20, std=2).iloc[:, :3]
      return df
  ```
- **實時交易**：建議使用 Interactive Brokers（IB）API（需帳戶），回測使用 yfinance 或 fixer.io。
- **資料預處理**：處理缺失值，使用前向填補：
  ```python
  df = df.fillna(method='ffill').fillna(method='bfill')
  ```
- **API 限額管理**：使用 Redis 快取減少 API 呼叫：
  ```python
  import redis
  import json
  from datetime import timedelta

  redis_client = redis.Redis(host='localhost', port=6379, db=0)

  def cache_data(key: str, data: dict, expiry: int = 3600):
      redis_client.setex(key, timedelta(seconds=expiry), json.dumps(data))

  def get_cached_data(key: str) -> dict:
      data = redis_client.get(key)
      return json.loads(data) if data else None
  ```

## 4. 技術指標應用
| 指標                  | 類型       | 應用                                                 | HX 370 優化                      |
|-----------------------|------------|------------------------------------------------------|----------------------------------|
| EMA/SMA               | 趨勢       | 價格預測（LSTM 輸入）；交易策略（交叉訊號）         | CPU 多線程；GPU 批量計算        |
| MACD                  | 趨勢       | 價格預測（Transformer 輸入）；風險管理（背離預測）   | GPU 加速矩陣運算                |
| Ichimoku Cloud        | 趨勢       | 交易策略（雲層突破輸入 PPO）                        | NPU 實時計算                    |
| RSI                   | 動量       | 價格預測（輔助情緒）；風險管理（極值止損）          | CPU 批量計算                    |
| Stochastic Oscillator | 動量       | 交易策略（%K/%D 交叉輸入 PPO）                     | GPU 加速多時間框                |
| Bollinger Bands       | 動量       | 風險管理（突破輸入 XGBoost）                        | NPU 實時監測                    |
| ATR                   | 波動性     | 風險管理（動態止損）；交易策略（高 ATR 避免入場）   | CPU 處理歷史數據                |
| ADX                   | 趨勢強度   | 交易策略（確認 PPO 趨勢）；價格預測（濾噪音）       | GPU 加速過濾                    |

## 5. 基本邏輯應用
### 配置
從 `config/system_config.json` 和 `api_key.json` 載入設置，使用環境變數儲存敏感資訊：
```python
import json
import os
from dotenv import load_dotenv

load_dotenv()
with open('config/system_config.json') as f:
    config = json.load(f)
api_key = os.getenv('POLYGON_API_KEY')  # 從環境變數讀取
```

### 代理
檢查是否需要 PROXY，動態提取代理資訊：
```python
import os

def setup_proxy():
    proxy = os.getenv('HTTP_PROXY')
    if proxy:
        os.environ['http_proxy'] = proxy
        os.environ['https_proxy'] = proxy
        print("代理已設置:", proxy)
    else:
        print("無代理設置")
```

### 模型更新策略
- **性能檢查**：使用滾動窗口（最近 1年數據）計算預測誤差，若 RMSE 超過閾值（0.01），觸發重新訓練。
- **數據分佈變化檢測**：使用 KS 檢驗比較新舊數據分佈。
  ```python
  from scipy.stats import ks_2samp
  import os

  def detect_drift(old_data: pd.DataFrame, new_data: pd.DataFrame, threshold: float = 0.05) -> bool:
      stat, p_value = ks_2samp(old_data['Close'], new_data['Close'])
      return p_value < threshold

  def update_model(df: pd.DataFrame, model_path: str = 'models/lstm_model.pth'):
      if not os.path.exists(model_path) or detect_drift(df.iloc[:-1000], df.iloc[-1000:]):
          model = train_lstm_model(df)
          torch.save(model.state_dict(), model_path)
          logging.info("模型已更新並保存至 %s", model_path)
      else:
          model = LSTMModel()
          model.load_state_dict(torch.load(model_path))
          logging.info("載入現有模型: %s", model_path)
      return model
  ```

### 資料庫
使用 `aiosqlite` 管理策略數據，定義表結構：
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME,
    symbol TEXT,
    price REAL,
    action TEXT,
    volume REAL,
    stop_loss REAL,
    take_profit REAL
);
```
範例程式碼：
```python
import aiosqlite
import pandas as pd
import asyncio

async def save_data(df: pd.DataFrame, db_path: str = "data/trades.db"):
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
        await df.to_sql("trades", db, if_exists="append", index=False)

async def load_data(db_path: str = "data/trades.db") -> pd.DataFrame:
    async with aiosqlite.connect(db_path) as db:
        return pd.read_sql("SELECT * FROM trades", db)
```

### 日誌
- 回測：`reports/backtest_log_[日期]_[時間].log`
- 實時：`reports/live_log_[日期]_[時間].log`
- 使用 logging 模塊記錄錯誤和決策：
  ```python
  import logging
  import datetime

  logging.basicConfig(
      filename=f'logs/app_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
  )
  logging.info("交易決策: 買入 USD/JPY")
  ```

### 交易報告
- 回測：`reports/backtest_report_[日期]_[時間].csv`
- 實時：`reports/live_report_[日期]_[時間].csv`
  ```

### py to txt小工具
- 制作一個獨立程式將本軟件的所有PY FILEs 內容合拼到一個TXT文件方便與GROK交流

### 錯誤處理
- **API 失敗**：延時 5 秒用備用 API 重試，價格數據優先順序：Polygon.io → yfinance → FCS API；情緒數據：X API → investpy。
  ```python
  import time
  import requests
  import logging

  def fetch_data(primary_url, backup_urls):
      try:
          response = requests.get(primary_url)
          response.raise_for_status()
          return response.json()
      except Exception as e:
          logging.error(f"Primary API 失敗: {e}")
          time.sleep(5)
          for backup in backup_urls:
              try:
                  response = requests.get(backup)
                  response.raise_for_status()
                  return response.json()
              except Exception as e:
                  logging.error(f"Backup API {backup} 失敗: {e}")
                  continue
          raise ValueError("所有 API 失敗")
  ```

### 邊緣案例處理
檢測高波動性事件（如黑天鵝），暫停交易：
```python
def check_volatility(atr: float, threshold: float = 0.02) -> bool:
    if atr > threshold:
        logging.warning("高波動事件檢測，暫停交易")
        return False
    return True
```

## 6. 交易執行邏輯
### 決策流程
1. 獲取最新資料和技術指標。
2. 運行價格預測和情緒分析。
3. 使用 PPO 產生行動（買、賣、持有）。
4. 應用風險管理：計算 ATR-based 止損/止盈，確定倉位大小。
5. 執行模擬交易或透過 IB API 執行，設訂單時間上限，混合使用市價單及限價單。

### Interactive Brokers API 整合
- **訂單類型**：
  - **限價單**：低波動期確保價格控制。
  - **止損單**：動態調整基於 ATR。
  - **括號訂單**：自動附加止損/止盈。
  - **OCA 群組**：多訂單互斥，避免過度暴露。
- **策略**：高波動期使用 OCA 結合止損單；低波動期使用限價單等待突破；最大暴露限額為資本的 5%。
- **範例程式碼（trading_strategy.py，需 pip install ib_insync）**：
  ```python
  from ib_insync import IB, Forex, LimitOrder, BracketOrder
  import logging

  def connect_ib(host='127.0.0.1', port=7497, client_id=1):
      ib = IB()
      ib.connect(host, port, client_id)
      return ib

  def execute_trade(ib, action: str, price: float, quantity: float, stop_loss: float, take_profit: float):
      contract = Forex('USDJPY')
      if action == "買入":
          order = BracketOrder('BUY', quantity, price, takeProfitPrice=take_profit, stopLossPrice=stop_loss)
      elif action == "賣出":
          order = BracketOrder('SELL', quantity, price, takeProfitPrice=stop_loss, stopLossPrice=take_profit)
      else:
          return
      trade = ib.placeOrder(contract, order)
      ib.sleep(1)  # 等待確認
      logging.info(f"訂單狀態: {trade.orderStatus.status}")
  ```

### 交易策略範例（trading_strategy.py）
```python
from stable_baselines3 import PPO
import gym
import numpy as np

class ForexEnv(gym.Env):
    def __init__(self, df, spread=0.0002):
        super().__init__()
        self.df = df
        self.spread = spread
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # 買、賣、持有
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self._get_obs()
    
    def _get_obs(self):
        return self.df[['Close', 'RSI', 'MACD']].iloc[self.current_step].values
    
    def step(self, action):
        price = self.df['Close'].iloc[self.current_step]
        reward = 0
        if action == 0:  # 買入
            reward -= self.spread
        elif action == 1:  # 賣出
            reward -= self.spread
        self.current_step += 1
        done = self.current_step >= len(self.df)
        return self._get_obs(), reward, done, {}

def train_ppo(env):
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=10000)
    model.save("models/ppo_model")
    return model

def make_decision(model, obs):
    action, _ = model.predict(obs)
    if action == 0:
        return "買入"
    elif action == 1:
        return "賣出"
    else:
        return "持有"
```

### 風險管理
- **止損/止盈**：
  ```python
  def calculate_stop_loss(current_price: float, atr: float, multiplier: float = 2) -> float:
      return current_price - (multiplier * atr)  # 適用於長倉

  def calculate_take_profit(current_price: float, atr: float, multiplier: float = 2) -> float:
      return current_price + (multiplier * atr)  # 適用於長倉
  ```
- **倉位大小**：
  ```python
  def calculate_position_size(capital: float, risk_percent: float, stop_loss_distance: float) -> float:
      return (capital * risk_percent) / stop_loss_distance
  ```

## 7. 硬體優化
- **環境**：Python 3.10，依賴清單：
  ```bash
  pip install torch-directml onnxruntime transformers stable-baselines3 pandas-ta lightgbm aiosqlite requests gym pytest python-dotenv redis streamlit prometheus-client ib_insync cryptography scipy yfinance pandas numpy torch sklearn
  ```
- **requirements.txt**：
  ```
  torch-directml
  onnxruntime
  transformers
  stable-baselines3
  pandas-ta
  lightgbm
  aiosqlite
  requests
  gym
  pytest
  python-dotenv
  redis
  streamlit
  prometheus-client
  ib_insync
  cryptography
  scipy
  yfinance
  pandas
  numpy
  torch
  sklearn
  ```
- **GPU（Radeon 890M）**：DirectML 加速模型訓練/推理。
- **NPU（XDNA 2）**：ONNX 模型量化至 INT8，實現低延遲決策。
- **CPU**：24 線程處理批量數據。
- **記憶體**：32GB LPDDR5X-8000MHz 支援大型模型。

### 硬體檢測
```python
import torch
import onnxruntime as ort
import logging

def check_hardware():
    # 檢測 GPU (DirectML)
    try:
        import torch_directml
        if torch_directml.is_available():
            gpu_device = torch_directml.device()
            logging.info("GPU (DirectML) 支援，使用 Radeon 890M 加速。")
        else:
            gpu_device = torch.device('cpu')
            logging.info("GPU (DirectML) 不支援，回退到 CPU。")
    except ImportError:
        gpu_device = torch.device('cpu')
        logging.info("torch_directml 未安裝，回退到 CPU。")
    # 檢測 NPU (XDNA 2 with ONNX)
    providers = ort.get_available_providers()
    if 'VitisAIExecutionProvider' in providers:
        logging.info("NPU (XDNA 2) 支援，使用 ONNX INT8 量化低延遲決策。")
        session = ort.InferenceSession('model.onnx', providers=['VitisAIExecutionProvider'])
    else:
        logging.info("NPU (XDNA 2) 不支援，回退到 CPU。")
        session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
    return gpu_device, session
```

### 多線程優化
```python
from concurrent.futures import ThreadPoolExecutor

def batch_process_data(data_list):
    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(compute_indicators, data_list))
    return results
```

## 8. 回測框架
- **回測邏輯**：模擬資金管理（初始資本、交易成本、滑點），計算績效指標（Sharpe Ratio、最大回撤、勝率、年化回報）。
- **範例程式碼**：
  ```python
  import pandas as pd
  import numpy as np

  def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
      return (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))

  def calculate_max_drawdown(equity_curve: pd.Series) -> float:
      return (equity_curve / equity_curve.cummax() - 1).min()

  def backtest(df: pd.DataFrame, strategy: callable, initial_capital: float = 10000, spread: float = 0.0002) -> dict:
      capital = initial_capital
      positions = []
      equity_curve = []
      for i, row in df.iterrows():
          action = strategy(row)
          if action == "買入":
              positions.append(row['Close'] + spread)
          elif action == "賣出" and positions:
              capital += (row['Close'] - positions.pop()) * 1000
          equity_curve.append(capital)
      returns = pd.Series(equity_curve).pct_change().dropna()
      return {
          "final_capital": capital,
          "sharpe_ratio": calculate_sharpe_ratio(returns),
          "max_drawdown": calculate_max_drawdown(pd.Series(equity_curve)),
          "win_rate": len([r for r in returns if r > 0]) / len(returns) if returns else 0
      }
  ```

## 9. 測試與部署
### 單元測試
使用 pytest 進行單元測試。
- **範例（tests/test_data_acquisition.py）**：
  ```python
  import pytest
  from data_acquisition import fetch_data

  def test_fetch_data():
      primary_url = "https://example.com"
      backup_urls = ["https://backup1.com", "https://backup2.com"]
      data = fetch_data(primary_url, backup_urls)
      assert isinstance(data, dict)
  ```
- **回測測試**：驗證 Sharpe Ratio > 0。
  ```python
  import pytest
  from trading_strategy import backtest

  def test_backtest():
      df = pd.DataFrame({'Close': [100, 101, 102], 'RSI': [50, 60, 70], 'MACD': [0, 0.1, 0.2]})
      result = backtest(df, lambda x: "買入" if x['RSI'] > 60 else "持有")
      assert result['sharpe_ratio'] > 0
  ```
- **黑天鵝測試**：
  ```python
  def test_high_volatility():
      df_high_vol = pd.DataFrame({'Close': [100, 120, 80], 'ATR': [0.03, 0.04, 0.05]})
      assert not check_volatility(df_high_vol['ATR'].mean())
  ```

### 部署
根目錄:"C:\Trading"

api_key.json:
{
  "fmp_api_key": "uUFGG1dA6jFHeFgf1XHNJpt4obL3uJfS",
  "fred_api_key": "81ffe6474ddf0bc54bb72e0f26918fcc",
  "x_bearer_token": "AAAAAAAAAAAAAAAAAAAAAIKJ3gEAAAAA0zJ2qC00iUuIIxFK70aMIN1zk5s%3D913jMzVxOA8azfWxs2unudDsAXF6OKjZDlIUn43pxrp94oZnps",
  "FCS_API_Key": "bN1WbcGvs5Bn9KjXxWkKwQPVoNsf1HJQ5",  
  "currencylayer_API_Key": "a013222877c20f54227a3f436c5e6031",
  "fixer_API_Key": "def5c1c73ccc267cf9edc754f61d3c0b",
  "exchangeratesio_API_Key": "4ed1a1e247b8c6cb4231f038deef375d",
  "exchangeratehost_API_Key": "cc83b43b2847ee6482d25b4273c0f32f",
  "marketstack_API_Key": "163ccf72c25454f66775951018ad6b8f",
  "twelvedata_API_Key": "583a65d8b04b47bda1cddec0c5a83a51",
  "polygon_api_key": "VWSmFPevufl8GGsAZC9VqbiW0xco0ZpO"
}
system_config.json:
{
    "data_source": "yfinance",
    "symbol": "USDJPY=X",
    "timeframe": "1d",
    "capital": 10000,
    "risk_percent": 0.01,
    "atr_threshold": 0.02,
    "root_dir": "C:\\Trading",
    "db_path": "C:\\Trading\\data\\trades.db",
    "min_backtest_days": 180,
  "proxies": {
        "http": "http://proxy1.scig.gov.hk:8080",
        "https": "http://proxy1.scig.gov.hk:8080"
  },
    "dependencies": [],
    "model_dir": "models",
    "model_periods": ["short_term", "medium_term", "long_term"]
}

使用 Docker 容器化：
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### 用戶介面
使用 Streamlit 提供 Web 儀表板：
```python
import streamlit as st
import pandas as pd

def display_dashboard(df: pd.DataFrame):
    st.title("USD/JPY 交易儀表板")
    st.line_chart(df['Close'])
    st.write("最新決策:", df['Action'].iloc[-1])
    st.write("最新技術指標:", df[['RSI', 'MACD', 'ATR']].iloc[-1])
    override_action = st.selectbox("手動覆寫決策", ["無", "買入", "賣出", "持有"])
    if override_action != "無":
        logging.info(f"用戶覆寫: {override_action}")
```

## 10. 限制與注意事項
- **API 限制**：免費 API 請求限額低（100-10,000 次/月），不適合高頻交易。
- **數據質量**：免費數據可能有延遲或缺失，需預處理。
- **法律合規**：
  - 遵守 API 條款和外匯監管（如 FCA、CFTC），避免過度爬蟲。
  - 合規檢查：
    ```python
    def compliance_check(trade: dict) -> bool:
        return trade['leverage'] <= 30  # 假設監管要求槓桿不超過 30:1
    ```
- **安全性**：
  - 使用 python-dotenv 儲存 API 密鑰：
    ```python
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv('POLYGON_API_KEY')
    ```
  - 加密敏感數據：
    ```python
    from cryptography.fernet import Fernet

    key = Fernet.generate_key()  # 儲存於安全處
    cipher = Fernet(key)
    encrypted_key = cipher.encrypt(api_key.encode())
    # 解密：decrypted = cipher.decrypt(encrypted_key).decode()
    ```
- **建議**：
  - 回測驗證模型，確保 Sharpe Ratio > 1.0，最大回撤 < 20%。
  - 結合 OANDA 或 IB API 執行實時交易。
  - 每月檢查模型，使用滾動窗口訓練。
  - 使用 Prometheus 和 Grafana 監控性能：
    ```yaml
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'usd_jpy_trader'
        static_configs:
          - targets: ['localhost:8000']
    ```
- **風險披露**：本軟件僅供教育用途，不構成投資建議。用戶需自行承擔風險。

## 11. 主程式範例（main.py）
```python
import asyncio
import logging
import pandas as pd
import streamlit as st
from data_acquisition import fetch_data, compute_indicators
from ai_models import LSTMModel, predict_sentiment, train_lstm_model, integrate_sentiment
from trading_strategy import train_ppo, make_decision, ForexEnv, connect_ib, execute_trade
from risk_management import calculate_stop_loss, calculate_position_size
from utils import check_hardware, update_model, check_volatility
from display import display_dashboard

async def main(mode: str = 'backtest'):
    logging.basicConfig(filename='logs/app.log', level=logging.INFO)
    gpu_device, onnx_session = check_hardware()
    # 獲取資料
    df = pd.DataFrame()  # 假設從 API 或資料庫獲取
    df = compute_indicators(df)
    await save_data(df)
    # 檢查波動性
    if not check_volatility(df['ATR'].mean()):
        return
    # 價格預測
    model = update_model(df, 'models/lstm_model.pth').to(gpu_device)
    input_data = df[['Close', 'RSI', 'MACD']].values[-10:]
    prediction = model(torch.tensor(input_data, dtype=torch.float32))
    # 情緒分析
    sentiment = predict_sentiment("USD/JPY expected to rise due to Fed policy")
    sentiment_score = integrate_sentiment(sentiment)
    # 交易決策
    env = ForexEnv(df)
    ppo_model = train_ppo(env)
    action = make_decision(ppo_model, input_data)
    # 風險管理
    atr = df['ATR'].iloc[-1]
    stop_loss = calculate_stop_loss(df['Close'].iloc[-1], atr)
    take_profit = calculate_take_profit(df['Close'].iloc[-1], atr)
    position_size = calculate_position_size(capital=10000, risk_percent=0.01, stop_loss_distance=df['Close'].iloc[-1] - stop_loss)
    logging.info(f"行動: {action}, 預測: {prediction}, 情緒: {sentiment}, 止損: {stop_loss}, 止盈: {take_profit}, 倉位大小: {position_size}")
    # 實時執行
    if mode == 'live':
        ib = connect_ib()
        execute_trade(ib, action, df['Close'].iloc[-1], position_size, stop_loss, take_profit)
    # 可視化
    if mode == 'live':
        display_dashboard(df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="USD/JPY 自動交易系統")
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest', help="運行模式")
    args = parser.parse_args()
    asyncio.run(main(args.mode))
```

## 12. 用戶指南
### 安裝步驟
1. 安裝 Python 3.10。
2. 運行 `pip install -r requirements.txt`。
3. 設定 .env 檔案（e.g., `POLYGON_API_KEY=your_key`）。
4. 運行 `python main.py --mode=backtest`。

### 運行模式
- **回測**：`--mode=backtest`（模擬歷史數據）。
- **實時**：`--mode=live`（需 IB 帳戶，啟動 Streamlit 儀表板）。

### 貢獻指南
- Fork 儲存庫，提交 PR。
- 測試覆蓋率 > 80%。