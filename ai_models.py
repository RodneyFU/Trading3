import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from transformers import pipeline
import logging
from scipy.stats import ks_2samp
import os

class LSTMModel(nn.Module):
    """LSTM 模型：用於價格預測，捕捉時間序列模式。
    邏輯：輸入技術指標，輸出未來價格預測。
    """
    def __init__(self, input_size=3, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train_lstm_model(df: pd.DataFrame, epochs: int = 50):
    """訓練 LSTM：使用歷史數據訓練，保存模型。
    邏輯：數據分割 → 優化 MSE 損失 → 保存 PTH。
    """
    try:
        X = df[['Close', 'RSI', 'MACD']].values[:-1]
        y = df['Close'].shift(-1).dropna().values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LSTMModel()
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1))
            loss = criterion(output.squeeze(), torch.tensor(y_train, dtype=torch.float32))
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), 'models/lstm_model.pth')
        return model
    except Exception as e:
        logging.error(f"LSTM 訓練錯誤: {e}")
        return None

# 情緒分析（補充 FinBERT）
sentiment_pipeline = pipeline('sentiment-analysis', model='ProsusAI/finbert')
def predict_sentiment(text: str):
    """情緒分析：使用 FinBERT 分析文字情緒。
    邏輯：輸入新聞/貼文，輸出 positive/negative/neutral。
    """
    try:
        return sentiment_pipeline(text)[0]['label']
    except Exception as e:
        logging.error(f"情緒分析錯誤: {e}")
        return 'neutral'

def integrate_sentiment(sentiment: str) -> float:
    """整合情緒分數：轉換為決策調整值。
    邏輯：positive 增加買入傾向，negative 增加賣出。
    """
    if sentiment == 'positive':
        return 0.1
    elif sentiment == 'negative':
        return -0.1
    return 0.0

def detect_drift(old_data: pd.DataFrame, new_data: pd.DataFrame, threshold: float = 0.05) -> bool:
    """檢測數據漂移：使用 KS 檢驗比較分佈。
    邏輯：p-value < threshold 則需更新模型。
    """
    stat, p_value = ks_2samp(old_data['Close'], new_data['Close'])
    return p_value < threshold

def update_model(df: pd.DataFrame, model_path: str = 'models/lstm_model.pth'):
    """更新模型：檢查漂移或性能，重新訓練。
    邏輯：若存在漂移或無模型，訓練新模型；否則載入。
    """
    old_data = df.iloc[:-1000] if len(df) > 1000 else df
    new_data = df.iloc[-1000:]
    if not os.path.exists(model_path) or detect_drift(old_data, new_data):
        model = train_lstm_model(df)
        logging.info("模型已更新")
    else:
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path))
        logging.info("載入現有模型")
    return model