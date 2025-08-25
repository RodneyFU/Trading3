import asyncio
import logging
import pandas as pd
import torch
import json
import os
from dotenv import load_dotenv
from data_acquisition import fetch_data, compute_indicators
from ai_models import LSTMModel, train_lstm_model, predict_sentiment, integrate_sentiment, update_model
from trading_strategy import ForexEnv, train_ppo, make_decision, backtest, connect_ib, execute_trade
from risk_management import calculate_stop_loss, calculate_take_profit, calculate_position_size, predict_volatility
from utils import check_hardware, setup_proxy, check_volatility, save_data, detect_drift
import streamlit as st
from datetime import datetime

# 載入配置檔
load_dotenv()
with open('config/system_config.json') as f:
    config = json.load(f)
with open('config/api_key.json') as f:
    api_keys = json.load(f)

# 設置日誌
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f'logs/app_{log_time}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
print("設定檔載入成功")  # 中文控制台輸出

async def main(mode: str = 'backtest'):
    """主程式入口：負責協調所有模塊，執行回測或實時模式。
    邏輯：載入配置 → 硬體檢測 → 獲取資料 → 模型更新 → 決策 → 執行/報告。
    """
    setup_proxy()
    gpu_device, onnx_session = check_hardware()
    
    # 獲取並預處理資料
    df = fetch_data(primary_api='yfinance', backup_apis=['polygon', 'fcs'])
    if df.empty:
        logging.error("Failed to fetch data")
        print("資料獲取失敗")
        return
    df = compute_indicators(df)
    await save_data(df)
    print("資料預處理完成")
    
    # 檢查波動性
    if not check_volatility(df['ATR'].mean()):
        logging.warning("High volatility detected, halting execution")
        print("偵測到高波動，暫停執行")
        return
    
    # 模型更新與訓練
    if detect_drift(df):  # 檢查數據漂移
        print("偵測到數據漂移，正在更新模型...")
    model = update_model(df, 'models/lstm_model.pth').to(gpu_device)
    
    # 情緒分析整合
    sentiment_text = "USD/JPY expected to rise due to Fed policy"  # 模擬，從 X API 獲取
    sentiment = predict_sentiment(sentiment_text)
    sentiment_score = integrate_sentiment(sentiment)
    print(f"情緒分析結果：{sentiment} (分數：{sentiment_score})")
    
    # 準備交易環境與 PPO
    env = ForexEnv(df)
    ppo_model = train_ppo(env)
    
    # 交易決策
    input_data = df[['Close', 'RSI', 'MACD']].values[-10:]
    action = make_decision(ppo_model, input_data)
    print(f"交易決策：{action}")
    
    # 風險管理（整合 XGBoost 預測波動）
    atr = df['ATR'].iloc[-1]
    predicted_vol = predict_volatility(df)  # 使用 XGBoost
    current_price = df['Close'].iloc[-1]
    stop_loss = calculate_stop_loss(current_price, atr)
    take_profit = calculate_take_profit(current_price, atr)
    position_size = calculate_position_size(config['capital'], config['risk_percent'], current_price - stop_loss)
    
    logging.info(f"Action: {action}, Stop Loss: {stop_loss}, Take Profit: {take_profit}, Position Size: {position_size}")
    
    # 回測模式
    if mode == 'backtest':
        print("正在執行回測...")
        result = backtest(df, lambda x: make_decision(ppo_model, x[['Close', 'RSI', 'MACD']].values), initial_capital=config['capital'])
        logging.info(f"Backtest Results: {result}")
        pd.DataFrame([result]).to_csv(f'reports/backtest_report_{log_time}.csv', index=False)
        print("回測完成，報告已生成")
    
    # 實時模式
    if mode == 'live':
        ib = connect_ib()
        execute_trade(ib, action, current_price, position_size, stop_loss, take_profit)
        st.title("USD/JPY 交易儀表板")
        st.line_chart(df['Close'])
        st.write("最新決策:", action)
        st.write("最新指標:", df[['RSI', 'MACD', 'ATR']].iloc[-1])
        print("實時模式執行中")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="USD/JPY 自動交易系統")
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest', help="運行模式")
    args = parser.parse_args()
    asyncio.run(main(args.mode))