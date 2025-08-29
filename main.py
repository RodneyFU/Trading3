import asyncio
import logging
import pandas as pd
import torch
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from data_acquisition import fetch_data, compute_indicators, fetch_economic_calendar
from ai_models import update_model, predict_sentiment, integrate_sentiment
from trading_strategy import ForexEnv, train_ppo, make_decision, backtest, connect_ib, execute_trade
from risk_management import calculate_stop_loss, calculate_take_profit, calculate_position_size, predict_volatility
from utils import check_hardware, setup_proxy, check_volatility, save_data, save_periodically, initialize_db, load_settings, decrypt_key
import streamlit as st
from prometheus_client import Counter, Histogram
from pathlib import Path
# Prometheus 指標，用於監控交易次數和 API 延遲
trade_counter = Counter('usd_jpy_trades_total', 'Total number of trades executed', ['action', 'mode'])
api_latency = Histogram('usd_jpy_api_latency_seconds', 'API call latency', ['mode'])
# 結構化 JSON 日誌格式，方便後續分析
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'filename': record.filename,
            'funcName': record.funcName,
            'mode': getattr(record, 'mode', 'unknown')
        }
        return json.dumps(log_data, ensure_ascii=False)
# 配置日誌，區分回測和實時模式
def setup_logging(mode: str):
    """設置日誌：根據模式（回測/實時）創建不同的日誌檔案，並使用 JSON 格式。
    邏輯：每天輪替日誌檔案，保留 7 天備份，確保日誌結構化且易於解析。
    """
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'app_{mode}_{log_time}.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight', backupCount=7)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 日誌設定完成")
    logging.info(f"Logging setup completed for mode: {mode}", extra={'mode': mode})
# 清理舊備份檔案
def clean_old_backups(root_dir: str, days_to_keep: int = 7):
    """清理舊備份檔案：僅保留指定天數的資料庫備份檔案。
    邏輯：遍歷備份目錄，刪除早於指定天數的檔案，確保磁碟空間不被過度佔用。
    """
    backup_dir = Path(root_dir) / 'backups'
    if not backup_dir.exists():
        return
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    for backup_file in backup_dir.glob('trading_data_*.db'):
        file_date = datetime.strptime(backup_file.stem.split('_')[-1], '%Y%m%d')
        if file_date < cutoff_date:
            backup_file.unlink()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 舊備份已刪除")
            logging.info(f"Deleted old backup file: {backup_file}", extra={'mode': 'cleanup'})
# 載入配置檔並設置環境變數
def load_config():
    """載入配置檔：從環境變數和 JSON 檔案載入配置，確保安全性。
    邏輯：優先從環境變數載入加密密鑰，然後解密 API 密鑰，確保敏感資訊不硬編碼。
    """
    load_dotenv()
    config = load_settings()
    api_key = config.get('api_key', {})
    for k in api_key:
        if isinstance(api_key[k], bytes):
            api_key[k] = decrypt_key(api_key[k])
    system_config = config.get('system_config', {})
    trading_params = config.get('trading_params', {})
    fernet_key = api_key.get('FERNET_KEY', '')
    if not fernet_key:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 環境變數未設置")
        logging.error("FERNET_KEY environment variable not set", extra={'mode': 'config'})
        raise ValueError("FERNET_KEY environment variable not set")
    return config, api_key, system_config, trading_params
async def main(mode: str = 'backtest'):
    """主程式入口：協調資料獲取、模型訓練、交易決策和風險管理。
    參數：
        mode: 'backtest' 或 'live'，決定運行回測或實時交易模式。
    邏輯：
        1. 設置日誌和代理，初始化資料庫。
        2. 獲取多時間框架資料和經濟日曆。
        3. 檢查波動性，更新模型，進行情緒分析。
        4. 根據模式執行回測或實時交易。
        5. 定期保存數據並清理舊備份。
    """
    # 設置日誌
    setup_logging(mode)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 程式啟動中")
    logging.info(f"Starting program in {mode} mode", extra={'mode': mode})
    # 載入配置
    try:
        config, api_key, system_config, trading_params = load_config()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定載入成功")
        logging.info("Configuration loaded successfully", extra={'mode': mode})
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定載入失敗")
        logging.error(f"Failed to load configuration: {str(e)}", extra={'mode': mode})
        return
    # 設置代理
    setup_proxy()
    device_config, onnx_session = check_hardware()
    db_path = system_config['db_path']
    await initialize_db(db_path)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫初始化完成")
    logging.info("Database initialized", extra={'mode': mode})
    # 定義日期範圍
    date_range = {
        'start': (datetime.now() - pd.Timedelta(days=trading_params['min_backtest_days'])).strftime('%Y-%m-%d'),
        'end': datetime.now().strftime('%Y-%m-%d')
    }
    # 獲取多時間框架資料
    timeframes = ['1h', '4h', '1d'] # 修正時間框架
    data_frames = {}
    tasks = []
    for tf in timeframes:
        start_time = time.time()
        print(f"獲取 {tf} 資料中")
        df = await fetch_data(
            primary_api=system_config['data_source'],
            backup_apis=['yfinance', 'fcs'],
            date_range=date_range,
            timeframe=tf,
            db_path=db_path,
            config=config
        )
        api_latency.labels(mode=mode).observe(time.time() - start_time)
        if df.empty:
            print(f"{tf} 資料獲取失敗")
            logging.error(f"Failed to fetch {tf} data", extra={'mode': mode})
            for task in tasks:
                task.cancel()
            return
        # 傳遞 db_path, timeframe, config 給 compute_indicators，讓其內部存入 DB 和 CSV
        df = await compute_indicators(df, db_path, tf, config)
        data_frames[tf] = df
        print(f"{tf} 資料處理完成")
        logging.info(f"{tf} data preprocessing completed", extra={'mode': mode})

    # 日期對齊：確保所有時間框架的數據日期範圍一致
    common_dates = None
    for tf in timeframes:
        if not data_frames[tf].empty:
            dates = set(data_frames[tf]['date'])
            common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if common_dates:
        for tf in timeframes:
            if not data_frames[tf].empty:
                data_frames[tf] = data_frames[tf][data_frames[tf]['date'].isin(common_dates)].copy()
                logging.info(f"Aligned {tf} data to common dates, rows={len(data_frames[tf])}")
    
    # 獲取經濟日曆
    economic_calendar = await fetch_economic_calendar(date_range, db_path, config)
    if not economic_calendar.empty:
        logging.info("Economic calendar is not empty", extra={'mode': mode})
        data_frames['1d'] = data_frames['1d'].merge(
            economic_calendar[['date', 'event', 'impact', 'fed_funds_rate']], on='date', how='left'
        )
        data_frames['1d']['impact'] = data_frames['1d']['impact'].fillna('Low')
    # 啟動定期儲存任務
    for tf in timeframes:
        if not data_frames[tf].empty:
            tasks.append(asyncio.create_task(save_periodically(data_frames[tf], tf, db_path, system_config['root_dir'], data_type='ohlc')))
            tasks.append(asyncio.create_task(save_periodically(data_frames[tf], tf, db_path, system_config['root_dir'], data_type='indicators')))
    if not economic_calendar.empty:
        tasks.append(asyncio.create_task(save_periodically(economic_calendar, '1d', db_path, system_config['root_dir'], data_type='economic')))
    # 清理舊備份
    clean_old_backups(system_config['root_dir'])
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 備份清理完成")
    logging.info("Old backup files cleaned", extra={'mode': mode})
    # 檢查波動性
    if '1h' not in data_frames or data_frames['1h'].empty or 'ATR' not in data_frames['1h'].columns:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 1小时数据或ATR列缺失，使用默认会话模式 'normal'")
        logging.warning("1h data or ATR column missing, defaulting to 'normal' session", extra={'mode': mode})
        session = 'normal'
    else:
        if not check_volatility(data_frames['1h']['ATR'].mean(), threshold=trading_params['atr_threshold']):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 高波动，暂停执行")
            logging.warning("High volatility detected, halting execution", extra={'mode': mode})
            for task in tasks:
                task.cancel()
            return
    # 模型更新與訓練
    session = 'high_volatility' if data_frames['1h']['ATR'].iloc[-1] > trading_params['atr_threshold'] else 'normal'
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 更新模型中")
    models = update_model(data_frames['1d'], 'models', session, device_config)
    if not models:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 模型更新失敗")
        logging.error("Model update failed", extra={'mode': mode})
        for task in tasks:
            task.cancel()
        return
    # 情緒分析整合
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 進行情緒分析中")
    sentiment_score = await predict_sentiment(date_range['end'], db_path, config)
    sentiment_adjustment = integrate_sentiment(sentiment_score)
    print(f"情緒分析完成，分數={sentiment_score:.2f}")
    logging.info(f"Sentiment analysis result: score={sentiment_score}, adjustment={sentiment_adjustment}", extra={'mode': mode})
    # 準備交易環境與 PPO
    env = ForexEnv(data_frames)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 訓練 PPO 模型中")
    ppo_model = train_ppo(env, device_config)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PPO 訓練完成")
    logging.info("PPO model training completed", extra={'mode': mode})
    # 交易決策
    action = make_decision(ppo_model, data_frames, sentiment_score)
    print(f"交易決策：{action}")
    logging.info(f"Trading decision: {action}", extra={'mode': mode})
    # 風險管理
    if '1h' not in data_frames or data_frames['1h'].empty or 'ATR' not in data_frames['1h'].columns or 'close' not in data_frames['1h'].columns:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 1小时数据或必要列缺失，无法进行风险管理")
        logging.error("1h data or required columns missing, cannot proceed with risk management", extra={'mode': mode})
        for task in tasks:
            task.cancel()
        return
    atr = data_frames['1h']['ATR'].iloc[-1]
    predicted_vol = predict_volatility(data_frames['1h'], model_path='models/volatility_model.pkl')
    current_price = data_frames['1h']['close'].iloc[-1]
    stop_loss = calculate_stop_loss(current_price, atr, action)
    take_profit = calculate_take_profit(current_price, atr, action)
    position_size = await calculate_position_size(trading_params['capital'], trading_params['risk_percent'], current_price - stop_loss, sentiment_score, db_path)
    logging.info(f"Action: {action}, Stop Loss: {stop_loss}, Take Profit: {take_profit}, Position Size: {position_size}", extra={'mode': mode})
    trade = {'action': action, 'price': current_price, 'quantity': position_size, 'stop_loss': stop_loss, 'take_profit': take_profit, 'leverage': 1}
    if not compliance_check(trade):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 交易不符合規定")
        logging.warning("Trade does not comply with leverage limits", extra={'mode': mode})
        for task in tasks:
            task.cancel()
        return
    # 根據模式執行回測或實時交易
    if mode == 'backtest':
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 執行回測中")
        result = backtest(data_frames['1d'], lambda x: make_decision(ppo_model, data_frames, sentiment_score), initial_capital=trading_params['capital'])
        logging.info(f"Backtest Results: {result}", extra={'mode': mode})
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        pd.DataFrame([result]).to_csv(report_dir / f'backtest_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 回測完成")
        logging.info("Backtest completed, report generated", extra={'mode': mode})
    elif mode == 'live':
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 啟動實時交易")
        ib = connect_ib()
        trade_counter.labels(action=action, mode=mode).inc()
        execute_trade(ib, action, current_price, position_size, stop_loss, take_profit)
        st.title("USD/JPY 交易儀表板")
        st.line_chart(data_frames['1h']['close'])
        st.write("最新決策:", action)
        st.write("最新指標:", data_frames['1h'][['RSI', 'MACD', 'Stoch_k', 'ADX', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26']].iloc[-1])
        override_action = st.selectbox("手動覆寫決策", ["無", "買入", "賣出", "持有"])
        if override_action != "無":
            action = override_action
            logging.info(f"User overridden decision: {action}", extra={'mode': mode})
            trade_counter.labels(action=action, mode=mode).inc()
            execute_trade(ib, action, current_price, position_size, stop_loss, take_profit)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 實時交易執行中")
        logging.info("Live trading mode running", extra={'mode': mode})
    # 清理任務
    for task in tasks:
        task.cancel()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 程式執行完畢")
    logging.info("Program execution completed", extra={'mode': mode})
def compliance_check(trade: dict) -> bool:
    """檢查交易是否符合槓桿限制。
    邏輯：確保槓桿不超過 30:1，符合監管要求。
    """
    leverage = trade.get('leverage', 1)
    is_compliant = leverage <= 30
    logging.info(f"Leverage check: leverage={leverage}, compliant={is_compliant}", extra={'mode': 'compliance'})
    return is_compliant
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="USD/JPY 自動交易系統")
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest', help="運行模式：回測或實時交易")
    args = parser.parse_args()
    asyncio.run(main(args.mode))