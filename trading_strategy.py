from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import pandas as pd
from ib_insync import IB, Forex, BracketOrder
import logging
from risk_management import check_resources, calculate_stop_loss, calculate_position_size
from utils import load_settings
from ai_models import FEATURES                     
class ForexEnv(gym.Env):
    """外匯環境：用於 PPO 強化學習訓練，支援多時間框架。"""
    # 類別說明：自定義 Gym 環境，用於模擬外匯交易，支援多時間框架觀察空間。
    def __init__(self, data_frames: dict, spread: float = 0.0002):
        super().__init__()
        tf_mapping = {'1 hour': '1h', '4 hours': '4h', '1 day': '1d'}
        self.data_frames = {tf_mapping.get(k, k): v for k, v in data_frames.items()}
        self.spread = spread
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3) # 買, 賣, 持
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        ) # 9 features × 3 timeframes 1h/4h/daily 的 5 個指標
    def reset(self):
        self.current_step = 0
        return self._get_obs()
    def _get_obs(self):
        # 關鍵邏輯：從多時間框架提取觀察值，若數據不足則填充 0。
        obs = []
        for tf in ['1h', '4h', '1d']:
            df = self.data_frames[tf]
            if self.current_step < len(df):
                obs.extend(df[FEATURES[:-1]].iloc[self.current_step].values)  # 排除fed_funds_rate如果不需要
            else:
                obs.extend([0] * 9) # 填充 0 以保持形狀一致
        return np.array(obs, dtype=np.float32)
    def step(self, action):
        price = self.data_frames['1h']['close'].iloc[self.current_step]
        reward = 0
        if action == 0: # 買
            reward -= self.spread
        elif action == 1: # 賣
            reward -= self.spread
        self.current_step += 1
        done = self.current_step >= min(len(self.data_frames[tf]) for tf in self.data_frames) - 1
        return self._get_obs(), reward, done, {}
def train_ppo(env, device_config: dict = None):
    """訓練 PPO：強化學習優化交易決策。，從 config 載入參數。
    邏輯：使用 MlpPolicy，根據配置的步數學習，保存模型。
    """
    # 函數說明：訓練 PPO 模型，用於強化學習決策優化。
    try:
        config = load_settings() # 載入配置
        total_timesteps = config.get('trading_params', {}).get('ppo_timesteps', 1000) # 從 config 獲取，若無則預設 1000
        learning_rate = config.get('trading_params', {}).get('ppo_learning_rate', 0.0003)# 從 config 獲取，若無則預設 0.0003
        device = device_config.get('ppo', torch.device('cpu')) if device_config else torch.device('cpu')
        logging.info(f"PPO 訓練：使用 total_timesteps={total_timesteps}, learning_rate={learning_rate}, device={device}")
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate, device=device)
        model.learn(total_timesteps=total_timesteps)
        model.save("models/ppo_model")
        logging.info("PPO 模型訓練完成")
        return model
    except Exception as e:
        logging.error(f"PPO 訓練錯誤: {e}")
        return None
def make_decision(model, data_frames: dict, sentiment: float) -> str:
    """產生決策：結合多時間框架技術指標和情緒分數。"""
    # 函數說明：結合技術指標、情緒分數和 PPO 模型產生買賣持倉決策。
    try:
        # 檢查系統資源
        if not check_resources():
            logging.warning("資源不足，暫停交易")
            return "持有"
        # 多框架技術指標邏輯
        buy_signals = []
        sell_signals = []
        for tf in ['1h', '4h', '1d']:
            df = data_frames[tf]
            if df.empty:
                continue
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            stoch_k = df['Stoch_k'].iloc[-1]
            adx = df['ADX'].iloc[-1]
            # 更新 Ichimoku 信號：考慮 Tenkan-sen 和 Kijun-sen
            ichimoku_buy = (df['Ichimoku_tenkan'].iloc[-1] > df['Ichimoku_kijun'].iloc[-1] and
                           df['close'].iloc[-1] > df['Ichimoku_cloud_top'].iloc[-1])
            ichimoku_sell = (df['Ichimoku_tenkan'].iloc[-1] < df['Ichimoku_kijun'].iloc[-1] and
                            df['close'].iloc[-1] < df['Ichimoku_cloud_top'].iloc[-1])
            bb_signal = df['close'].iloc[-1] < df['BB_lower'].iloc[-1]
            ema_signal = df['EMA_12'].iloc[-1] > df['EMA_26'].iloc[-1]
									  
            economic_impact = df['impact'].iloc[-1] if 'impact' in df.columns and not pd.isna(df['impact'].iloc[-1]) else 'Low'
            economic_pause = economic_impact in ['High', 'Medium']
            buy_signals.append(rsi < 30 and macd > macd_signal and stoch_k < 20 and adx > 25 and ichimoku_buy and bb_signal and ema_signal and not economic_pause)
            sell_signals.append(rsi > 70 and macd < macd_signal and stoch_k > 80 and adx > 25 and ichimoku_sell and df['close'].iloc[-1] > df['BB_upper'].iloc[-1] and df['EMA_12'].iloc[-1] < df['EMA_26'].iloc[-1] and not economic_pause)
        # 關鍵邏輯：計算買賣信號分數並根據情緒調整。
        buy_score = sum(1 for s in buy_signals if s) / len(buy_signals) if buy_signals else 0
        sell_score = sum(1 for s in sell_signals if s) / len(sell_signals) if sell_signals else 0
        # 情緒調整
        if abs(sentiment) > 0.8:
            logging.warning(f"極端情緒分數: {sentiment}，暫停交易")
            return "持有"
        sentiment_adjust = 0.2 if sentiment > 0.4 else -0.2 if sentiment < -0.4 else 0.0
        buy_score += sentiment_adjust
        sell_score -= sentiment_adjust
        # PPO 決策
        obs = []
        for tf in ['1h', '4h', '1d']:
            df = data_frames.get(tf, pd.DataFrame())
            if not df.empty:
                obs.extend(df[FEATURES[:-1]].iloc[-1].values)  # 排除fed_funds_rate
            else:
                obs.extend([0] * 9)
        action, _ = model.predict(np.array(obs, dtype=np.float32))
        ppo_action = ["買入", "賣出", "持有"][action]
        # 最終決策：結合多框架信號和 PPO
        if buy_score > 0.6 and ppo_action == "買入":
            return "買入"
        elif sell_score > 0.6 and ppo_action == "賣出":
            return "賣出"
        return "持有"
    except Exception as e:
        logging.error(f"決策錯誤: {e}")
        return "持有"
def backtest(df: pd.DataFrame, strategy: callable, initial_capital: float = 10000, spread: float = 0.0002) -> dict:
    """回測：模擬交易，計算績效指標，包含持倉管理。"""
    # 函數說明：模擬歷史數據上的交易決策，計算最終資本、夏普比率等指標。
    capital = initial_capital
    position_size = 0.0
    total_cost = 0.0
    entry_price = None
    trades = []
    equity_curve = []
    for i, row in df.iterrows():
        if not check_resources():
            logging.warning("資源不足，跳過交易")
            continue
        action = strategy(row)
        current_price = row['close']
        sentiment = row.get('sentiment', 0.0)
        atr = row['ATR']
        stop_loss_distance = abs(calculate_stop_loss(current_price, atr) - current_price)
        calc_position = calculate_position_size(initial_capital, 0.01, stop_loss_distance, sentiment)
        # 關鍵邏輯：處理買入/賣出決策，包括平倉和開倉。
        if action == "買入" and position_size <= 0:
            if position_size < 0: # 平空頭
                profit = (entry_price - current_price) * abs(position_size)
                capital += profit
                trades.append({
                    'date': row['date'],
                    'action': '平空',
                    'price': current_price,
                    'profit': profit,
                    'position_size': position_size
                })
            position_size = calc_position
            total_cost = current_price * position_size
            entry_price = current_price
            leverage_cost = abs(position_size) * 0.0001  # 假設 0.01% 融資成本
            capital -= leverage_cost
            trades.append({
                'date': row['date'],
                'action': '買入',
                'price': current_price,
                'profit': 0.0,
                'position_size': position_size
            })
        elif action == "賣出" and position_size >= 0:
            if position_size > 0: # 平多頭
                profit = (current_price - entry_price) * position_size
                capital += profit
                trades.append({
                    'date': row['date'],
                    'action': '平多',
                    'price': current_price,
                    'profit': profit,
                    'position_size': position_size
                })
            position_size = -calc_position
            total_cost = -current_price * abs(position_size)
            entry_price = current_price
            leverage_cost = abs(position_size) * 0.0001  # 假設 0.01% 融資成本
            capital -= leverage_cost
            trades.append({
                'date': row['date'],
                'action': '賣出',
                'price': current_price,
                'profit': 0.0,
                'position_size': position_size
            })
        equity_curve.append(capital + (current_price - entry_price) * position_size if position_size != 0 else capital)
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    return {
        "final_capital": capital,
        "sharpe_ratio": (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        "max_drawdown": (equity_series / equity_series.cummax() - 1).min(),
        "win_rate": len([r for r in returns if r > 0]) / len(returns) if len(returns) > 0 else 0,
        "trades": trades
    }
def connect_ib(host='127.0.0.1', port=7497, client_id=1):
    """連接 IB API：用於實時交易。
    邏輯：建立連接，返回 IB 物件。
    """
    # 函數說明：連接 Interactive Brokers API 用於實時交易。
    ib = IB()
    ib.connect(host, port, client_id)
    return ib
def execute_trade(ib, action: str, price: float, quantity: float, stop_loss: float, take_profit: float):
    """執行交易：使用括號訂單。
    邏輯：根據行動創建訂單，附加止損/止盈。
    """
    # 函數說明：使用 BracketOrder 執行買賣訂單，並附加止損和止盈。
    contract = Forex('USDJPY')
    for attempt in range(3):
        try:
            if action == "買入":
                order = BracketOrder('BUY', quantity, price, takeProfitPrice=take_profit, stopLossPrice=stop_loss)
            elif action == "賣出":
                order = BracketOrder('SELL', quantity, price, takeProfitPrice=stop_loss, stopLossPrice=take_profit)
            else:
                return
            trade = ib.placeOrder(contract, order)
            ib.sleep(1)
            if trade.orderStatus.status in ['Filled', 'Submitted']:
                logging.info(f"訂單狀態: {trade.orderStatus.status}")
                return
            else:
                logging.warning(f"訂單失敗: {trade.orderStatus.status}, 重試 {attempt + 1}/3")
        except Exception as e:
            logging.error(f"交易執行錯誤: {e}, 重試 {attempt + 1}/3")
        ib.sleep(2 ** attempt * 2)
    logging.error("交易執行失敗，超過最大重試次數")