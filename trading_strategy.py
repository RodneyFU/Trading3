from stable_baselines3 import PPO
import gym
import numpy as np
import pandas as pd
from ib_insync import IB, Forex, BracketOrder
import logging

class ForexEnv(gym.Env):
    """外匯環境：用於 PPO 強化學習訓練。
    邏輯：觀察狀態（Close, RSI, MACD），行動（買/賣/持），計算獎勵。
    """
    def __init__(self, df, spread=0.0002):
        super().__init__()
        self.df = df
        self.spread = spread
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # 買, 賣, 持
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self._get_obs()
    
    def _get_obs(self):
        return self.df[['Close', 'RSI', 'MACD']].iloc[self.current_step].values
    
    def step(self, action):
        price = self.df['Close'].iloc[self.current_step]
        reward = 0
        if action == 0:  # 買
            reward -= self.spread
        elif action == 1:  # 賣
            reward -= self.spread
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, {}

def train_ppo(env):
    """訓練 PPO：強化學習優化交易決策。
    邏輯：使用 MlpPolicy，學習 1000 步，保存模型。
    """
    try:
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
        model.learn(total_timesteps=1000)
        model.save("models/ppo_model")
        return model
    except Exception as e:
        logging.error(f"PPO 訓練錯誤: {e}")
        return None

def make_decision(model, obs):
    """產生決策：使用 PPO 預測行動。
    邏輯：輸入觀察，輸出買/賣/持。
    """
    try:
        action, _ = model.predict(obs)
        return ["買入", "賣出", "持有"][action]
    except Exception as e:
        logging.error(f"決策錯誤: {e}")
        return "持有"

def backtest(df: pd.DataFrame, strategy: callable, initial_capital: float = 10000, spread: float = 0.0002) -> dict:
    """回測：模擬交易，計算績效指標。
    邏輯：迭代數據，執行策略，計算 Sharpe、回撤等。
    """
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
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    return {
        "final_capital": capital,
        "sharpe_ratio": (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        "max_drawdown": (equity_series / equity_series.cummax() - 1).min(),
        "win_rate": len([r for r in returns if r > 0]) / len(returns) if len(returns) > 0 else 0
    }

def connect_ib(host='127.0.0.1', port=7497, client_id=1):
    """連接 IB API：用於實時交易。
    邏輯：建立連接，返回 IB 物件。
    """
    ib = IB()
    ib.connect(host, port, client_id)
    return ib

def execute_trade(ib, action: str, price: float, quantity: float, stop_loss: float, take_profit: float):
    """執行交易：使用括號訂單。
    邏輯：根據行動創建訂單，附加止損/止盈。
    """
    contract = Forex('USDJPY')
    if action == "買入":
        order = BracketOrder('BUY', quantity, price, takeProfitPrice=take_profit, stopLossPrice=stop_loss)
    elif action == "賣出":
        order = BracketOrder('SELL', quantity, price, takeProfitPrice=stop_loss, stopLossPrice=take_profit)
    else:
        return
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    logging.info(f"訂單狀態: {trade.orderStatus.status}")