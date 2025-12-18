"""
SellEnv - Sell Knowledge RL 環境

根據論文設計:
- 狀態空間: 70 維 (69 個正規化特徵 + SellReturn)
- 動作空間: 2 (Action 1: 持有, Action 2: 賣出)
- 獎勵函數: 基於相對排名的獎勵 (鼓勵賣在高點)
- 訓練資料: 只使用最終達到 ≥10% 報酬的買入訊號
- 最大持有天數: 120 天
- 賣出條件: |sell_prob - hold_prob| > 0.85 且 sell_prob > hold_prob
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from loguru import logger


class SellEnv(gym.Env):
    """
    Sell Knowledge RL 環境
    
    角色:
        在買入後的 120 天內，決定何時賣出以最大化報酬
    
    狀態 (70 維):
        - 69 個正規化技術特徵
        - SellReturn: 當前價格 / 買入價格 (公式 20)
    
    動作 (2 個):
        - Action 0: 持有 (Hold)
        - Action 1: 賣出 (Sell)
    
    獎勵:
        基於賣出時機的相對排名獎勵:
        - 賣在最高點: +1 到 +2
        - 賣在最低點: -1 到 0
        - 持有超過 120 天: 強制賣出
    
    使用方式:
        env = SellEnv(trade_data, config)
        obs, info = env.reset()
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 trade_data: Dict[str, pd.DataFrame],
                 config: dict = None):
        """
        初始化 SellEnv
        
        Args:
            trade_data: 交易資料字典
                {
                    'symbol': {
                        'features': pd.DataFrame (每日特徵),
                        'buy_date': pd.Timestamp (買入日期),
                        'buy_price': float (買入價格)
                    },
                    ...
                }
            config: 設定字典
        """
        super().__init__()
        
        config = config or {}
        self.max_holding_days = config.get('max_holding_days', 120)
        self.success_threshold = config.get('success_threshold', 0.10)
        self.sell_threshold = config.get('sell_threshold', 0.85)
        
        # 將交易資料轉換為訓練用序列
        self.episodes = self._prepare_episodes(trade_data)
        
        # 取得特徵欄位名稱
        if self.episodes:
            first_episode = self.episodes[0]
            self.feature_cols = [col for col in first_episode['features'].columns 
                                if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
        else:
            self.feature_cols = []
        
        # 定義狀態空間 (70 維 = 69 特徵 + SellReturn)
        n_features = len(self.feature_cols) + 1  # +1 for SellReturn
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features,), 
            dtype=np.float32
        )
        
        # 定義動作空間 (0: 持有, 1: 賣出)
        self.action_space = spaces.Discrete(2)
        
        # 內部狀態
        self.current_episode_idx = 0
        self.current_episode = None
        self.current_day = 0
        self.buy_price = 0.0
        self.holding_days = 0
        
        logger.info(f"SellEnv 初始化完成 - {len(self.episodes)} 個 episodes, {n_features} 維特徵")
    
    def _prepare_episodes(self, trade_data: Dict) -> List[Dict]:
        """
        準備訓練用 episodes
        
        每個 episode 代表一筆交易 (從買入到賣出)
        只使用最終達到 ≥10% 報酬的交易
        """
        episodes = []
        
        for key, data in trade_data.items():
            features = data.get('features')
            buy_date = data.get('buy_date')
            buy_price = data.get('buy_price')
            
            if features is None or buy_price is None:
                continue
            
            # 計算該交易的最終報酬率
            if 'Close' in features.columns:
                # 找到 120 天內的最高價
                holding_period = features.iloc[:self.max_holding_days] if len(features) > self.max_holding_days else features
                max_price = holding_period['Close'].max()
                max_return = (max_price - buy_price) / buy_price
                
                # 只使用成功的交易 (論文要求)
                if max_return >= self.success_threshold:
                    episodes.append({
                        'key': key,
                        'features': features,
                        'buy_price': buy_price,
                        'buy_date': buy_date,
                        'max_return': max_return
                    })
        
        logger.info(f"準備了 {len(episodes)} 個成功交易 episodes")
        return episodes
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        重置環境，開始新的 episode
        """
        super().reset(seed=seed)
        
        if len(self.episodes) == 0:
            raise ValueError("沒有可用的訓練 episode")
        
        # 選擇下一個 episode
        self.current_episode_idx = (self.current_episode_idx + 1) % len(self.episodes)
        self.current_episode = self.episodes[self.current_episode_idx]
        
        # 重置狀態
        self.current_day = 0
        self.buy_price = self.current_episode['buy_price']
        self.holding_days = 0
        
        # 取得初始觀察
        obs = self._get_observation()
        
        info = {
            'episode_idx': self.current_episode_idx,
            'buy_price': self.buy_price,
            'max_return': self.current_episode['max_return']
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        執行動作
        
        Args:
            action: 0 (持有) 或 1 (賣出)
        """
        features = self.current_episode['features']
        
        # 取得當前價格
        if self.current_day < len(features):
            current_close = features.iloc[self.current_day]['Close']
        else:
            current_close = features.iloc[-1]['Close']
        
        # 計算當前報酬率
        current_return = (current_close - self.buy_price) / self.buy_price
        
        # 計算賣出時機的相對排名
        sell_return = current_return
        
        # 檢查是否達到最大持有天數
        self.holding_days += 1
        force_sell = self.holding_days >= self.max_holding_days
        
        # 決定是否結束
        terminated = (action == 1) or force_sell
        truncated = False
        
        # 計算獎勵
        if terminated:
            reward = self._calculate_reward(sell_return, features)
        else:
            reward = 0.0  # 持有期間無獎勵
        
        # 移動到下一天
        self.current_day += 1
        
        # 取得下一個觀察
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            'action': action,
            'current_return': current_return,
            'holding_days': self.holding_days,
            'force_sell': force_sell,
            'sell_return': sell_return if terminated else None
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """取得當前觀察 (70 維)"""
        features = self.current_episode['features']
        
        if self.current_day >= len(features):
            return np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
        
        # 取得 69 個特徵
        row = features.iloc[self.current_day]
        feature_values = row[self.feature_cols].values.astype(np.float32)
        
        # 計算 SellReturn (公式 20)
        current_open = row.get('Open', row.get('Close', self.buy_price))
        sell_return = current_open / self.buy_price
        
        # 合併為 70 維
        obs = np.concatenate([feature_values, [sell_return]])
        
        # 處理 NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _calculate_reward(self, sell_return: float, features: pd.DataFrame) -> float:
        """
        計算獎勵
        
        基於論文的相對排名獎勵:
        - 計算該交易期間所有可能賣出點的報酬率
        - 根據當前賣出點的排名給予獎勵
        - 賣在最高點得最高獎勵
        """
        holding_period = features.iloc[:self.max_holding_days] if len(features) > self.max_holding_days else features
        
        # 計算所有可能的賣出報酬率
        all_returns = (holding_period['Close'] - self.buy_price) / self.buy_price
        
        if len(all_returns) == 0:
            return 0.0
        
        # 計算排名 (0-1)
        rank = (all_returns <= sell_return).mean()
        
        # 轉換為獎勵 (-1 到 +2)
        # rank=1 (最高) -> reward=2
        # rank=0.5 (中間) -> reward=0.5
        # rank=0 (最低) -> reward=-1
        reward = 3 * rank - 1
        
        return float(reward)
    
    def get_state_dim(self) -> int:
        """取得狀態維度 (70)"""
        return len(self.feature_cols) + 1


class SellEnvSimple(gym.Env):
    """
    簡化版 SellEnv (用於單一股票的訓練)
    
    輸入: 單一股票的日線資料
    每個 episode: 一次完整的買入-持有-賣出過程
    """
    
    def __init__(self, 
                 daily_features: pd.DataFrame,
                 buy_signals: pd.Series,
                 config: dict = None):
        """
        Args:
            daily_features: 每日特徵資料 (正規化後)
            buy_signals: 買入訊號 (True/False Series)
            config: 設定
        """
        super().__init__()
        
        config = config or {}
        self.max_holding_days = config.get('max_holding_days', 120)
        self.success_threshold = config.get('success_threshold', 0.10)
        
        self.features = daily_features
        self.buy_signals = buy_signals
        
        # 找出所有成功的買入點
        self.successful_entries = self._find_successful_entries()
        
        # 特徵欄位
        self.feature_cols = [col for col in daily_features.columns 
                             if '_norm' in col or col in ['Return', 'Index_Return', 'SuperTrend_14', 'SuperTrend_21']]
        
        # 狀態空間 (70 維)
        n_features = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        
        # 動作空間
        self.action_space = spaces.Discrete(2)
        
        # 狀態
        self.current_entry_idx = 0
        self.current_day = 0
        self.buy_price = 0.0
        self.buy_idx = 0
        
        logger.info(f"SellEnvSimple 初始化 - {len(self.successful_entries)} 個成功入場點")
    
    def _find_successful_entries(self) -> List[int]:
        """找出所有成功的買入點索引"""
        entries = []
        signal_indices = self.buy_signals[self.buy_signals == True].index.tolist()
        
        for idx in signal_indices:
            if idx not in self.features.index:
                continue
            
            loc = self.features.index.get_loc(idx)
            buy_price = self.features.loc[idx, 'Close'] if 'Close' in self.features.columns else 0
            
            if buy_price <= 0:
                continue
            
            # 檢查後續 120 天內是否達到 10%
            end_loc = min(loc + self.max_holding_days, len(self.features))
            future_prices = self.features.iloc[loc:end_loc]['Close'] if 'Close' in self.features.columns else pd.Series()
            
            if len(future_prices) > 0:
                max_return = (future_prices.max() - buy_price) / buy_price
                if max_return >= self.success_threshold:
                    entries.append(loc)
        
        return entries
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置環境"""
        super().reset(seed=seed)
        
        if len(self.successful_entries) == 0:
            raise ValueError("沒有成功的入場點")
        
        # 隨機選擇入場點
        self.current_entry_idx = np.random.randint(len(self.successful_entries))
        self.buy_idx = self.successful_entries[self.current_entry_idx]
        self.current_day = 0
        self.buy_price = self.features.iloc[self.buy_idx]['Close']
        
        obs = self._get_observation()
        info = {'buy_idx': self.buy_idx, 'buy_price': self.buy_price}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """執行動作"""
        current_idx = self.buy_idx + self.current_day
        
        if current_idx >= len(self.features):
            return np.zeros(len(self.feature_cols) + 1, dtype=np.float32), 0.0, True, False, {}
        
        current_price = self.features.iloc[current_idx]['Close']
        current_return = (current_price - self.buy_price) / self.buy_price
        
        self.current_day += 1
        force_sell = self.current_day >= self.max_holding_days
        terminated = (action == 1) or force_sell
        
        if terminated:
            # 計算相對排名獎勵
            end_idx = min(self.buy_idx + self.max_holding_days, len(self.features))
            all_prices = self.features.iloc[self.buy_idx:end_idx]['Close']
            all_returns = (all_prices - self.buy_price) / self.buy_price
            rank = (all_returns <= current_return).mean()
            reward = 3 * rank - 1
        else:
            reward = 0.0
        
        obs = self._get_observation() if not terminated else np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
        
        info = {
            'current_return': current_return,
            'holding_days': self.current_day,
            'force_sell': force_sell
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """取得觀察"""
        current_idx = self.buy_idx + self.current_day
        
        if current_idx >= len(self.features):
            return np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
        
        row = self.features.iloc[current_idx]
        features = row[self.feature_cols].values.astype(np.float32) if self.feature_cols else np.array([])
        
        current_open = row.get('Open', row.get('Close', self.buy_price))
        sell_return = current_open / self.buy_price if self.buy_price > 0 else 1.0
        
        obs = np.concatenate([features, [sell_return]]) if len(features) > 0 else np.array([sell_return], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    print("SellEnv 模組載入成功")
    print("SellEnv: 70 維狀態 (69 特徵 + SellReturn)")
    print("動作: 0=持有, 1=賣出")
    print("獎勵: 基於相對排名 (-1 到 +2)")
